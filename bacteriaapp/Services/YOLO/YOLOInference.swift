//
//  YOLOInference.swift
//  bacteriaapp
//
//  Created by Regina Celine Adiwinata on 06/08/26.
//

@preconcurrency import CoreML
import CoreGraphics
import Foundation

nonisolated struct BoundingBox: @unchecked Sendable, Identifiable {
    let id = UUID()
    /// Rect ternormalisasi, origin kiri-bawah, x/y/width/height dalam 0...1.
    let normalizedRect: CGRect
    let label: String
    let confidence: Float

    func rect(in imageSize: CGSize) -> CGRect {
        let width = normalizedRect.width * imageSize.width
        let height = normalizedRect.height * imageSize.height
        let x = normalizedRect.minX * imageSize.width
        let y = (1 - normalizedRect.minY - normalizedRect.height) * imageSize.height
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

nonisolated enum YOLOInferenceError: LocalizedError {
    case modelNotFound
    case resizeFailed
    case pixelBufferCreationFailed
    case invalidOutputShape
    case detectionFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "YOLODishDetector.mlmodelc was not found in the app bundle."
        case .resizeFailed:
            "The cropped dish image could not be resized for inference."
        case .pixelBufferCreationFailed:
            "The image could not be converted for model input."
        case .invalidOutputShape:
            "Unexpected confidence/coordinates output shape."
        case .detectionFailed(let message):
            "YOLO inference failed: \(message)"
        }
    }
}

actor YOLODetector {
    static let inputSize = 1024

    private let confidenceThreshold: Double = 0.25
    private let iouThreshold: Double = 0.45

    private var model: MLModel?

    /// Menjalankan deteksi bakteri pada gambar dish yang sudah di-crop.
    func detect(in croppedImage: CGImage) async throws -> [BoundingBox] {
        guard let resized = Self.resize(croppedImage, to: Self.inputSize) else {
            throw YOLOInferenceError.resizeFailed
        }
        guard let pixelBuffer = Self.pixelBuffer(from: resized, size: Self.inputSize) else {
            throw YOLOInferenceError.pixelBufferCreationFailed
        }

        let model = try await loadModel()

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixelBuffer),
            "iouThreshold": MLFeatureValue(double: iouThreshold),
            "confidenceThreshold": MLFeatureValue(double: confidenceThreshold)
        ])

        let output: MLFeatureProvider
        do {
            output = try await model.prediction(from: input)
        } catch {
            throw YOLOInferenceError.detectionFailed(error.localizedDescription)
        }

        guard let confidenceArray = output.featureValue(for: "confidence")?.multiArrayValue,
              let coordinatesArray = output.featureValue(for: "coordinates")?.multiArrayValue else {
            throw YOLOInferenceError.invalidOutputShape
        }

        return try decode(confidence: confidenceArray, coordinates: coordinatesArray)
    }

    private func loadModel() async throws -> MLModel {
        if let model {
            return model
        }
        guard let modelURL = Bundle.main.url(
            forResource: "YOLOv11s",
            withExtension: "mlmodelc"
        ) else {
            throw YOLOInferenceError.modelNotFound
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let loadedModel = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        model = loadedModel
        return loadedModel
    }

    /// `confidence`: [N, numClasses]. `coordinates`: [N, 4] = [x, y, width, height], sudah relatif (0...1).
    private func decode(
        confidence: MLMultiArray,
        coordinates: MLMultiArray
    ) throws -> [BoundingBox] {
        let confShape = confidence.shape.map(\.intValue)
        let coordShape = coordinates.shape.map(\.intValue)
        guard confShape.count == 2,
              coordShape.count == 2,
              coordShape[1] == 4,
              confShape[0] == coordShape[0] else {
            throw YOLOInferenceError.invalidOutputShape
        }

        let numBoxes = confShape[0]
        let numClasses = confShape[1]

        let confRowStride = confidence.strides[0].intValue
        let confColStride = confidence.strides[1].intValue
        let confPointer = confidence.dataPointer.bindMemory(to: Float.self, capacity: confidence.count)

        let coordRowStride = coordinates.strides[0].intValue
        let coordColStride = coordinates.strides[1].intValue
        let coordPointer = coordinates.dataPointer.bindMemory(to: Float.self, capacity: coordinates.count)

        var results: [BoundingBox] = []
        results.reserveCapacity(numBoxes)

        for box in 0..<numBoxes {
            var bestScore: Float = 0
            for classIndex in 0..<numClasses {
                let score = confPointer[box * confRowStride + classIndex * confColStride]
                if score > bestScore {
                    bestScore = score
                }
            }
            guard bestScore > 0 else { continue }

            // x, y adalah CENTER point (bukan top-left), width/height juga
            // dalam skala relatif (0...1), origin top-left/y-down (standar image,
            // bukan Vision-style bottom-left).
            let x = coordPointer[box * coordRowStride + 0 * coordColStride]
            let y = coordPointer[box * coordRowStride + 1 * coordColStride]
            let width = coordPointer[box * coordRowStride + 2 * coordColStride]
            let height = coordPointer[box * coordRowStride + 3 * coordColStride]

            // Konversi: center → top-left (top-down), lalu flip ke Vision-style
            // (bottom-left origin) supaya konsisten dengan BoundingBox.rect(in:).
            results.append(
                BoundingBox(
                    normalizedRect: CGRect(
                        x: CGFloat(x - width / 2),
                        y: CGFloat(1 - y - height / 2),
                        width: CGFloat(width),
                        height: CGFloat(height)
                    ),
                    label: "bacteria",
                    confidence: bestScore
                )
            )
        }

        return results
    }

    private static func resize(_ image: CGImage, to size: Int) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return nil
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return context.makeImage()
    }

    private static func pixelBuffer(from image: CGImage, size: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32ARGB,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return buffer
    }
}
