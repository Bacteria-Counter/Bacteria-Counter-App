//
//  YOLOInference.swift
//  bacteriaapp
//
//  Created by Regina Celine Adiwinata on 06/08/26.
//

import CoreML
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
    case modelNotFound(String)
    case resizeFailed
    case pixelBufferCreationFailed
    case invalidOutputShape
    case detectionFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            "\(name).mlmodelc was not found in the app bundle."
        case .resizeFailed:
            "The cropped dish image could not be resized for inference."
        case .pixelBufferCreationFailed:
            "The image could not be converted for model input."
        case .invalidOutputShape:
            "Unexpected detection output shape."
        case .detectionFailed(let message):
            "YOLO inference failed: \(message)"
        }
    }
}

actor YOLODetector {
    // Cache model per varian, supaya ganti-ganti model tidak perlu reload dari disk tiap kali
    private var loadedModels: [YOLOModelVariant: MLModel] = [:]

    /// Menjalankan deteksi bakteri pada gambar dish yang sudah di-crop,
    /// menyesuaikan format input/output sesuai varian model yang dipilih.
    func detect(in croppedImage: CGImage, variant: YOLOModelVariant) async throws -> [BoundingBox] {
        let inputSize = variant.inputSize

        guard let resized = Self.resize(croppedImage, to: inputSize) else {
            throw YOLOInferenceError.resizeFailed
        }
        guard let pixelBuffer = Self.pixelBuffer(from: resized, size: inputSize) else {
            throw YOLOInferenceError.pixelBufferCreationFailed
        }

        let model = try await loadModel(variant: variant)

        if variant.usesLegacyNMSPipeline {
            return try await detectLegacyPipeline(
                model: model,
                pixelBuffer: pixelBuffer,
                variant: variant
            )
        } else {
            return try await detectEnd2End(
                model: model,
                pixelBuffer: pixelBuffer,
                variant: variant
            )
        }
    }

    // MARK: - Model loading

    private func loadModel(variant: YOLOModelVariant) async throws -> MLModel {
        if let cached = loadedModels[variant] {
            return cached
        }
        guard let modelURL = Bundle.main.url(
            forResource: variant.resourceName,
            withExtension: "mlmodelc"
        ) else {
            throw YOLOInferenceError.modelNotFound(variant.resourceName)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let loadedModel = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        loadedModels[variant] = loadedModel
        return loadedModel
    }

    // MARK: - YOLOv11s: pipeline dengan NMS terpisah (confidence + coordinates)

    private func detectLegacyPipeline(
        model: MLModel,
        pixelBuffer: CVPixelBuffer,
        variant: YOLOModelVariant
    ) async throws -> [BoundingBox] {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixelBuffer),
            "iouThreshold": MLFeatureValue(double: variant.iouThreshold),
            "confidenceThreshold": MLFeatureValue(double: variant.confidenceThreshold)
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

        return try decodeLegacy(confidence: confidenceArray, coordinates: coordinatesArray)
    }

    /// `confidence`: [N, numClasses]. `coordinates`: [N, 4] = [x, y, width, height] center-based, relatif (0...1).
    private func decodeLegacy(
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
                if score > bestScore { bestScore = score }
            }
            guard bestScore > 0 else { continue }

            let x = coordPointer[box * coordRowStride + 0 * coordColStride]
            let y = coordPointer[box * coordRowStride + 1 * coordColStride]
            let width = coordPointer[box * coordRowStride + 2 * coordColStride]
            let height = coordPointer[box * coordRowStride + 3 * coordColStride]

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

    // MARK: - YOLOv26s: end2end (1 output tensor [1, N, 6] = xyxy+conf+classId)

    private func detectEnd2End(
        model: MLModel,
        pixelBuffer: CVPixelBuffer,
        variant: YOLOModelVariant
    ) async throws -> [BoundingBox] {
        // Model end2end hanya butuh 1 input: "image"
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixelBuffer)
        ])

        let output: MLFeatureProvider
        do {
            output = try await model.prediction(from: input)
        } catch {
            throw YOLOInferenceError.detectionFailed(error.localizedDescription)
        }

        // Nama output auto-generated (mis. "var_1441") bisa berubah tiap re-export,
        // jadi ambil output pertama apapun namanya.
        guard let outputName = output.featureNames.first,
              let detections = output.featureValue(for: outputName)?.multiArrayValue else {
            throw YOLOInferenceError.invalidOutputShape
        }

        return try decodeEnd2End(
            detections: detections,
            confidenceThreshold: Float(variant.confidenceThreshold),
            inputSize: variant.inputSize
        )
    }

    /// `detections`: shape [1, N, 6] = [x1, y1, x2, y2, confidence, classId] dalam pixel
    /// absolut terhadap ukuran input model.
    private func decodeEnd2End(
        detections: MLMultiArray,
        confidenceThreshold: Float,
        inputSize: Int
    ) throws -> [BoundingBox] {
        let shape = detections.shape.map(\.intValue)
        guard shape.count == 3, shape[0] == 1, shape[2] == 6 else {
            throw YOLOInferenceError.invalidOutputShape
        }

        let numDetections = shape[1]
        let rowStride = detections.strides[1].intValue
        let colStride = detections.strides[2].intValue
        let pointer = detections.dataPointer.bindMemory(to: Float.self, capacity: detections.count)

        let scale = Float(inputSize)
        var results: [BoundingBox] = []
        results.reserveCapacity(numDetections)

        for row in 0..<numDetections {
            let base = row * rowStride
            let x1 = pointer[base + 0 * colStride]
            let y1 = pointer[base + 1 * colStride]
            let x2 = pointer[base + 2 * colStride]
            let y2 = pointer[base + 3 * colStride]
            let confidence = pointer[base + 4 * colStride]
            // classId di kolom ke-5 diabaikan, cuma 1 kelas ("bakteri")

            guard confidence >= confidenceThreshold, x2 > x1, y2 > y1 else { continue }

            let x1n = x1 / scale
            let y1n = y1 / scale
            let x2n = x2 / scale
            let y2n = y2 / scale

            results.append(
                BoundingBox(
                    normalizedRect: CGRect(
                        x: CGFloat(x1n),
                        y: CGFloat(1 - y2n),
                        width: CGFloat(x2n - x1n),
                        height: CGFloat(y2n - y1n)
                    ),
                    label: "bakteri",
                    confidence: confidence
                )
            )
        }

        return results
    }

    // MARK: - Shared image utils

    private static func resize(_ image: CGImage, to size: Int) -> CGImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: size, height: size, bitsPerComponent: 8,
            bytesPerRow: 0, space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
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
        CVPixelBufferCreate(kCFAllocatorDefault, size, size, kCVPixelFormatType_32ARGB, attrs as CFDictionary, &pixelBuffer)
        guard let buffer = pixelBuffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer), width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return buffer
    }
}
