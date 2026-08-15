@preconcurrency import CoreML
import CoreGraphics
import CoreVideo
import Foundation

nonisolated enum RetinaNetInferenceError: LocalizedError {
    case modelNotFound
    case pixelBufferCreationFailed
    case missingOutput
    case invalidOutputShape(boxes: [Int], scores: [Int])
    case detectionFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "ColonyRetinaNet.mlmodelc was not found in the app bundle."
        case .pixelBufferCreationFailed:
            "The cropped dish image could not be prepared for RetinaNet inference."
        case .missingOutput:
            "RetinaNet did not return boxes and scores."
        case .invalidOutputShape(let boxes, let scores):
            "Unexpected RetinaNet output shapes: boxes \(boxes), scores \(scores)."
        case .detectionFailed(let message):
            "RetinaNet inference failed: \(message)"
        }
    }
}

actor RetinaNetDetector {
    static let inputSize = 1024

    private let confidenceThreshold: Double = 0.40
    private let nmsIoUThreshold: Double = 0.50
    private var model: MLModel?

    /// Runs colony detection on the square dish crop produced by `DishCropper`.
    func detect(in croppedImage: CGImage) async throws -> [BoundingBox] {
        guard let pixelBuffer = Self.makeInputPixelBuffer(from: croppedImage) else {
            throw RetinaNetInferenceError.pixelBufferCreationFailed
        }

        let model = try await loadModel()
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: pixelBuffer)
        ])

        let output: MLFeatureProvider
        do {
            output = try await model.prediction(from: input)
        } catch {
            throw RetinaNetInferenceError.detectionFailed(error.localizedDescription)
        }

        guard let boxes = output.featureValue(for: "boxes")?.multiArrayValue,
              let scores = output.featureValue(for: "scores")?.multiArrayValue else {
            throw RetinaNetInferenceError.missingOutput
        }

        return try Self.decode(
            boxes: boxes,
            scores: scores,
            confidenceThreshold: confidenceThreshold,
            nmsIoUThreshold: nmsIoUThreshold
        )
    }

    private func loadModel() async throws -> MLModel {
        if let model {
            return model
        }

        guard let modelURL = Bundle.main.url(
            forResource: "ColonyRetinaNet",
            withExtension: "mlmodelc"
        ) else {
            throw RetinaNetInferenceError.modelNotFound
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let loadedModel = try await MLModel.load(
            contentsOf: modelURL,
            configuration: configuration
        )
        model = loadedModel
        return loadedModel
    }

    /// The model expects an opaque RGB 1024x1024 image. Aspect-fit drawing keeps
    /// non-square inputs undistorted and turns the crop's transparent area black.
    private static func makeInputPixelBuffer(from image: CGImage) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputSize,
            inputSize,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue
            | CGImageAlphaInfo.premultipliedFirst.rawValue
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: inputSize,
            height: inputSize,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo
        ) else {
            return nil
        }

        let canvas = CGRect(x: 0, y: 0, width: inputSize, height: inputSize)
        context.setFillColor(CGColor(gray: 0, alpha: 1))
        context.fill(canvas)
        context.interpolationQuality = .high

        let width = CGFloat(image.width)
        let height = CGFloat(image.height)
        guard width > 0, height > 0 else { return nil }
        let scale = min(CGFloat(inputSize) / width, CGFloat(inputSize) / height)
        let renderedSize = CGSize(width: width * scale, height: height * scale)
        let destination = CGRect(
            x: (CGFloat(inputSize) - renderedSize.width) * 0.5,
            y: (CGFloat(inputSize) - renderedSize.height) * 0.5,
            width: renderedSize.width,
            height: renderedSize.height
        )
        context.draw(image, in: destination)
        return pixelBuffer
    }

    private struct Candidate {
        let rect: CGRect
        let score: Float
    }

    /// `boxes` is [N, 4] in detector-space top-left xyxy pixels; `scores` is [N].
    private static func decode(
        boxes: MLMultiArray,
        scores: MLMultiArray,
        confidenceThreshold: Double,
        nmsIoUThreshold: Double
    ) throws -> [BoundingBox] {
        let boxShape = boxes.shape.map(\.intValue)
        let scoreShape = scores.shape.map(\.intValue)
        guard boxShape.count == 2,
              boxShape[1] == 4,
              scoreShape.count == 1,
              scoreShape[0] == boxShape[0] else {
            throw RetinaNetInferenceError.invalidOutputShape(
                boxes: boxShape,
                scores: scoreShape
            )
        }

        let maximumCoordinate = Double(inputSize)
        var candidates: [Candidate] = []
        candidates.reserveCapacity(boxShape[0])

        for index in 0..<boxShape[0] {
            let row = NSNumber(value: index)
            let score = scores[[row]].doubleValue
            guard score.isFinite, score >= confidenceThreshold else { continue }

            let x1 = boxes[[row, NSNumber(value: 0)]].doubleValue
            let y1 = boxes[[row, NSNumber(value: 1)]].doubleValue
            let x2 = boxes[[row, NSNumber(value: 2)]].doubleValue
            let y2 = boxes[[row, NSNumber(value: 3)]].doubleValue
            guard x1.isFinite, y1.isFinite, x2.isFinite, y2.isFinite else {
                continue
            }

            let clampedX1 = min(maximumCoordinate, max(0, x1))
            let clampedY1 = min(maximumCoordinate, max(0, y1))
            let clampedX2 = min(maximumCoordinate, max(0, x2))
            let clampedY2 = min(maximumCoordinate, max(0, y2))
            guard clampedX2 > clampedX1, clampedY2 > clampedY1 else {
                continue
            }

            candidates.append(
                Candidate(
                    rect: CGRect(
                        x: clampedX1,
                        y: clampedY1,
                        width: clampedX2 - clampedX1,
                        height: clampedY2 - clampedY1
                    ),
                    score: Float(score)
                )
            )
        }

        return nonMaximumSuppression(
            candidates,
            iouThreshold: nmsIoUThreshold
        ).map { candidate in
            let normalizedX = candidate.rect.minX / CGFloat(inputSize)
            let normalizedY = 1 - candidate.rect.maxY / CGFloat(inputSize)
            return BoundingBox(
                normalizedRect: CGRect(
                    x: normalizedX,
                    y: normalizedY,
                    width: candidate.rect.width / CGFloat(inputSize),
                    height: candidate.rect.height / CGFloat(inputSize)
                ),
                label: "koloni",
                confidence: candidate.score
            )
        }
    }

    private static func nonMaximumSuppression(
        _ candidates: [Candidate],
        iouThreshold: Double
    ) -> [Candidate] {
        var remaining = candidates.sorted { $0.score > $1.score }
        var selected: [Candidate] = []
        selected.reserveCapacity(remaining.count)

        while let best = remaining.first {
            selected.append(best)
            remaining.removeFirst()
            remaining.removeAll {
                intersectionOverUnion(best.rect, $0.rect) > iouThreshold
            }
        }
        return selected
    }

    private static func intersectionOverUnion(
        _ first: CGRect,
        _ second: CGRect
    ) -> Double {
        let intersection = first.intersection(second)
        guard !intersection.isNull, !intersection.isEmpty else { return 0 }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = first.width * first.height
            + second.width * second.height
            - intersectionArea
        return unionArea > 0 ? Double(intersectionArea / unionArea) : 0
    }
}
