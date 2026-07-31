@preconcurrency import CoreML
@preconcurrency import Vision
import CoreGraphics
import Foundation
import ImageIO

nonisolated struct DishSegmentationResult: @unchecked Sendable {
    let mask: CGImage
    let foregroundFraction: Double
}

nonisolated enum DishSegmentationError: LocalizedError {
    case modelNotFound
    case imageConversionFailed
    case outputNotFound
    case invalidOutputType
    case invalidOutputShape([Int])
    case emptyMask
    case cannotCreateMask

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "PetriDishSegmentation.mlmodelc was not found in the app bundle."
        case .imageConversionFailed:
            "The selected image could not be prepared for segmentation."
        case .outputNotFound:
            "The model did not return dish_probability."
        case .invalidOutputType:
            "dish_probability is not a Float32 MLMultiArray."
        case .invalidOutputShape(let shape):
            "Unexpected dish_probability shape: \(shape)."
        case .emptyMask:
            "The model did not detect a petri dish in this image."
        case .cannotCreateMask:
            "The petri dish mask image could not be created."
        }
    }
}

actor PetriDishSegmenter {
    private static let inputSize = 512
    private let threshold: Float = 0.5
    private var visionModel: VNCoreMLModel?

    func makeMask(
        from cgImage: CGImage,
        orientation: CGImagePropertyOrientation = .up
    ) async throws -> DishSegmentationResult {
        let request = VNCoreMLRequest(model: try await loadVisionModel())

        // The Python preprocessing resizes directly to 512×512 without cropping.
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: orientation,
            options: [:]
        )
        try handler.perform([request])

        guard let observation = request.results?
            .compactMap({ $0 as? VNCoreMLFeatureValueObservation })
            .first(where: { $0.featureName == "dish_probability" }),
              let probabilities = observation.featureValue.multiArrayValue else {
            throw DishSegmentationError.outputNotFound
        }

        return try binaryMask(from: probabilities)
    }

    private func loadVisionModel() async throws -> VNCoreMLModel {
        if let visionModel {
            return visionModel
        }

        guard let modelURL = Bundle.main.url(
            forResource: "PetriDishSegmentation",
            withExtension: "mlmodelc"
        ) else {
            throw DishSegmentationError.modelNotFound
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let coreMLModel = try await MLModel.load(
            contentsOf: modelURL,
            configuration: configuration
        )
        let loadedModel = try VNCoreMLModel(for: coreMLModel)
        visionModel = loadedModel
        return loadedModel
    }

    private func binaryMask(
        from array: MLMultiArray
    ) throws -> DishSegmentationResult {
        guard array.dataType == .float32 else {
            throw DishSegmentationError.invalidOutputType
        }

        let shape = array.shape.map(\.intValue)
        guard shape.count == 4,
              shape[0] == 1,
              shape[1] == 1,
              shape[2] == Self.inputSize,
              shape[3] == Self.inputSize else {
            throw DishSegmentationError.invalidOutputShape(shape)
        }

        let height = shape[2]
        let width = shape[3]
        let rowStride = array.strides[2].intValue
        let columnStride = array.strides[3].intValue
        let values = array.dataPointer.bindMemory(
            to: Float.self,
            capacity: array.count
        )
        var pixels = [UInt8](repeating: 0, count: width * height)
        var foregroundCount = 0

        for y in 0..<height {
            for x in 0..<width {
                let inputIndex = y * rowStride + x * columnStride
                if values[inputIndex] >= threshold {
                    pixels[y * width + x] = 255
                    foregroundCount += 1
                }
            }
        }

        guard foregroundCount > 0 else {
            throw DishSegmentationError.emptyMask
        }

        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let mask = CGImage(
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bitsPerPixel: 8,
                  bytesPerRow: width,
                  space: CGColorSpaceCreateDeviceGray(),
                  bitmapInfo: CGBitmapInfo(
                      rawValue: CGImageAlphaInfo.none.rawValue
                  ),
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              ) else {
            throw DishSegmentationError.cannotCreateMask
        }

        return DishSegmentationResult(
            mask: mask,
            foregroundFraction: Double(foregroundCount) / Double(width * height)
        )
    }
}
