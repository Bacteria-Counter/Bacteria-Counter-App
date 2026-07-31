@preconcurrency import CoreML
@preconcurrency import Vision
import CoreGraphics
import Foundation
import ImageIO

nonisolated struct DishSegmentationResult: @unchecked Sendable {
    let mask: CGImage
    let foregroundFraction: Double
}

nonisolated enum DishMaskPostProcessor {
    static func clean(
        _ mask: [UInt8],
        width: Int,
        height: Int
    ) -> [UInt8] {
        guard width > 0,
              height > 0,
              mask.count == width * height else {
            return []
        }

        let largest = largestConnectedComponent(
            in: mask,
            width: width,
            height: height
        )
        guard largest.contains(where: { $0 != 0 }) else {
            return [UInt8](repeating: 0, count: mask.count)
        }

        let kernelSize = max(3, (min(width, height) / 150) | 1)
        let kernel = ellipticalKernelOffsets(size: kernelSize)
        let closed = erode(
            dilate(
                largest,
                width: width,
                height: height,
                kernel: kernel
            ),
            width: width,
            height: height,
            kernel: kernel
        )

        return fillHoles(
            in: closed,
            width: width,
            height: height
        )
    }

    private static func largestConnectedComponent(
        in mask: [UInt8],
        width: Int,
        height: Int
    ) -> [UInt8] {
        var labels = [Int32](repeating: 0, count: mask.count)
        var nextLabel: Int32 = 0
        var largestLabel: Int32 = 0
        var largestArea = 0
        var queue: [Int] = []
        queue.reserveCapacity(mask.count)

        for start in mask.indices where mask[start] != 0 && labels[start] == 0 {
            nextLabel += 1
            labels[start] = nextLabel
            queue.removeAll(keepingCapacity: true)
            queue.append(start)
            var queueIndex = 0
            var area = 0

            while queueIndex < queue.count {
                let index = queue[queueIndex]
                queueIndex += 1
                area += 1
                let x = index % width
                let y = index / width

                // Match cv2.connectedComponentsWithStats(..., connectivity: 8).
                for deltaY in -1...1 {
                    let neighborY = y + deltaY
                    guard neighborY >= 0, neighborY < height else { continue }

                    for deltaX in -1...1 where deltaX != 0 || deltaY != 0 {
                        let neighborX = x + deltaX
                        guard neighborX >= 0, neighborX < width else { continue }

                        let neighborIndex = neighborY * width + neighborX
                        guard mask[neighborIndex] != 0,
                              labels[neighborIndex] == 0 else {
                            continue
                        }
                        labels[neighborIndex] = nextLabel
                        queue.append(neighborIndex)
                    }
                }
            }

            // Keep the first component when two components have equal area,
            // matching np.argmax's tie behavior.
            if area > largestArea {
                largestArea = area
                largestLabel = nextLabel
            }
        }

        guard largestLabel != 0 else {
            return [UInt8](repeating: 0, count: mask.count)
        }
        return labels.map { $0 == largestLabel ? 1 : 0 }
    }

    private static func ellipticalKernelOffsets(
        size: Int
    ) -> [(x: Int, y: Int)] {
        let radius = size / 2
        guard radius > 0 else { return [(0, 0)] }

        var offsets: [(x: Int, y: Int)] = []
        for y in -radius...radius {
            let normalizedY = Double(y) / Double(radius)
            let horizontalRadius = Int(
                (
                    Double(radius)
                        * sqrt(max(0, 1 - normalizedY * normalizedY))
                ).rounded()
            )
            for x in -horizontalRadius...horizontalRadius {
                offsets.append((x, y))
            }
        }
        return offsets
    }

    private static func dilate(
        _ mask: [UInt8],
        width: Int,
        height: Int,
        kernel: [(x: Int, y: Int)]
    ) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: mask.count)

        for y in 0..<height {
            for x in 0..<width {
                for offset in kernel {
                    let sourceX = x + offset.x
                    let sourceY = y + offset.y
                    guard sourceX >= 0,
                          sourceX < width,
                          sourceY >= 0,
                          sourceY < height else {
                        continue
                    }
                    if mask[sourceY * width + sourceX] != 0 {
                        result[y * width + x] = 1
                        break
                    }
                }
            }
        }
        return result
    }

    private static func erode(
        _ mask: [UInt8],
        width: Int,
        height: Int,
        kernel: [(x: Int, y: Int)]
    ) -> [UInt8] {
        var result = [UInt8](repeating: 0, count: mask.count)

        for y in 0..<height {
            for x in 0..<width {
                var isForeground = true
                for offset in kernel {
                    let sourceX = x + offset.x
                    let sourceY = y + offset.y

                    // OpenCV's default morphology border is neutral for
                    // erosion, so out-of-bounds kernel positions are ignored.
                    guard sourceX >= 0,
                          sourceX < width,
                          sourceY >= 0,
                          sourceY < height else {
                        continue
                    }
                    if mask[sourceY * width + sourceX] == 0 {
                        isForeground = false
                        break
                    }
                }
                if isForeground {
                    result[y * width + x] = 1
                }
            }
        }
        return result
    }

    private static func fillHoles(
        in mask: [UInt8],
        width: Int,
        height: Int
    ) -> [UInt8] {
        var exterior = [UInt8](repeating: 0, count: mask.count)
        var queue: [Int] = []
        queue.reserveCapacity(mask.count)

        func enqueue(_ index: Int) {
            guard mask[index] == 0, exterior[index] == 0 else { return }
            exterior[index] = 1
            queue.append(index)
        }

        for x in 0..<width {
            enqueue(x)
            enqueue((height - 1) * width + x)
        }
        for y in 0..<height {
            enqueue(y * width)
            enqueue(y * width + width - 1)
        }

        var queueIndex = 0
        while queueIndex < queue.count {
            let index = queue[queueIndex]
            queueIndex += 1
            let x = index % width
            let y = index / width

            // Four-connected exterior background is complementary to the
            // eight-connected foreground component.
            if x > 0 { enqueue(index - 1) }
            if x + 1 < width { enqueue(index + 1) }
            if y > 0 { enqueue(index - width) }
            if y + 1 < height { enqueue(index + width) }
        }

        return mask.indices.map { index in
            mask[index] != 0 || exterior[index] == 0 ? 1 : 0
        }
    }
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
        var thresholded = [UInt8](repeating: 0, count: width * height)

        for y in 0..<height {
            for x in 0..<width {
                let inputIndex = y * rowStride + x * columnStride
                if values[inputIndex] >= threshold {
                    thresholded[y * width + x] = 1
                }
            }
        }

        let cleaned = DishMaskPostProcessor.clean(
            thresholded,
            width: width,
            height: height
        )
        let foregroundCount = cleaned.reduce(0) {
            $0 + ($1 == 0 ? 0 : 1)
        }
        guard foregroundCount > 0 else {
            throw DishSegmentationError.emptyMask
        }
        let pixels = cleaned.map { $0 == 0 ? UInt8(0) : UInt8(255) }

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

// YG LAMA TANPA POST PROCESSING DI SWIFT STLH OUTPUT MODEL
//@preconcurrency import CoreML
//@preconcurrency import Vision
//import CoreGraphics
//import Foundation
//import ImageIO
//
//nonisolated struct DishSegmentationResult: @unchecked Sendable {
//    let mask: CGImage
//    let foregroundFraction: Double
//}
//
//nonisolated enum DishSegmentationError: LocalizedError {
//    case modelNotFound
//    case imageConversionFailed
//    case outputNotFound
//    case invalidOutputType
//    case invalidOutputShape([Int])
//    case emptyMask
//    case cannotCreateMask
//
//    var errorDescription: String? {
//        switch self {
//        case .modelNotFound:
//            "PetriDishSegmentation.mlmodelc was not found in the app bundle."
//        case .imageConversionFailed:
//            "The selected image could not be prepared for segmentation."
//        case .outputNotFound:
//            "The model did not return dish_probability."
//        case .invalidOutputType:
//            "dish_probability is not a Float32 MLMultiArray."
//        case .invalidOutputShape(let shape):
//            "Unexpected dish_probability shape: \(shape)."
//        case .emptyMask:
//            "The model did not detect a petri dish in this image."
//        case .cannotCreateMask:
//            "The petri dish mask image could not be created."
//        }
//    }
//}
//
//actor PetriDishSegmenter {
//    private static let inputSize = 512
//    private let threshold: Float = 0.5
//    private var visionModel: VNCoreMLModel?
//
//    func makeMask(
//        from cgImage: CGImage,
//        orientation: CGImagePropertyOrientation = .up
//    ) async throws -> DishSegmentationResult {
//        let request = VNCoreMLRequest(model: try await loadVisionModel())
//
//        // The Python preprocessing resizes directly to 512×512 without cropping.
//        request.imageCropAndScaleOption = .scaleFill
//
//        let handler = VNImageRequestHandler(
//            cgImage: cgImage,
//            orientation: orientation,
//            options: [:]
//        )
//        try handler.perform([request])
//
//        guard let observation = request.results?
//            .compactMap({ $0 as? VNCoreMLFeatureValueObservation })
//            .first(where: { $0.featureName == "dish_probability" }),
//              let probabilities = observation.featureValue.multiArrayValue else {
//            throw DishSegmentationError.outputNotFound
//        }
//
//        return try binaryMask(from: probabilities)
//    }
//
//    private func loadVisionModel() async throws -> VNCoreMLModel {
//        if let visionModel {
//            return visionModel
//        }
//
//        guard let modelURL = Bundle.main.url(
//            forResource: "PetriDishSegmentation",
//            withExtension: "mlmodelc"
//        ) else {
//            throw DishSegmentationError.modelNotFound
//        }
//
//        let configuration = MLModelConfiguration()
//        configuration.computeUnits = .all
//        let coreMLModel = try await MLModel.load(
//            contentsOf: modelURL,
//            configuration: configuration
//        )
//        let loadedModel = try VNCoreMLModel(for: coreMLModel)
//        visionModel = loadedModel
//        return loadedModel
//    }
//
//    private func binaryMask(
//        from array: MLMultiArray
//    ) throws -> DishSegmentationResult {
//        guard array.dataType == .float32 else {
//            throw DishSegmentationError.invalidOutputType
//        }
//
//        let shape = array.shape.map(\.intValue)
//        guard shape.count == 4,
//              shape[0] == 1,
//              shape[1] == 1,
//              shape[2] == Self.inputSize,
//              shape[3] == Self.inputSize else {
//            throw DishSegmentationError.invalidOutputShape(shape)
//        }
//
//        let height = shape[2]
//        let width = shape[3]
//        let rowStride = array.strides[2].intValue
//        let columnStride = array.strides[3].intValue
//        let values = array.dataPointer.bindMemory(
//            to: Float.self,
//            capacity: array.count
//        )
//        var pixels = [UInt8](repeating: 0, count: width * height)
//        var foregroundCount = 0
//
//        for y in 0..<height {
//            for x in 0..<width {
//                let inputIndex = y * rowStride + x * columnStride
//                if values[inputIndex] >= threshold {
//                    pixels[y * width + x] = 255
//                    foregroundCount += 1
//                }
//            }
//        }
//
//        guard foregroundCount > 0 else {
//            throw DishSegmentationError.emptyMask
//        }
//
//        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
//              let mask = CGImage(
//                  width: width,
//                  height: height,
//                  bitsPerComponent: 8,
//                  bitsPerPixel: 8,
//                  bytesPerRow: width,
//                  space: CGColorSpaceCreateDeviceGray(),
//                  bitmapInfo: CGBitmapInfo(
//                      rawValue: CGImageAlphaInfo.none.rawValue
//                  ),
//                  provider: provider,
//                  decode: nil,
//                  shouldInterpolate: false,
//                  intent: .defaultIntent
//              ) else {
//            throw DishSegmentationError.cannotCreateMask
//        }
//
//        return DishSegmentationResult(
//            mask: mask,
//            foregroundFraction: Double(foregroundCount) / Double(width * height)
//        )
//    }
//}
