@preconcurrency import CoreML
import Accelerate
import CoreGraphics
import Foundation

nonisolated struct ColonyAnalysis: Sendable {
    let totalColonies: Int
    let averageConfidence: Int
    let boundingBoxes: [ColonyBox]
}

nonisolated enum ColonyAnalysisError: LocalizedError {
    case resourceNotFound(String)
    case invalidParameters(String)
    case imageConversionFailed
    case plateNotDetected
    case invalidModelOutput(String)
    case imageProcessingFailed(String)

    var errorDescription: String? {
        switch self {
        case .resourceNotFound(let name):
            "Required analysis resource was not found: \(name)"
        case .invalidParameters(let reason):
            "Invalid analysis parameters: \(reason)"
        case .imageConversionFailed:
            "The captured image could not be prepared for analysis."
        case .plateNotDetected:
            "No petri dish was detected in the captured image."
        case .invalidModelOutput(let reason):
            "A model returned an unexpected output: \(reason)"
        case .imageProcessingFailed(let reason):
            "Image processing failed: \(reason)"
        }
    }
}

actor ColonyAnalysisService {
    typealias ProgressHandler = @MainActor @Sendable (
        _ progress: Double,
        _ currentCount: Int
    ) -> Void

    private struct Models {
        let plate: MLModel
        let colony: MLModel
    }

    private struct AnalysisParameters: Decodable {
        let plateInputSize: Int
        let plateThreshold: Double
        let colonyTileSize: Int
        let colonyTileOverlap: Int
        let colonyOutputStride: Int
        let colonyScoreThreshold: Double
        let tileNmsIou: Double
        let globalNmsIou: Double
        let tileEdgeMargin: Int
        let imagenetMean: [Double]
        let imagenetStd: [Double]
        let plateRoiTargetSize: Int
        let plateCropPaddingRatio: Double
        let plateCountingScale: Double
        let flatfieldSigmaRatio: Double
        let safeErodeRatio: Double
        let safeMinimumFraction: Double
        let flatfieldPercentileLow: Double
        let flatfieldPercentileHigh: Double

        func validate() throws {
            guard plateInputSize > 0 else {
                throw ColonyAnalysisError.invalidParameters("plate_input_size must be positive")
            }
            guard colonyTileSize > 0,
                  colonyTileOverlap >= 0,
                  colonyTileOverlap < colonyTileSize else {
                throw ColonyAnalysisError.invalidParameters(
                    "colony tile size and overlap are inconsistent"
                )
            }
            guard colonyOutputStride > 0 else {
                throw ColonyAnalysisError.invalidParameters(
                    "colony_output_stride must be positive"
                )
            }
            guard plateRoiTargetSize > 0, plateCountingScale > 0 else {
                throw ColonyAnalysisError.invalidParameters(
                    "plate ROI size and counting scale must be positive"
                )
            }
            guard flatfieldSigmaRatio > 0,
                  flatfieldPercentileLow >= 0,
                  flatfieldPercentileHigh <= 100,
                  flatfieldPercentileLow < flatfieldPercentileHigh else {
                throw ColonyAnalysisError.invalidParameters(
                    "flat-field sigma and percentile range are invalid"
                )
            }
            guard imagenetMean.count >= 3, imagenetStd.count >= 3,
                  imagenetStd.prefix(3).allSatisfy({ $0 > 0 }) else {
                throw ColonyAnalysisError.invalidParameters(
                    "ImageNet mean and standard deviation require three channels"
                )
            }
        }
    }

    private struct PixelRect {
        var x: Int
        var y: Int
        var width: Int
        var height: Int

        var maxX: Int { x + width }
        var maxY: Int { y + height }
    }

    private struct PlateMask {
        let width: Int
        let height: Int
        let ellipse: PlateEllipse
    }

    private struct PlateEllipse {
        let centerX: Double
        let centerY: Double
        let axis1: Double
        let axis2: Double
        let angle: Double
    }

    private struct RGBAImage {
        let width: Int
        let height: Int
        let pixels: [UInt8]

        init(cgImage: CGImage) throws {
            let width = cgImage.width
            let height = cgImage.height
            guard width > 0, height > 0 else {
                throw ColonyAnalysisError.imageConversionFailed
            }

            var pixels = [UInt8](repeating: 0, count: width * height * 4)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue
                | CGImageAlphaInfo.premultipliedLast.rawValue

            let didDraw = pixels.withUnsafeMutableBytes { rawBuffer -> Bool in
                guard let baseAddress = rawBuffer.baseAddress,
                      let context = CGContext(
                        data: baseAddress,
                        width: width,
                        height: height,
                        bitsPerComponent: 8,
                        bytesPerRow: width * 4,
                        space: colorSpace,
                        bitmapInfo: bitmapInfo
                      ) else {
                    return false
                }

                context.interpolationQuality = .high
                context.draw(
                    cgImage,
                    in: CGRect(x: 0, y: 0, width: width, height: height)
                )
                return true
            }

            guard didDraw else {
                throw ColonyAnalysisError.imageConversionFailed
            }

            self.width = width
            self.height = height
            self.pixels = pixels
        }

        private init(width: Int, height: Int, pixels: [UInt8]) {
            self.width = width
            self.height = height
            self.pixels = pixels
        }

        init(grayscale: [UInt8], width: Int, height: Int) throws {
            guard width > 0,
                  height > 0,
                  grayscale.count == width * height else {
                throw ColonyAnalysisError.imageProcessingFailed(
                    "invalid flat-field grayscale dimensions"
                )
            }
            var pixels = [UInt8](repeating: 255, count: width * height * 4)
            for index in grayscale.indices {
                let destination = index * 4
                let value = grayscale[index]
                pixels[destination] = value
                pixels[destination + 1] = value
                pixels[destination + 2] = value
            }
            self.init(width: width, height: height, pixels: pixels)
        }

        func resized(
            crop: PixelRect? = nil,
            width destinationWidth: Int,
            height destinationHeight: Int
        ) throws -> RGBAImage {
            let crop = crop ?? PixelRect(x: 0, y: 0, width: width, height: height)
            guard crop.x >= 0,
                  crop.y >= 0,
                  crop.width > 0,
                  crop.height > 0,
                  crop.maxX <= width,
                  crop.maxY <= height,
                  destinationWidth > 0,
                  destinationHeight > 0 else {
                throw ColonyAnalysisError.imageProcessingFailed("invalid crop rectangle")
            }

            var output = [UInt8](
                repeating: 0,
                count: destinationWidth * destinationHeight * 4
            )

            let error = pixels.withUnsafeBytes { sourceBytes in
                output.withUnsafeMutableBytes { destinationBytes in
                    guard let sourceBase = sourceBytes.baseAddress,
                          let destinationBase = destinationBytes.baseAddress else {
                        return kvImageNullPointerArgument
                    }

                    let sourceOffset = crop.y * width * 4 + crop.x * 4
                    var sourceBuffer = vImage_Buffer(
                        data: UnsafeMutableRawPointer(
                            mutating: sourceBase.advanced(by: sourceOffset)
                        ),
                        height: vImagePixelCount(crop.height),
                        width: vImagePixelCount(crop.width),
                        rowBytes: width * 4
                    )
                    var destinationBuffer = vImage_Buffer(
                        data: destinationBase,
                        height: vImagePixelCount(destinationHeight),
                        width: vImagePixelCount(destinationWidth),
                        rowBytes: destinationWidth * 4
                    )

                    return vImageScale_ARGB8888(
                        &sourceBuffer,
                        &destinationBuffer,
                        nil,
                        vImage_Flags(kvImageHighQualityResampling)
                    )
                }
            }

            guard error == kvImageNoError else {
                throw ColonyAnalysisError.imageProcessingFailed(
                    "vImage scaling returned \(error)"
                )
            }

            return RGBAImage(
                width: destinationWidth,
                height: destinationHeight,
                pixels: output
            )
        }

        func normalizedTensor(
            mean: [Double],
            standardDeviation: [Double]
        ) throws -> MLMultiArray {
            guard mean.count >= 3, standardDeviation.count >= 3 else {
                throw ColonyAnalysisError.invalidParameters(
                    "normalization requires three channels"
                )
            }

            let pixelCount = width * height
            let array = try MLMultiArray(
                shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
                dataType: .float32
            )
            let values = array.dataPointer.bindMemory(
                to: Float.self,
                capacity: pixelCount * 3
            )

            let redMean = Float(mean[0])
            let greenMean = Float(mean[1])
            let blueMean = Float(mean[2])
            let redStd = Float(standardDeviation[0])
            let greenStd = Float(standardDeviation[1])
            let blueStd = Float(standardDeviation[2])

            pixels.withUnsafeBufferPointer { source in
                for index in 0..<pixelCount {
                    let sourceIndex = index * 4
                    let red = Float(source[sourceIndex]) / 255
                    let green = Float(source[sourceIndex + 1]) / 255
                    let blue = Float(source[sourceIndex + 2]) / 255

                    values[index] = (red - redMean) / redStd
                    values[pixelCount + index] = (green - greenMean) / greenStd
                    values[pixelCount * 2 + index] = (blue - blueMean) / blueStd
                }
            }

            return array
        }
    }

    private var cachedModels: Models?
    private var cachedParameters: AnalysisParameters?

    func analyze(
        image: CGImage,
        progress: ProgressHandler
    ) async throws -> ColonyAnalysis {
        try Task.checkCancellation()
        await progress(0.02, 0)

        let parameters = try loadParameters()
        let models = try loadModels()
        let sourceImage = try RGBAImage(cgImage: image)

        let plateInputImage = try sourceImage.resized(
            width: parameters.plateInputSize,
            height: parameters.plateInputSize
        )
        let plateInput = try plateInputImage.normalizedTensor(
            mean: parameters.imagenetMean,
            standardDeviation: parameters.imagenetStd
        )
        let platePrediction = try predict(
            model: models.plate,
            input: plateInput
        )
        guard let probability = platePrediction.featureValue(
            for: "plate_probability"
        )?.multiArrayValue else {
            throw ColonyAnalysisError.invalidModelOutput(
                "PlateU2NetP did not return plate_probability"
            )
        }

        let plateMask = try makePlateMask(
            probability: probability,
            threshold: parameters.plateThreshold
        )
        let plateCrop = makePlateCrop(
            plateMask: plateMask,
            sourceWidth: sourceImage.width,
            sourceHeight: sourceImage.height,
            paddingRatio: parameters.plateCropPaddingRatio
        )

        let roiSize = max(
            parameters.colonyTileSize,
            parameters.plateRoiTargetSize
        )
        let rawPlateImage = try sourceImage.resized(
            crop: plateCrop,
            width: roiSize,
            height: roiSize
        )
        let countingMask = makeCountingMask(
            plateMask: plateMask,
            plateCrop: plateCrop,
            sourceWidth: sourceImage.width,
            sourceHeight: sourceImage.height,
            roiSize: roiSize,
            countingScale: parameters.plateCountingScale
        )
        let flatField = try FlatFieldNormalizer.normalize(
            rgba: rawPlateImage.pixels,
            width: roiSize,
            height: roiSize,
            countingMask: countingMask,
            configuration: FlatFieldNormalizer.Configuration(
                safeErodeRatio: parameters.safeErodeRatio,
                safeMinimumFraction: parameters.safeMinimumFraction,
                outlierMADMultiplier: 3.5,
                sigmaRatio: parameters.flatfieldSigmaRatio,
                percentileLow: parameters.flatfieldPercentileLow,
                percentileHigh: parameters.flatfieldPercentileHigh
            )
        )
        let plateImage = try RGBAImage(
            grayscale: flatField,
            width: roiSize,
            height: roiSize
        )

        try Task.checkCancellation()
        await progress(0.18, 0)

        let xOrigins = CenterNetDecoder.tileOrigins(
            length: roiSize,
            tileSize: parameters.colonyTileSize,
            overlap: parameters.colonyTileOverlap
        )
        let yOrigins = CenterNetDecoder.tileOrigins(
            length: roiSize,
            tileSize: parameters.colonyTileSize,
            overlap: parameters.colonyTileOverlap
        )
        let totalTiles = xOrigins.count * yOrigins.count
        var completedTiles = 0
        var allBoxes: [ColonyBox] = []

        for y in yOrigins {
            for x in xOrigins {
                try Task.checkCancellation()

                let tile = try plateImage.resized(
                    crop: PixelRect(
                        x: x,
                        y: y,
                        width: parameters.colonyTileSize,
                        height: parameters.colonyTileSize
                    ),
                    width: parameters.colonyTileSize,
                    height: parameters.colonyTileSize
                )
                let input = try tile.normalizedTensor(
                    mean: parameters.imagenetMean,
                    standardDeviation: parameters.imagenetStd
                )
                let output = try predict(model: models.colony, input: input)

                guard let heatmap = output.featureValue(for: "heatmap")?.multiArrayValue,
                      let size = output.featureValue(for: "size")?.multiArrayValue,
                      let offset = output.featureValue(for: "offset")?.multiArrayValue else {
                    throw ColonyAnalysisError.invalidModelOutput(
                        "ColonyResNet50FPN must return heatmap, size, and offset"
                    )
                }

                let decoded = try CenterNetDecoder.decode(
                    heatmap: heatmap,
                    size: size,
                    offset: offset,
                    tileOriginX: x,
                    tileOriginY: y,
                    stride: parameters.colonyOutputStride,
                    scoreThreshold: parameters.colonyScoreThreshold,
                    tileSize: parameters.colonyTileSize,
                    tileNMSIoU: parameters.tileNmsIou
                )
                let safeTileBoxes = CenterNetDecoder.removeTileEdgeDetections(
                    decoded,
                    tileOriginX: x,
                    tileOriginY: y,
                    sourceWidth: roiSize,
                    sourceHeight: roiSize,
                    tileSize: parameters.colonyTileSize,
                    margin: parameters.tileEdgeMargin
                )
                allBoxes.append(contentsOf: safeTileBoxes)
                completedTiles += 1

                let interim = filteredAndMergedBoxes(
                    allBoxes,
                    countingMask: countingMask,
                    roiSize: roiSize,
                    iouThreshold: parameters.globalNmsIou,
                    duplicateCenterDistance: Double(
                        parameters.colonyOutputStride
                    ) * 1.5
                )
                let fraction = Double(completedTiles) / Double(max(1, totalTiles))
                await progress(0.18 + fraction * 0.77, interim.count)
            }
        }

        try Task.checkCancellation()
        let boxes = filteredAndMergedBoxes(
            allBoxes,
            countingMask: countingMask,
            roiSize: roiSize,
            iouThreshold: parameters.globalNmsIou,
            duplicateCenterDistance: Double(parameters.colonyOutputStride) * 1.5
        )
        let sourceBoxes = mapBoxesToSourceImage(
            boxes,
            plateCrop: plateCrop,
            roiSize: roiSize,
            sourceWidth: sourceImage.width,
            sourceHeight: sourceImage.height
        )
        let confidence = sourceBoxes.isEmpty
            ? 0
            : Int(
                (
                    sourceBoxes.reduce(0.0) { $0 + $1.score }
                        / Double(sourceBoxes.count)
                    * 100
                ).rounded()
            )

        await progress(1.0, sourceBoxes.count)
        return ColonyAnalysis(
            totalColonies: sourceBoxes.count,
            averageConfidence: confidence,
            boundingBoxes: sourceBoxes
        )
    }

    private func loadModels() throws -> Models {
        if let cachedModels {
            return cachedModels
        }

        let plateURL = try resourceURL(
            name: "PlateU2NetP",
            extension: "mlmodelc"
        )
        let colonyURL = try resourceURL(
            name: "ColonyResNet50FPN",
            extension: "mlmodelc"
        )
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all

        let models = Models(
            plate: try MLModel(contentsOf: plateURL, configuration: configuration),
            colony: try MLModel(contentsOf: colonyURL, configuration: configuration)
        )
        cachedModels = models
        return models
    }

    private func loadParameters() throws -> AnalysisParameters {
        if let cachedParameters {
            return cachedParameters
        }

        let url = try resourceURL(
            name: "production_parameters",
            extension: "json"
        )
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let parameters = try decoder.decode(
            AnalysisParameters.self,
            from: Data(contentsOf: url)
        )
        try parameters.validate()
        cachedParameters = parameters
        return parameters
    }

    private func resourceURL(name: String, extension fileExtension: String) throws -> URL {
        if let url = Bundle.main.url(
            forResource: name,
            withExtension: fileExtension
        ) {
            return url
        }
        if let url = Bundle.main.url(
            forResource: name,
            withExtension: fileExtension,
            subdirectory: "Models"
        ) {
            return url
        }
        throw ColonyAnalysisError.resourceNotFound("\(name).\(fileExtension)")
    }

    private func predict(
        model: MLModel,
        input: MLMultiArray
    ) throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(multiArray: input)]
        )
        return try model.prediction(from: provider)
    }

    private func makePlateMask(
        probability: MLMultiArray,
        threshold: Double
    ) throws -> PlateMask {
        guard probability.shape.count == 4 else {
            throw ColonyAnalysisError.invalidModelOutput(
                "plate_probability must use NCHW layout"
            )
        }

        let height = probability.shape[2].intValue
        let width = probability.shape[3].intValue
        guard width > 0, height > 0 else {
            throw ColonyAnalysisError.invalidModelOutput(
                "plate_probability has empty dimensions"
            )
        }

        var foreground = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * probability.strides[2].intValue
                    + x * probability.strides[3].intValue
                foreground[y * width + x] =
                    probability[offset].doubleValue >= threshold ? 1 : 0
            }
        }

        var labels = [Int32](repeating: 0, count: foreground.count)
        var nextLabel: Int32 = 0
        var bestLabel: Int32 = 0
        var bestCount = 0
        var queue: [Int] = []
        queue.reserveCapacity(foreground.count)

        for start in foreground.indices where foreground[start] != 0 && labels[start] == 0 {
            nextLabel += 1
            labels[start] = nextLabel
            queue.removeAll(keepingCapacity: true)
            queue.append(start)
            var queueIndex = 0
            var count = 0

            while queueIndex < queue.count {
                let index = queue[queueIndex]
                queueIndex += 1
                count += 1
                let x = index % width
                let y = index / width

                if x > 0 {
                    appendPixel(
                        index - 1,
                        label: nextLabel,
                        foreground: foreground,
                        labels: &labels,
                        queue: &queue
                    )
                }
                if x + 1 < width {
                    appendPixel(
                        index + 1,
                        label: nextLabel,
                        foreground: foreground,
                        labels: &labels,
                        queue: &queue
                    )
                }
                if y > 0 {
                    appendPixel(
                        index - width,
                        label: nextLabel,
                        foreground: foreground,
                        labels: &labels,
                        queue: &queue
                    )
                }
                if y + 1 < height {
                    appendPixel(
                        index + width,
                        label: nextLabel,
                        foreground: foreground,
                        labels: &labels,
                        queue: &queue
                    )
                }
            }

            if count > bestCount {
                bestCount = count
                bestLabel = nextLabel
            }
        }

        guard bestLabel != 0, bestCount > 0 else {
            throw ColonyAnalysisError.plateNotDetected
        }

        let largestComponent: [UInt8] = labels.map {
            $0 == bestLabel ? 1 : 0
        }
        let closeRadius = max(
            2,
            Int((Double(min(width, height)) * 0.006).rounded())
        )
        let closed = erode(
            mask: dilate(
                mask: largestComponent,
                width: width,
                height: height,
                radius: closeRadius
            ),
            width: width,
            height: height,
            radius: closeRadius
        )
        let filled = fillHoles(
            mask: closed,
            width: width,
            height: height
        )
        let filledCount = filled.reduce(0) { $0 + Int($1) }
        guard Double(filledCount) / Double(width * height) >= 0.05 else {
            throw ColonyAnalysisError.plateNotDetected
        }
        let ellipse = try fitPlateEllipse(
            mask: filled,
            width: width,
            height: height
        )
        let axisRatio = min(ellipse.axis1, ellipse.axis2)
            / max(ellipse.axis1, ellipse.axis2)
        guard axisRatio >= 0.55 else {
            throw ColonyAnalysisError.plateNotDetected
        }

        return PlateMask(
            width: width,
            height: height,
            ellipse: ellipse
        )
    }

    private func appendPixel(
        _ index: Int,
        label: Int32,
        foreground: [UInt8],
        labels: inout [Int32],
        queue: inout [Int]
    ) {
        guard foreground[index] != 0, labels[index] == 0 else {
            return
        }
        labels[index] = label
        queue.append(index)
    }

    private func erode(
        mask: [UInt8],
        width: Int,
        height: Int,
        radius: Int
    ) -> [UInt8] {
        guard radius > 0 else {
            return mask
        }

        let prefixWidth = width + 1
        var prefix = [Int](repeating: 0, count: prefixWidth * (height + 1))
        for y in 0..<height {
            var rowSum = 0
            for x in 0..<width {
                rowSum += Int(mask[y * width + x])
                prefix[(y + 1) * prefixWidth + x + 1] =
                    prefix[y * prefixWidth + x + 1] + rowSum
            }
        }

        var result = [UInt8](repeating: 0, count: mask.count)
        let diameter = radius * 2 + 1
        let requiredArea = diameter * diameter

        guard diameter <= width, diameter <= height else {
            return result
        }

        for y in radius..<(height - radius) {
            for x in radius..<(width - radius) where mask[y * width + x] != 0 {
                let x1 = x - radius
                let y1 = y - radius
                let x2 = x + radius + 1
                let y2 = y + radius + 1
                let sum = prefix[y2 * prefixWidth + x2]
                    - prefix[y1 * prefixWidth + x2]
                    - prefix[y2 * prefixWidth + x1]
                    + prefix[y1 * prefixWidth + x1]
                if sum == requiredArea {
                    result[y * width + x] = 1
                }
            }
        }
        return result
    }

    private func dilate(
        mask: [UInt8],
        width: Int,
        height: Int,
        radius: Int
    ) -> [UInt8] {
        guard radius > 0 else { return mask }
        let prefixWidth = width + 1
        var prefix = [Int](repeating: 0, count: prefixWidth * (height + 1))
        for y in 0..<height {
            var rowSum = 0
            for x in 0..<width {
                rowSum += Int(mask[y * width + x])
                prefix[(y + 1) * prefixWidth + x + 1] =
                    prefix[y * prefixWidth + x + 1] + rowSum
            }
        }

        var result = [UInt8](repeating: 0, count: mask.count)
        for y in 0..<height {
            for x in 0..<width {
                let x1 = max(0, x - radius)
                let y1 = max(0, y - radius)
                let x2 = min(width, x + radius + 1)
                let y2 = min(height, y + radius + 1)
                let sum = prefix[y2 * prefixWidth + x2]
                    - prefix[y1 * prefixWidth + x2]
                    - prefix[y2 * prefixWidth + x1]
                    + prefix[y1 * prefixWidth + x1]
                if sum > 0 {
                    result[y * width + x] = 1
                }
            }
        }
        return result
    }

    private func fillHoles(
        mask: [UInt8],
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
            if x > 0 { enqueue(index - 1) }
            if x + 1 < width { enqueue(index + 1) }
            if y > 0 { enqueue(index - width) }
            if y + 1 < height { enqueue(index + width) }
        }

        return mask.indices.map { index in
            mask[index] != 0 || exterior[index] == 0 ? 1 : 0
        }
    }

    private func fitPlateEllipse(
        mask: [UInt8],
        width: Int,
        height: Int
    ) throws -> PlateEllipse {
        var count = 0.0
        var sumX = 0.0
        var sumY = 0.0
        for index in mask.indices where mask[index] != 0 {
            count += 1
            sumX += Double(index % width) + 0.5
            sumY += Double(index / width) + 0.5
        }
        guard count >= 20 else {
            throw ColonyAnalysisError.plateNotDetected
        }

        let centerX = sumX / count
        let centerY = sumY / count
        var covarianceXX = 0.0
        var covarianceYY = 0.0
        var covarianceXY = 0.0
        for index in mask.indices where mask[index] != 0 {
            let dx = Double(index % width) + 0.5 - centerX
            let dy = Double(index / width) + 0.5 - centerY
            covarianceXX += dx * dx
            covarianceYY += dy * dy
            covarianceXY += dx * dy
        }
        covarianceXX /= count
        covarianceYY /= count
        covarianceXY /= count

        let difference = covarianceXX - covarianceYY
        let discriminant = sqrt(
            max(0, difference * difference + 4 * covarianceXY * covarianceXY)
        )
        let largestEigenvalue = max(
            1,
            (covarianceXX + covarianceYY + discriminant) * 0.5
        )
        let smallestEigenvalue = max(
            1,
            (covarianceXX + covarianceYY - discriminant) * 0.5
        )
        return PlateEllipse(
            centerX: centerX,
            centerY: centerY,
            axis1: 4 * sqrt(largestEigenvalue),
            axis2: 4 * sqrt(smallestEigenvalue),
            angle: 0.5 * atan2(2 * covarianceXY, difference)
        )
    }

    private func makePlateCrop(
        plateMask: PlateMask,
        sourceWidth: Int,
        sourceHeight: Int,
        paddingRatio: Double
    ) -> PixelRect {
        let xScale = Double(sourceWidth) / Double(plateMask.width)
        let yScale = Double(sourceHeight) / Double(plateMask.height)
        let centerX = plateMask.ellipse.centerX * xScale
        let centerY = plateMask.ellipse.centerY * yScale
        let majorAxis = max(
            plateMask.ellipse.axis1,
            plateMask.ellipse.axis2
        )
        let requestedSide = max(majorAxis * xScale, majorAxis * yScale)
            * max(1, paddingRatio)
        let side = max(
            1,
            min(Int(requestedSide.rounded(.up)), min(sourceWidth, sourceHeight))
        )
        let x = min(
            max(0, Int((centerX - Double(side) * 0.5).rounded())),
            sourceWidth - side
        )
        let y = min(
            max(0, Int((centerY - Double(side) * 0.5).rounded())),
            sourceHeight - side
        )
        return PixelRect(x: x, y: y, width: side, height: side)
    }

    private func makeCountingMask(
        plateMask: PlateMask,
        plateCrop: PixelRect,
        sourceWidth: Int,
        sourceHeight: Int,
        roiSize: Int,
        countingScale: Double
    ) -> [UInt8] {
        let ellipse = plateMask.ellipse
        let cosine = cos(ellipse.angle)
        let sine = sin(ellipse.angle)
        let semiAxis1 = max(1, ellipse.axis1 * countingScale * 0.5)
        let semiAxis2 = max(1, ellipse.axis2 * countingScale * 0.5)
        let maskXScale = Double(plateMask.width) / Double(sourceWidth)
        let maskYScale = Double(plateMask.height) / Double(sourceHeight)
        let cropX = Double(plateCrop.x)
        let cropY = Double(plateCrop.y)
        let cropWidth = Double(plateCrop.width)
        let cropHeight = Double(plateCrop.height)
        let roi = Double(roiSize)
        var output = [UInt8](repeating: 0, count: roiSize * roiSize)

        for y in 0..<roiSize {
            let sourceY = cropY + (Double(y) + 0.5) / roi * cropHeight
            let maskY = sourceY * maskYScale
            for x in 0..<roiSize {
                let sourceX = cropX + (Double(x) + 0.5) / roi * cropWidth
                let maskX = sourceX * maskXScale
                let dx = maskX - ellipse.centerX
                let dy = maskY - ellipse.centerY
                let rotatedX = dx * cosine + dy * sine
                let rotatedY = -dx * sine + dy * cosine
                let normalizedRadius = rotatedX * rotatedX
                        / (semiAxis1 * semiAxis1)
                    + rotatedY * rotatedY / (semiAxis2 * semiAxis2)
                if normalizedRadius <= 1 {
                    output[y * roiSize + x] = 1
                }
            }
        }
        return output
    }

    private func filteredAndMergedBoxes(
        _ boxes: [ColonyBox],
        countingMask: [UInt8],
        roiSize: Int,
        iouThreshold: Double,
        duplicateCenterDistance: Double
    ) -> [ColonyBox] {
        let merged = CenterNetDecoder.nonMaximumSuppression(
            boxes,
            iouThreshold: iouThreshold
        )
        let withoutNearbyDuplicates = CenterNetDecoder.suppressNearbyCenters(
            merged,
            minimumDistance: duplicateCenterDistance
        )
        return CenterNetDecoder.keepCentersInsideMask(
            withoutNearbyDuplicates,
            mask: countingMask,
            width: roiSize,
            height: roiSize
        )
    }

    private func mapBoxesToSourceImage(
        _ boxes: [ColonyBox],
        plateCrop: PixelRect,
        roiSize: Int,
        sourceWidth: Int,
        sourceHeight: Int
    ) -> [ColonyBox] {
        let xScale = Double(plateCrop.width) / Double(roiSize)
        let yScale = Double(plateCrop.height) / Double(roiSize)
        let cropX = Double(plateCrop.x)
        let cropY = Double(plateCrop.y)
        let maximumX = Double(sourceWidth)
        let maximumY = Double(sourceHeight)

        return boxes.compactMap { box in
            let x1 = min(maximumX, max(0, cropX + box.x1 * xScale))
            let y1 = min(maximumY, max(0, cropY + box.y1 * yScale))
            let x2 = min(maximumX, max(0, cropX + box.x2 * xScale))
            let y2 = min(maximumY, max(0, cropY + box.y2 * yScale))

            guard x2 > x1, y2 > y1 else { return nil }
            return ColonyBox(
                x1: x1,
                y1: y1,
                x2: x2,
                y2: y2,
                score: box.score
            )
        }
    }
}
