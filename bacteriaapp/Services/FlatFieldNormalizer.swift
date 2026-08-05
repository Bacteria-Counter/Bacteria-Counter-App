import Accelerate
import Foundation

nonisolated enum FlatFieldNormalizer {
    struct Configuration: Sendable {
        let safeErodeRatio: Double
        let safeMinimumFraction: Double
        let outlierMADMultiplier: Double
        let sigmaRatio: Double
        let percentileLow: Double
        let percentileHigh: Double
    }

    enum NormalizationError: LocalizedError {
        case invalidInput(String)
        case insufficientPixels(String)
        case imageProcessingFailed(String)

        var errorDescription: String? {
            switch self {
            case .invalidInput(let reason):
                "Invalid flat-field input: \(reason)"
            case .insufficientPixels(let reason):
                "Flat-field normalization needs more valid pixels: \(reason)"
            case .imageProcessingFailed(let reason):
                "Flat-field image processing failed: \(reason)"
            }
        }
    }

    static func normalize(
        rgba: [UInt8],
        width: Int,
        height: Int,
        countingMask: [UInt8],
        configuration: Configuration
    ) throws -> [UInt8] {
        let pixelCount = width * height
        guard width > 0,
              height > 0,
              rgba.count == pixelCount * 4,
              countingMask.count == pixelCount else {
            throw NormalizationError.invalidInput("image and mask dimensions do not match")
        }

        let countingArea = countingMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        guard countingArea >= 100 else {
            throw NormalizationError.insufficientPixels("counting mask is empty")
        }

        let countingSamples = sampledIndices(
            mask: countingMask,
            maximumCount: 250_000
        )
        guard !countingSamples.isEmpty else {
            throw NormalizationError.insufficientPixels("counting mask has no samples")
        }

        let fillColor = medianColor(rgba: rgba, indices: countingSamples)
        var luminance = [Float](repeating: 0, count: pixelCount)
        var labA = [Float](repeating: 0, count: pixelCount)
        var labB = [Float](repeating: 0, count: pixelCount)
        let fillLab = rgbToLab(
            red: fillColor.red,
            green: fillColor.green,
            blue: fillColor.blue
        )

        for index in 0..<pixelCount {
            let lab: LabColor
            if countingMask[index] == 0 {
                lab = fillLab
            } else {
                let source = index * 4
                lab = rgbToLab(
                    red: rgba[source],
                    green: rgba[source + 1],
                    blue: rgba[source + 2]
                )
            }
            luminance[index] = max(1.0 / 255.0, lab.l / 255.0)
            labA[index] = lab.a
            labB[index] = lab.b
        }

        let erodeRadius = max(
            1,
            Int(
                (
                    Double(min(width, height))
                        * max(0, configuration.safeErodeRatio)
                ).rounded()
            )
        )
        var candidateMask = erodeSquare(
            mask: countingMask,
            width: width,
            height: height,
            radius: erodeRadius
        )
        var candidateCount = candidateMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        if candidateCount < 100 {
            candidateMask = countingMask
            candidateCount = countingArea
        }

        let candidateSamples = sampledIndices(
            mask: candidateMask,
            maximumCount: 250_000
        )
        let medianL = median(candidateSamples.map { luminance[$0] * 255 })
        let medianA = median(candidateSamples.map { labA[$0] })
        let medianB = median(candidateSamples.map { labB[$0] })
        let madL = max(
            2,
            median(candidateSamples.map { abs(luminance[$0] * 255 - medianL) })
        )
        let madA = max(
            1,
            median(candidateSamples.map { abs(labA[$0] - medianA) })
        )
        let madB = max(
            1,
            median(candidateSamples.map { abs(labB[$0] - medianB) })
        )

        let madMultiplier = Float(max(0.1, configuration.outlierMADMultiplier))
        var safeMask = [UInt8](repeating: 0, count: pixelCount)
        var safeCount = 0
        for index in 0..<pixelCount where candidateMask[index] != 0 {
            let zL = abs(luminance[index] * 255 - medianL) / madL
            let zA = abs(labA[index] - medianA) / madA
            let zB = abs(labB[index] - medianB) / madB
            if max(zL, max(zA, zB)) <= madMultiplier {
                safeMask[index] = 1
                safeCount += 1
            }
        }

        let minimumSafePixels = max(
            100,
            Int(
                (
                    Double(countingArea)
                        * max(0, configuration.safeMinimumFraction)
                ).rounded()
            )
        )
        if safeCount < minimumSafePixels {
            let distanceSamples = candidateSamples.map { index -> Float in
                let zL = (luminance[index] * 255 - medianL) / madL
                let zA = (labA[index] - medianA) / madA
                let zB = (labB[index] - medianB) / madB
                return sqrt(zL * zL + zA * zA + zB * zB)
            }
            let requiredFraction = min(
                1,
                Double(minimumSafePixels) / Double(max(1, candidateCount))
            )
            let distanceThreshold = percentile(
                distanceSamples,
                percentage: requiredFraction * 100
            )
            safeMask = [UInt8](repeating: 0, count: pixelCount)
            safeCount = 0
            for index in 0..<pixelCount where candidateMask[index] != 0 {
                let zL = (luminance[index] * 255 - medianL) / madL
                let zA = (labA[index] - medianA) / madA
                let zB = (labB[index] - medianB) / madB
                let distance = sqrt(zL * zL + zA * zA + zB * zB)
                if distance <= distanceThreshold {
                    safeMask[index] = 1
                    safeCount += 1
                }
            }
        }

        guard safeCount >= 100 else {
            throw NormalizationError.insufficientPixels(
                "safe medium mask is too small"
            )
        }

        let safeSamples = sampledIndices(mask: safeMask, maximumCount: 250_000)
        let mediumLuminance = median(safeSamples.map { luminance[$0] })
        let sigma = max(
            3,
            Double(min(width, height)) * max(0, configuration.sigmaRatio)
        )
        let background = try normalizedGaussianBackground(
            luminance: luminance,
            safeMask: safeMask,
            width: width,
            height: height,
            sigma: sigma,
            fallback: mediumLuminance
        )

        let epsilon: Float = 1e-6
        var residual = [Float](repeating: 0, count: pixelCount)
        for index in 0..<pixelCount {
            residual[index] = log(max(epsilon, background[index]))
                - log(max(epsilon, luminance[index]))
        }

        let percentileSamples = sampledValues(
            residual,
            mask: countingMask,
            maximumCount: 500_000
        )
        let low = percentile(
            percentileSamples,
            percentage: configuration.percentileLow
        )
        var high = percentile(
            percentileSamples,
            percentage: configuration.percentileHigh
        )
        if high - low < epsilon {
            high = low + epsilon
        }

        var grayscale = [UInt8](repeating: 0, count: pixelCount)
        let range = high - low
        for index in 0..<pixelCount {
            let scaled = min(1, max(0, (residual[index] - low) / range))
            grayscale[index] = UInt8(
                min(255, max(0, Int((scaled * 255).rounded())))
            )
        }

        let fillGray = UInt8(
            min(
                255,
                max(
                    0,
                    Int(
                        median(safeSamples.map { Float(grayscale[$0]) })
                            .rounded()
                    )
                )
            )
        )
        for index in 0..<pixelCount where countingMask[index] == 0 {
            grayscale[index] = fillGray
        }
        return grayscale
    }

    private struct LabColor {
        let l: Float
        let a: Float
        let b: Float
    }

    private struct RGBColor {
        let red: UInt8
        let green: UInt8
        let blue: UInt8
    }

    private static let linearRGB: [Float] = (0...255).map { value in
        let normalized = Float(value) / 255
        return normalized <= 0.04045
            ? normalized / 12.92
            : pow((normalized + 0.055) / 1.055, 2.4)
    }

    private static func rgbToLab(
        red: UInt8,
        green: UInt8,
        blue: UInt8
    ) -> LabColor {
        let r = linearRGB[Int(red)]
        let g = linearRGB[Int(green)]
        let b = linearRGB[Int(blue)]
        let x = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b)
            / 0.950_47
        let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b
        let z = (0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b)
            / 1.088_83
        let fx = labPivot(x)
        let fy = labPivot(y)
        let fz = labPivot(z)
        return LabColor(
            l: (116 * fy - 16) * 2.55,
            a: 500 * (fx - fy) + 128,
            b: 200 * (fy - fz) + 128
        )
    }

    private static func labPivot(_ value: Float) -> Float {
        value > 0.008_856
            ? pow(value, 1.0 / 3.0)
            : 7.787 * value + 16.0 / 116.0
    }

    private static func medianColor(
        rgba: [UInt8],
        indices: [Int]
    ) -> RGBColor {
        var red: [UInt8] = []
        var green: [UInt8] = []
        var blue: [UInt8] = []
        red.reserveCapacity(indices.count)
        green.reserveCapacity(indices.count)
        blue.reserveCapacity(indices.count)
        for index in indices {
            let source = index * 4
            red.append(rgba[source])
            green.append(rgba[source + 1])
            blue.append(rgba[source + 2])
        }
        red.sort()
        green.sort()
        blue.sort()
        return RGBColor(
            red: red[red.count / 2],
            green: green[green.count / 2],
            blue: blue[blue.count / 2]
        )
    }

    private static func sampledIndices(
        mask: [UInt8],
        maximumCount: Int
    ) -> [Int] {
        let total = mask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        guard total > 0 else { return [] }
        let stride = max(1, total / max(1, maximumCount))
        var indices: [Int] = []
        indices.reserveCapacity(min(total, maximumCount + 1))
        var foregroundIndex = 0
        for index in mask.indices where mask[index] != 0 {
            if foregroundIndex % stride == 0 {
                indices.append(index)
            }
            foregroundIndex += 1
        }
        return indices
    }

    private static func sampledValues(
        _ values: [Float],
        mask: [UInt8],
        maximumCount: Int
    ) -> [Float] {
        sampledIndices(mask: mask, maximumCount: maximumCount).map {
            values[$0]
        }
    }

    private static func median(_ values: [Float]) -> Float {
        percentile(values, percentage: 50)
    }

    private static func percentile(
        _ values: [Float],
        percentage: Double
    ) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let position = min(100, max(0, percentage)) / 100
            * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = Int(position.rounded(.up))
        guard lower != upper else { return sorted[lower] }
        let fraction = Float(position - Double(lower))
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }

    private static func erodeSquare(
        mask: [UInt8],
        width: Int,
        height: Int,
        radius: Int
    ) -> [UInt8] {
        guard radius > 0 else {
            return mask.map { $0 == 0 ? 0 : 1 }
        }
        let prefixWidth = width + 1
        var prefix = [Int](repeating: 0, count: prefixWidth * (height + 1))
        for y in 0..<height {
            var rowSum = 0
            for x in 0..<width {
                rowSum += mask[y * width + x] == 0 ? 0 : 1
                prefix[(y + 1) * prefixWidth + x + 1] =
                    prefix[y * prefixWidth + x + 1] + rowSum
            }
        }

        var result = [UInt8](repeating: 0, count: mask.count)
        let diameter = radius * 2 + 1
        let requiredArea = diameter * diameter
        guard diameter <= width, diameter <= height else { return result }
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

    private static func normalizedGaussianBackground(
        luminance: [Float],
        safeMask: [UInt8],
        width: Int,
        height: Int,
        sigma: Double,
        fallback: Float
    ) throws -> [Float] {
        let pixelCount = width * height
        var weightedSignal = [Float](repeating: 0, count: pixelCount)
        var weights = [Float](repeating: 0, count: pixelCount)
        for index in 0..<pixelCount where safeMask[index] != 0 {
            weightedSignal[index] = luminance[index]
            weights[index] = 1
        }

        let maximumLowSide = 384
        let scale = min(
            1,
            Double(maximumLowSide) / Double(max(width, height))
        )
        let lowWidth = max(32, Int((Double(width) * scale).rounded()))
        let lowHeight = max(32, Int((Double(height) * scale).rounded()))
        let lowSignal = try resizePlanar(
            weightedSignal,
            sourceWidth: width,
            sourceHeight: height,
            destinationWidth: lowWidth,
            destinationHeight: lowHeight
        )
        let lowWeights = try resizePlanar(
            weights,
            sourceWidth: width,
            sourceHeight: height,
            destinationWidth: lowWidth,
            destinationHeight: lowHeight
        )
        let lowSigma = max(0.75, sigma * scale)
        let blurredSignal = try gaussianBlur(
            lowSignal,
            width: lowWidth,
            height: lowHeight,
            sigma: lowSigma
        )
        let blurredWeights = try gaussianBlur(
            lowWeights,
            width: lowWidth,
            height: lowHeight,
            sigma: lowSigma
        )
        var lowBackground = [Float](repeating: fallback, count: lowSignal.count)
        for index in lowBackground.indices {
            if blurredWeights[index] >= 0.01 {
                lowBackground[index] = min(
                    1,
                    max(1.0 / 255.0, blurredSignal[index] / blurredWeights[index])
                )
            }
        }
        return try resizePlanar(
            lowBackground,
            sourceWidth: lowWidth,
            sourceHeight: lowHeight,
            destinationWidth: width,
            destinationHeight: height
        )
    }

    private static func resizePlanar(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        destinationWidth: Int,
        destinationHeight: Int
    ) throws -> [Float] {
        var output = [Float](
            repeating: 0,
            count: destinationWidth * destinationHeight
        )
        let error = source.withUnsafeBufferPointer { sourceBuffer in
            output.withUnsafeMutableBufferPointer { destinationBuffer in
                guard let sourceBase = sourceBuffer.baseAddress,
                      let destinationBase = destinationBuffer.baseAddress else {
                    return kvImageNullPointerArgument
                }
                var input = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: sourceBase),
                    height: vImagePixelCount(sourceHeight),
                    width: vImagePixelCount(sourceWidth),
                    rowBytes: sourceWidth * MemoryLayout<Float>.stride
                )
                var destination = vImage_Buffer(
                    data: destinationBase,
                    height: vImagePixelCount(destinationHeight),
                    width: vImagePixelCount(destinationWidth),
                    rowBytes: destinationWidth * MemoryLayout<Float>.stride
                )
                return vImageScale_PlanarF(
                    &input,
                    &destination,
                    nil,
                    vImage_Flags(kvImageHighQualityResampling)
                )
            }
        }
        guard error == kvImageNoError else {
            throw NormalizationError.imageProcessingFailed(
                "vImage planar scaling returned \(error)"
            )
        }
        return output
    }

    private static func gaussianBlur(
        _ source: [Float],
        width: Int,
        height: Int,
        sigma: Double
    ) throws -> [Float] {
        let radius = max(1, Int((sigma * 3).rounded(.up)))
        let kernel = (-radius...radius).map { offset -> Float in
            let value = Double(offset)
            return Float(exp(-(value * value) / (2 * sigma * sigma)))
        }
        let sum = kernel.reduce(0, +)
        let normalizedKernel = kernel.map { $0 / sum }
        var output = [Float](repeating: 0, count: source.count)
        let error = source.withUnsafeBufferPointer { sourceBuffer in
            output.withUnsafeMutableBufferPointer { destinationBuffer in
                normalizedKernel.withUnsafeBufferPointer { kernelBuffer in
                    guard let sourceBase = sourceBuffer.baseAddress,
                          let destinationBase = destinationBuffer.baseAddress,
                          let kernelBase = kernelBuffer.baseAddress else {
                        return kvImageNullPointerArgument
                    }
                    var input = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: sourceBase),
                        height: vImagePixelCount(height),
                        width: vImagePixelCount(width),
                        rowBytes: width * MemoryLayout<Float>.stride
                    )
                    var destination = vImage_Buffer(
                        data: destinationBase,
                        height: vImagePixelCount(height),
                        width: vImagePixelCount(width),
                        rowBytes: width * MemoryLayout<Float>.stride
                    )
                    return vImageSepConvolve_PlanarF(
                        &input,
                        &destination,
                        nil,
                        0,
                        0,
                        kernelBase,
                        UInt32(normalizedKernel.count),
                        kernelBase,
                        UInt32(normalizedKernel.count),
                        0,
                        0,
                        vImage_Flags(kvImageEdgeExtend)
                    )
                }
            }
        }
        guard error == kvImageNoError else {
            throw NormalizationError.imageProcessingFailed(
                "vImage Gaussian blur returned \(error)"
            )
        }
        return output
    }
}
