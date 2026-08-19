//
//  BackgroundRemover.swift
//  bacteriaapp
//
//  Created by Regina Celine Adiwinata on 06/08/26.
//

import CoreGraphics
import Foundation

nonisolated struct DishCropResult: @unchecked Sendable {
    let image: CGImage
    let radius: CGFloat
    let center: CGPoint
}

nonisolated enum DishCropError: LocalizedError {
    case emptyMask
    case cannotReadMaskData
    case cannotCreateCroppedImage

    var errorDescription: String? {
        switch self {
        case .emptyMask:
            "The segmentation mask has no foreground pixels."
        case .cannotReadMaskData:
            "The segmentation mask pixel data could not be read."
        case .cannotCreateCroppedImage:
            "The cropped dish image could not be created."
        }
    }
}

nonisolated enum DishCropper {
    static let paddingRatio: CGFloat = 0.08

    static let radiusPercentile: Double = 0.95

    static func crop(
        image: CGImage,
        mask: CGImage
    ) throws -> DishCropResult {

        let scaleX = CGFloat(image.width) / CGFloat(mask.width)
        let scaleY = CGFloat(image.height) / CGFloat(mask.height)

        let (center, radius) = try boundingCircle(of: mask, scaleX: scaleX, scaleY: scaleY)

        let limit = CGFloat(min(image.width, image.height)) / 2
        let padded = min(radius * (1 + paddingRatio), limit)

        return try makeCircularCrop(from: image, center: center, radius: padded)
    }

    private static func boundingCircle(
        of mask: CGImage,
        scaleX: CGFloat,
        scaleY: CGFloat
    ) throws -> (center: CGPoint, radius: CGFloat) {
        let width = mask.width
        let height = mask.height

        guard let data = mask.dataProvider?.data,
              let pointer = CFDataGetBytePtr(data) else {
            throw DishCropError.cannotReadMaskData
        }
        let bytesPerRow = mask.bytesPerRow

        var xs: [Double] = []
        var ys: [Double] = []
        xs.reserveCapacity(width * height / 4)
        ys.reserveCapacity(width * height / 4)

        for y in 0..<height {
            let rowStart = y * bytesPerRow
            for x in 0..<width {
                if pointer[rowStart + x] != 0 {
                    xs.append((Double(x) + 0.5) * Double(scaleX))
                    ys.append((Double(y) + 0.5) * Double(scaleY))
                }
            }
        }

        guard !xs.isEmpty else {
            throw DishCropError.emptyMask
        }

        let centerX = median(of: xs)
        let centerY = median(of: ys)
        let center = CGPoint(x: centerX, y: centerY)

        var distances: [Double] = []
        distances.reserveCapacity(xs.count)
        for index in xs.indices {
            let dx = xs[index] - centerX
            let dy = ys[index] - centerY
            distances.append((dx * dx + dy * dy).squareRoot())
        }
        distances.sort()

        let percentileIndex = min(
            distances.count - 1,
            Int((Double(distances.count) * radiusPercentile).rounded())
        )
        let radius = CGFloat(distances[percentileIndex])

        return (center, radius)
    }

    private static func median(of values: [Double]) -> Double {
        let sorted = values.sorted()
        let count = sorted.count
        if count % 2 == 0 {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        } else {
            return sorted[count / 2]
        }
    }

    private static func makeCircularCrop(
        from image: CGImage,
        center: CGPoint,
        radius: CGFloat
    ) throws -> DishCropResult {
        let diameter = Int((radius * 2).rounded(.up))
        guard diameter > 0 else {
            throw DishCropError.cannotCreateCroppedImage
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: diameter,
            height: diameter,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw DishCropError.cannotCreateCroppedImage
        }

        context.addEllipse(in: CGRect(x: 0, y: 0, width: diameter, height: diameter))
        context.clip()

        let originX = center.x - radius
        let originYFromTop = center.y - radius
        let originY = CGFloat(image.height) - originYFromTop - CGFloat(diameter)

        context.draw(
            image,
            in: CGRect(
                x: -originX,
                y: -originY,
                width: CGFloat(image.width),
                height: CGFloat(image.height)
            )
        )

        guard let cropped = context.makeImage() else {
            throw DishCropError.cannotCreateCroppedImage
        }

        return DishCropResult(image: cropped, radius: radius, center: center)
    }
}
