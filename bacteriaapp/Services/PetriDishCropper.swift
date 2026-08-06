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
    /// Seberapa besar lingkaran crop dilebihkan dari lingkaran pembungkus
    /// mask yang sebenarnya. 0.08 = radius diperbesar 8%.
    static let paddingRatio: CGFloat = 0.08

    /// Meng-crop `image` menjadi area lingkaran (dengan padding) di sekitar
    /// dish yang tersegmentasi.
    /// - Parameters:
    ///   - image: Gambar asli full-resolution.
    ///   - mask: Binary mask hasil `PetriDishSegmenter` (biasanya 512x512).
    /// - Returns: Gambar persegi berisi crop lingkaran dish, area di luar
    ///   lingkaran transparan.
    static func crop(
        image: CGImage,
        mask: CGImage
    ) throws -> DishCropResult {
        let (center, radius) = try boundingCircle(of: mask)

        // Skala dari koordinat mask ke koordinat gambar asli.
        let scaleX = CGFloat(image.width) / CGFloat(mask.width)
        let scaleY = CGFloat(image.height) / CGFloat(mask.height)
        let scale = (scaleX + scaleY) / 2 // rata-rata biar lingkaran tetap bulat

        let paddedRadius = radius * scale * (1 + paddingRatio)
        let centerInImage = CGPoint(x: center.x * scaleX, y: center.y * scaleY)

        return try makeCircularCrop(
            from: image,
            center: centerInImage,
            radius: paddedRadius
        )
    }

    /// Mencari titik pusat (centroid) dan radius maksimum dari pixel
    /// foreground di mask (dalam koordinat pixel mask).
    private static func boundingCircle(
        of mask: CGImage
    ) throws -> (center: CGPoint, radius: CGFloat) {
        let width = mask.width
        let height = mask.height

        guard let data = mask.dataProvider?.data,
              let pointer = CFDataGetBytePtr(data) else {
            throw DishCropError.cannotReadMaskData
        }
        let bytesPerRow = mask.bytesPerRow

        var sumX = 0.0
        var sumY = 0.0
        var count = 0

        for y in 0..<height {
            let rowStart = y * bytesPerRow
            for x in 0..<width {
                if pointer[rowStart + x] != 0 {
                    sumX += Double(x)
                    sumY += Double(y)
                    count += 1
                }
            }
        }

        guard count > 0 else {
            throw DishCropError.emptyMask
        }

        let centerX = sumX / Double(count)
        let centerY = sumY / Double(count)
        let center = CGPoint(x: centerX, y: centerY)

        var maxDistanceSquared = 0.0
        for y in 0..<height {
            let rowStart = y * bytesPerRow
            for x in 0..<width {
                guard pointer[rowStart + x] != 0 else { continue }
                let dx = Double(x) - centerX
                let dy = Double(y) - centerY
                let distanceSquared = dx * dx + dy * dy
                if distanceSquared > maxDistanceSquared {
                    maxDistanceSquared = distanceSquared
                }
            }
        }

        return (center, CGFloat(maxDistanceSquared.squareRoot()))
    }

    /// Menggambar `image` di-crop persegi di sekitar `center`/`radius`,
    /// diklip jadi lingkaran (transparan di luar).
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

        // Klip jadi lingkaran, area luar dish jadi transparan.
        context.addEllipse(in: CGRect(x: 0, y: 0, width: diameter, height: diameter))
        context.clip()

        // CGContext punya origin kiri-bawah, sedangkan CGImage kiri-atas,
        // jadi konversi dulu titik crop-nya.
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
