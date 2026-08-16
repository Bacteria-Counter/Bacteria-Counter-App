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
    /// Seberapa besar lingkaran crop dilebihkan dari radius mayoritas mask.
    static let paddingRatio: CGFloat = 0.08

    /// Persentil jarak yang dipakai sebagai radius. 0.95 berarti radius
    /// dipilih sedemikian rupa sehingga 95% pixel foreground berada di
    /// dalam lingkaran — pixel outlier (misalnya salah deteksi bagian lain
    /// yang menonjol jauh) diabaikan.
    static let radiusPercentile: Double = 0.95

    static func crop(
        image: CGImage,
        mask: CGImage
    ) throws -> DishCropResult {
        let (center, radius) = try boundingCircle(of: mask)

        let scaleX = CGFloat(image.width) / CGFloat(mask.width)
        let scaleY = CGFloat(image.height) / CGFloat(mask.height)
        let scale = (scaleX + scaleY) / 2

        let paddedRadius = radius * scale * (1 + paddingRatio)
        let centerInImage = CGPoint(x: center.x * scaleX, y: center.y * scaleY)

        return try makeCircularCrop(
            from: image,
            center: centerInImage,
            radius: paddedRadius
        )
    }

    /// Mencari titik pusat (median, robust terhadap outlier) dan radius
    /// (persentil jarak, bukan jarak maksimum) dari pixel foreground mask,
    /// sehingga lingkaran yang dihasilkan mengikuti mayoritas bentuk bulat
    /// meskipun ada bagian lain yang ikut tersegmentasi secara tidak akurat.
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

        var xs: [Int] = []
        var ys: [Int] = []
        xs.reserveCapacity(width * height / 4)
        ys.reserveCapacity(width * height / 4)

        for y in 0..<height {
            let rowStart = y * bytesPerRow
            for x in 0..<width {
                if pointer[rowStart + x] != 0 {
                    xs.append(x)
                    ys.append(y)
                }
            }
        }

        guard !xs.isEmpty else {
            throw DishCropError.emptyMask
        }

        // Median jauh lebih tahan terhadap outlier dibanding mean: kalau
        // ada gumpalan salah-deteksi di satu sisi, mean akan tertarik ke
        // sana, sedangkan median tetap merepresentasikan bentuk mayoritas.
        let centerX = median(of: xs)
        let centerY = median(of: ys)
        let center = CGPoint(x: centerX, y: centerY)

        var distances: [Double] = []
        distances.reserveCapacity(xs.count)
        for index in xs.indices {
            let dx = Double(xs[index]) - centerX
            let dy = Double(ys[index]) - centerY
            distances.append((dx * dx + dy * dy).squareRoot())
        }
        distances.sort()

        // Radius dipilih dari persentil, bukan jarak maksimum, supaya
        // pixel outlier (bagian lain yang salah tersegmentasi dan
        // menonjol jauh dari bentuk utama) tidak ikut memperbesar lingkaran.
        let percentileIndex = min(
            distances.count - 1,
            Int((Double(distances.count) * radiusPercentile).rounded())
        )
        let radius = CGFloat(distances[percentileIndex])

        return (center, radius)
    }

    private static func median(of values: [Int]) -> Double {
        let sorted = values.sorted()
        let count = sorted.count
        if count % 2 == 0 {
            return Double(sorted[count / 2 - 1] + sorted[count / 2]) / 2
        } else {
            return Double(sorted[count / 2])
        }
    }

    // makeCircularCrop tetap sama seperti sebelumnya, tidak ada perubahan.
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
