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
        // The mask comes back 512x512 square because the segmenter feeds Vision
        // with .scaleFill, which STRETCHES the photo rather than letterboxing
        // it. A round dish therefore arrives as an ellipse whenever the photo is
        // not square, and the lab's photos are 3:4.
        //
        // So the circle is fitted in the ORIGINAL image's coordinates, not the
        // mask's: every foreground pixel is mapped back per axis first, and only
        // then does a centre and a radius mean anything. Scaling a radius
        // measured in the stretched space cannot be made correct by any single
        // factor -- the previous code used the mean of the two axis scales,
        // which on a 3:4 photo inflated the radius by about 17% before the 8%
        // padding was even added, and that is the ring of bench paper that ended
        // up inside the crop.
        let scaleX = CGFloat(image.width) / CGFloat(mask.width)
        let scaleY = CGFloat(image.height) / CGFloat(mask.height)

        let (center, radius) = try boundingCircle(of: mask, scaleX: scaleX, scaleY: scaleY)

        // Clamped so the crop can never come out BIGGER than the photo it came
        // from. makeCircularCrop allocates a diameter x diameter canvas with no
        // reference to the source size, so an over-covering mask produced a crop
        // with more pixels than the original -- which made every downstream
        // stage slower than not cropping at all, the opposite of the point.
        // A dish that is fully in frame cannot be wider than the short side, and
        // anything past that edge is black padding carrying no colonies.
        let limit = CGFloat(min(image.width, image.height)) / 2
        let padded = min(radius * (1 + paddingRatio), limit)

        return try makeCircularCrop(from: image, center: center, radius: padded)
    }

    /// Mencari titik pusat (median, robust terhadap outlier) dan radius
    /// (persentil jarak, bukan jarak maksimum) dari pixel foreground mask,
    /// sehingga lingkaran yang dihasilkan mengikuti mayoritas bentuk bulat
    /// meskipun ada bagian lain yang ikut tersegmentasi secara tidak akurat.
    /// - Parameters scaleX/scaleY: mask pixels to original-image pixels, per
    ///   axis. Applied while collecting, so the centre and every distance below
    ///   are already in the space where the dish is actually round.
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

        // Mapped to original-image pixels as they are collected -- see crop().
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

        // Median jauh lebih tahan terhadap outlier dibanding mean: kalau
        // ada gumpalan salah-deteksi di satu sisi, mean akan tertarik ke
        // sana, sedangkan median tetap merepresentasikan bentuk mayoritas.
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

    private static func median(of values: [Double]) -> Double {
        let sorted = values.sorted()
        let count = sorted.count
        if count % 2 == 0 {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2
        } else {
            return sorted[count / 2]
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
