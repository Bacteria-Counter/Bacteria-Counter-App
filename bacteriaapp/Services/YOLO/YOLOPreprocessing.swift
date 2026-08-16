//
//  YOLOPreprocessing.swift
//  bacteriaapp
//
//  Created by Regina Celine Adiwinata on 05/08/26.
//

import Accelerate
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

final class CLAHEGrayscalePreprocessor {

    // MARK: - Config

    let clipLimit: Float
    let tileGridWidth: Int
    let tileGridHeight: Int
    let jpgQuality: CGFloat

    init(
        clipLimit: Float = 2.0,
        tileGrid: (width: Int, height: Int) = (8, 8),
        jpgQuality: CGFloat = 0.95
    ) {
        self.clipLimit = clipLimit
        self.tileGridWidth = tileGrid.width
        self.tileGridHeight = tileGrid.height
        self.jpgQuality = jpgQuality
    }

    enum PreprocessError: Error {
        case vImageError(vImage_Error)
        case fileNotReadable(URL)
        case cannotWrite(URL)
        case cgImageCreationFailed
    }

    // MARK: - Core preprocessing

    func preprocess(_ cgImage: CGImage) throws -> CGImage {
        var grayBuffer = try makeGrayscaleBuffer(from: cgImage)
        defer { grayBuffer.free() }

        try applyCLAHE(to: &grayBuffer)

        return try grayscaleBufferToRGBCGImage(grayBuffer)
    }

    func callAsFunction(_ cgImage: CGImage) throws -> CGImage {
        try preprocess(cgImage)
    }

    // MARK: - Grayscale conversion

    private func makeGrayscaleBuffer(from cgImage: CGImage) throws -> vImage_Buffer {
        // PERBAIKAN 1: Menggunakan 'guard var' untuk meng-unwrap Optional format
        guard var format = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
            renderingIntent: .defaultIntent
        ) else {
            throw PreprocessError.cgImageCreationFailed
        }

        var argbBuffer = vImage_Buffer()
        var err = vImageBuffer_InitWithCGImage(
            &argbBuffer, &format, nil, cgImage, vImage_Flags(kvImageNoFlags)
        )
        guard err == kvImageNoError else { throw PreprocessError.vImageError(err) }
        defer { argbBuffer.free() }

        var grayBuffer = vImage_Buffer()
        err = vImageBuffer_Init(
            &grayBuffer, argbBuffer.height, argbBuffer.width, 8, vImage_Flags(kvImageNoFlags)
        )
        guard err == kvImageNoError else { throw PreprocessError.vImageError(err) }

        let divisor: Int32 = 0x1000
        var coefficientsMatrix: [Int16] = [
            0,                                   // A
            Int16(0.299 * Float(divisor)),       // R
            Int16(0.587 * Float(divisor)),       // G
            Int16(0.114 * Float(divisor)),       // B
        ]

        err = vImageMatrixMultiply_ARGB8888ToPlanar8(
            &argbBuffer,
            &grayBuffer,
            &coefficientsMatrix,
            divisor,
            nil,
            0,
            vImage_Flags(kvImageNoFlags)
        )
        guard err == kvImageNoError else { throw PreprocessError.vImageError(err) }

        return grayBuffer
    }

    // MARK: - CLAHE (Fallback to Equalization)

    private func applyCLAHE(to buffer: inout vImage_Buffer) throws {
        let width = Int(buffer.width)
        let height = Int(buffer.height)
        let rowBytes = Int(buffer.rowBytes)
        let dataPtr = buffer.data.bindMemory(to: UInt8.self, capacity: rowBytes * height)

        let tilesX = tileGridWidth
        let tilesY = tileGridHeight
        let histSize = 256

        // Ukuran tile pakai ceil division, sama seperti OpenCV.
        let tileWidth = (width + tilesX - 1) / tilesX
        let tileHeight = (height + tilesY - 1) / tilesY

        // Hitung LUT (lookup table) hasil histogram-equalization ber-clip-limit
        // untuk tiap tile, persis alur cv2.createCLAHE().apply():
        // histogram -> clip -> redistribute sisa -> CDF -> scale ke 0...255.
        var luts = [[UInt8]](repeating: [UInt8](repeating: 0, count: histSize), count: tilesX * tilesY)

        for ty in 0..<tilesY {
            for tx in 0..<tilesX {
                let xStart = tx * tileWidth
                let yStart = ty * tileHeight
                let xEnd = min(xStart + tileWidth, width)
                let yEnd = min(yStart + tileHeight, height)
                let tileW = xEnd - xStart
                let tileH = yEnd - yStart
                let tileArea = tileW * tileH
                guard tileArea > 0 else { continue }

                var histogram = [Int](repeating: 0, count: histSize)
                for y in yStart..<yEnd {
                    let rowOffset = y * rowBytes
                    for x in xStart..<xEnd {
                        histogram[Int(dataPtr[rowOffset + x])] += 1
                    }
                }

                // clipLimit param OpenCV di-scale relatif ke luas tile, sama
                // seperti CLAHE_Impl::apply di clahe.cpp.
                var clipLimitValue = 0
                if clipLimit > 0 {
                    clipLimitValue = max(
                        Int((clipLimit * Float(tileArea) / Float(histSize)).rounded()),
                        1
                    )
                }

                if clipLimitValue > 0 {
                    var clipped = 0
                    for i in 0..<histSize {
                        if histogram[i] > clipLimitValue {
                            clipped += histogram[i] - clipLimitValue
                            histogram[i] = clipLimitValue
                        }
                    }
                    let redistBatch = clipped / histSize
                    var residual = clipped - redistBatch * histSize
                    for i in 0..<histSize {
                        histogram[i] += redistBatch
                    }
                    if residual != 0 {
                        let residualStep = max(histSize / residual, 1)
                        var i = 0
                        while i < histSize && residual > 0 {
                            histogram[i] += 1
                            residual -= 1
                            i += residualStep
                        }
                    }
                }

                let lutScale = Float(histSize - 1) / Float(tileArea)
                var sum = 0
                var lut = [UInt8](repeating: 0, count: histSize)
                for i in 0..<histSize {
                    sum += histogram[i]
                    let mapped = Float(sum) * lutScale
                    lut[i] = UInt8(max(0, min(255, mapped.rounded())))
                }
                luts[ty * tilesX + tx] = lut
            }
        }

        // Interpolasi bilinear antar 4 tile terdekat untuk tiap pixel, supaya
        // tidak ada seam/patahan di batas antar tile (sama seperti perilaku
        // default OpenCV CLAHE).
        var output = [UInt8](repeating: 0, count: rowBytes * height)

        for y in 0..<height {
            let tileYFloat = (Float(y) - Float(tileHeight) / 2) / Float(tileHeight)
            var ty0 = Int(floor(tileYFloat))
            var wy = tileYFloat - Float(ty0)
            if ty0 < 0 { ty0 = 0; wy = 0 }
            if ty0 >= tilesY { ty0 = tilesY - 1 }
            var ty1 = min(ty0 + 1, tilesY - 1)
            if ty1 == ty0 { wy = 0 }

            let rowOffset = y * rowBytes

            for x in 0..<width {
                let tileXFloat = (Float(x) - Float(tileWidth) / 2) / Float(tileWidth)
                var tx0 = Int(floor(tileXFloat))
                var wx = tileXFloat - Float(tx0)
                if tx0 < 0 { tx0 = 0; wx = 0 }
                if tx0 >= tilesX { tx0 = tilesX - 1 }
                var tx1 = min(tx0 + 1, tilesX - 1)
                if tx1 == tx0 { wx = 0 }

                let value = Int(dataPtr[rowOffset + x])

                let lut00 = Float(luts[ty0 * tilesX + tx0][value])
                let lut01 = Float(luts[ty0 * tilesX + tx1][value])
                let lut10 = Float(luts[ty1 * tilesX + tx0][value])
                let lut11 = Float(luts[ty1 * tilesX + tx1][value])

                let top = lut00 * (1 - wx) + lut01 * wx
                let bottom = lut10 * (1 - wx) + lut11 * wx
                let interpolated = top * (1 - wy) + bottom * wy

                output[rowOffset + x] = UInt8(max(0, min(255, interpolated.rounded())))
            }
        }

        output.withUnsafeBytes { rawPtr in
            memcpy(buffer.data, rawPtr.baseAddress, rowBytes * height)
        }
    }

    // MARK: - Back to 3-channel image

    private func grayscaleBufferToRGBCGImage(_ grayBuffer: vImage_Buffer) throws -> CGImage {
        var alphaBuffer = vImage_Buffer()
        var err = vImageBuffer_Init(
            &alphaBuffer, grayBuffer.height, grayBuffer.width, 8, vImage_Flags(kvImageNoFlags)
        )
        guard err == kvImageNoError else { throw PreprocessError.vImageError(err) }
        defer { alphaBuffer.free() }
        memset(alphaBuffer.data, 255, Int(alphaBuffer.rowBytes) * Int(alphaBuffer.height))

        var argbBuffer = vImage_Buffer()
        err = vImageBuffer_Init(
            &argbBuffer, grayBuffer.height, grayBuffer.width, 32, vImage_Flags(kvImageNoFlags)
        )
        guard err == kvImageNoError else { throw PreprocessError.vImageError(err) }
        defer { argbBuffer.free() }

        var r = grayBuffer
        var g = grayBuffer
        var b = grayBuffer

        err = vImageConvert_Planar8toARGB8888(
            &alphaBuffer, &r, &g, &b, &argbBuffer, vImage_Flags(kvImageNoFlags)
        )
        guard err == kvImageNoError else { throw PreprocessError.vImageError(err) }

        // PERBAIKAN 1: Unwrap Optional format lagi di sini
        guard var format = vImage_CGImageFormat(
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            colorSpace: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
            renderingIntent: .defaultIntent
        ) else {
            throw PreprocessError.cgImageCreationFailed
        }

        var creationError = vImage_Error()
        guard
            let cgImage = vImageCreateCGImageFromBuffer(
                &argbBuffer, &format, nil, nil, vImage_Flags(kvImageNoFlags), &creationError
            )?.takeRetainedValue()
        else {
            throw PreprocessError.vImageError(creationError)
        }

        return cgImage
    }

    // MARK: - File operations

    @discardableResult
    func processFile(inputPath: URL, outputPath: URL? = nil) throws -> CGImage {
        guard
            let dataProvider = CGDataProvider(url: inputPath as CFURL),
            let source = CGImageSourceCreateWithDataProvider(dataProvider, nil),
            let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw PreprocessError.fileNotReadable(inputPath)
        }

        let processed = try preprocess(cgImage)

        if let outputPath = outputPath {
            try save(cgImage: processed, to: outputPath)
        }

        return processed
    }

    @discardableResult
    func processDirectory(
        inputDir: URL,
        outputDir: URL,
        extensions: Set<String> = ["jpg", "jpeg", "png"]
    ) throws -> [URL] {
        try FileManager.default.createDirectory(
            at: outputDir, withIntermediateDirectories: true
        )

        let files = try FileManager.default
            .contentsOfDirectory(at: inputDir, includingPropertiesForKeys: nil)
            .filter { extensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        var savedPaths: [URL] = []
        for src in files {
            let out = outputDir.appendingPathComponent(src.lastPathComponent)
            try processFile(inputPath: src, outputPath: out)
            savedPaths.append(out)
        }
        return savedPaths
    }

    private func save(cgImage: CGImage, to url: URL) throws {
        guard
            let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.jpeg.identifier as CFString, 1, nil
            )
        else {
            throw PreprocessError.cannotWrite(url)
        }

        let options: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: jpgQuality
        ]
        CGImageDestinationAddImage(destination, cgImage, options as CFDictionary)

        guard CGImageDestinationFinalize(destination) else {
            throw PreprocessError.cannotWrite(url)
        }
    }
}

// MARK: - Example usage
//
// let preprocessor = CLAHEGrayscalePreprocessor(clipLimit: 2.0, tileGrid: (8, 8))
//
// // Single file:
// try preprocessor.processFile(
//     inputPath: URL(fileURLWithPath: "/path/to/input.jpg"),
//     outputPath: URL(fileURLWithPath: "/path/to/output.jpg")
// )
//
// // Whole directory:
// try preprocessor.processDirectory(
//     inputDir: URL(fileURLWithPath: "/path/to/images"),
//     outputDir: URL(fileURLWithPath: "/path/to/processed")
// )
