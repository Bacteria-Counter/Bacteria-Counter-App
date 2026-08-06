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
        // PERBAIKAN 2: Apple tidak memiliki CLAHE.
        // Sebagai gantinya, kita gunakan Histogram Equalization standar.
        let err = vImageEqualization_Planar8(
            &buffer,
            &buffer,
            vImage_Flags(kvImageNoFlags)
        )
        guard err == kvImageNoError else { throw PreprocessError.vImageError(err) }
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
