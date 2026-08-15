import Accelerate
import CoreGraphics
import CoreImage
import Foundation
import ImageIO

/// A plain BGR/RGB byte buffer, matching how the Python pipeline holds images
/// so the two can be compared value for value during the port.
struct Bitmap {
    var width: Int
    var height: Int
    /// Interleaved RGBA8, the layout CoreGraphics gives us without a conversion pass.
    var rgba: [UInt8]

    subscript(x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8) {
        let i = (y * width + x) * 4
        return (rgba[i], rgba[i + 1], rgba[i + 2])
    }
}

enum ImageOps {
    static func load(_ path: String) -> Bitmap? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        return from(cg)
    }

    static func from(_ cg: CGImage) -> Bitmap? {
        let w = cg.width, h = cg.height
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        // Draw into the image's OWN colour space, not DeviceRGB. Converting
        // between the two shifts every channel by a few units -- measured at 1-4
        // on a real plate photo -- and OpenCV, which the whole pipeline was
        // calibrated against, performs no colour management at all when reading
        // a JPEG. Matching its raw values matters more than being colorimetric.
        let space = cg.colorSpace ?? CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &buf, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return Bitmap(width: w, height: h, rgba: buf)
    }

    static func cgImage(_ bmp: Bitmap) -> CGImage? {
        var buf = bmp.rgba
        return buf.withUnsafeMutableBytes { raw -> CGImage? in
            guard let ctx = CGContext(data: raw.baseAddress, width: bmp.width, height: bmp.height,
                                      bitsPerComponent: 8, bytesPerRow: bmp.width * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return nil }
            return ctx.makeImage()
        }
    }

    static func pngData(_ cg: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    /// Scale to fit `targetW`x`targetH` preserving aspect, pad the remainder with
    /// grey 114 centred -- ultralytics' LetterBox with auto=False, which is what
    /// the exported Core ML models were traced against.
    static func letterbox(_ bmp: Bitmap, targetW: Int, targetH: Int) -> Bitmap {
        let scale = min(Double(targetW) / Double(bmp.width),
                        Double(targetH) / Double(bmp.height))
        let newW = Int((Double(bmp.width) * scale).rounded())
        let newH = Int((Double(bmp.height) * scale).rounded())
        // Ultralytics splits the leftover evenly and rounds the same way; matching
        // it matters because the pad offset shifts every box coordinate later.
        let padW = Double(targetW - newW) / 2.0
        let padH = Double(targetH - newH) / 2.0
        let left = Int((padW - 0.1).rounded())
        let top = Int((padH - 0.1).rounded())

        var out = [UInt8](repeating: 114, count: targetW * targetH * 4)
        for i in stride(from: 3, to: out.count, by: 4) { out[i] = 255 }

        // Resampled here rather than by CoreGraphics. Its interpolation is a
        // different algorithm from cv2.resize's INTER_LINEAR, and that showed up
        // exactly where a rescale happens: benchmark plates letterbox at scale
        // 1.0 and matched Python closely, while lab photos rescale by 1.4 and
        // diverged by 10%. Same reason the sam_micro escalation to 4480 was the
        // worst case -- it is the largest rescale in the pipeline.
        let scaled = resizeBilinear(bmp, newW: newW, newH: newH)
        for y in 0..<newH {
            let dy = y + top
            if dy < 0 || dy >= targetH { continue }
            for x in 0..<newW {
                let dxp = x + left
                if dxp < 0 || dxp >= targetW { continue }
                let s = (y * newW + x) * 4, d = (dy * targetW + dxp) * 4
                out[d] = scaled.rgba[s]
                out[d + 1] = scaled.rgba[s + 1]
                out[d + 2] = scaled.rgba[s + 2]
            }
        }
        return Bitmap(width: targetW, height: targetH, rgba: out)
    }

    /// Bilinear resample following cv2.resize(INTER_LINEAR): source coordinates
    /// are taken at pixel centres, (dst + 0.5) * scale - 0.5, with edge clamping.
    static func resizeBilinear(_ bmp: Bitmap, newW: Int, newH: Int) -> Bitmap {
        if newW == bmp.width && newH == bmp.height { return bmp }
        var out = [UInt8](repeating: 255, count: newW * newH * 4)
        let sx = Double(bmp.width) / Double(newW)
        let sy = Double(bmp.height) / Double(newH)
        for y in 0..<newH {
            var fy = (Double(y) + 0.5) * sy - 0.5
            if fy < 0 { fy = 0 }
            let y0 = min(bmp.height - 1, Int(fy))
            let y1 = min(bmp.height - 1, y0 + 1)
            let wy = fy - Double(y0)
            for x in 0..<newW {
                var fx = (Double(x) + 0.5) * sx - 0.5
                if fx < 0 { fx = 0 }
                let x0 = min(bmp.width - 1, Int(fx))
                let x1 = min(bmp.width - 1, x0 + 1)
                let wx = fx - Double(x0)
                let i00 = (y0 * bmp.width + x0) * 4, i01 = (y0 * bmp.width + x1) * 4
                let i10 = (y1 * bmp.width + x0) * 4, i11 = (y1 * bmp.width + x1) * 4
                let d = (y * newW + x) * 4
                for c in 0..<3 {
                    let top = Double(bmp.rgba[i00 + c]) * (1 - wx) + Double(bmp.rgba[i01 + c]) * wx
                    let bot = Double(bmp.rgba[i10 + c]) * (1 - wx) + Double(bmp.rgba[i11 + c]) * wx
                    let v = top * (1 - wy) + bot * wy
                    out[d + c] = UInt8(max(0, min(255, Int(v.rounded()))))
                }
            }
        }
        return Bitmap(width: newW, height: newH, rgba: out)
    }
}
