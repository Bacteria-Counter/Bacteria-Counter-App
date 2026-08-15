import Accelerate
import Foundation

/// The three preprocessing transforms baked into the fine-tuned YOLO
/// checkpoints, reproduced from server.py's make_dog_blend / make_clahe /
/// make_lab_ab.
///
/// These are not cosmetic. Each of dog_blend, clahe and lab_ab was TRAINED on
/// its own transform, so feeding a differently-preprocessed image gives the
/// model a distribution it never saw -- the train/inference mismatch the server
/// comments call out. That makes these functions part of the model, and they
/// are matched to OpenCV's arithmetic for the same reason CLAHE and the LAB
/// conversion were.
enum Preprocess {
    /// cv2.cvtColor(BGR2GRAY) on 8-bit input: OpenCV uses fixed-point weights
    /// at 14 fractional bits, not the floating-point 0.299/0.587/0.114. The
    /// difference is under a unit per pixel, but this feeds a difference of
    /// Gaussians where the two blurs nearly cancel, so the residual is exactly
    /// where a rounding difference has room to show.
    static func grey(_ bmp: Bitmap) -> [Float] {
        let n = bmp.width * bmp.height
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let r = Int(bmp.rgba[i * 4]), g = Int(bmp.rgba[i * 4 + 1]), b = Int(bmp.rgba[i * 4 + 2])
            out[i] = Float((r * 4899 + g * 9617 + b * 1868 + (1 << 13)) >> 14)
        }
        return out
    }

    /// cv2.getGaussianKernel(n, sigma) -- the analytic form, which is what
    /// OpenCV uses whenever sigma is given explicitly (its hard-coded small
    /// kernel table only applies when sigma is left to be derived).
    static func gaussianKernel(sigma: Double) -> [Float] {
        // ksize from sigma, as GaussianBlur does for ksize=(0,0). The multiplier
        // is 4 for float input and 3 for 8-bit; make_dog_blend blurs a float32
        // image, so 4 it is -- 17 taps at sigma 2, 97 at sigma 12.
        var n = Int((sigma * 4 * 2 + 1).rounded())
        n |= 1
        let scale = -0.5 / (sigma * sigma)
        var k = [Double](repeating: 0, count: n)
        var sum = 0.0
        for i in 0..<n {
            let x = Double(i) - Double(n - 1) * 0.5
            let v = exp(scale * x * x)
            k[i] = v; sum += v
        }
        return k.map { Float($0 / sum) }
    }

    /// Separable Gaussian with BORDER_REFLECT_101, OpenCV's default: the edge
    /// pixel is not repeated, the reflection starts one in (abcd -> cba|abcd|dcb).
    static func gaussianBlur(_ src: [Float], w: Int, h: Int, sigma: Double) -> [Float] {
        let k = gaussianKernel(sigma: sigma)
        let r = k.count / 2
        func reflect(_ i: Int, _ n: Int) -> Int {
            if n == 1 { return 0 }
            var v = i
            while v < 0 || v >= n {
                if v < 0 { v = -v }
                if v >= n { v = 2 * (n - 1) - v }
            }
            return v
        }

        var tmp = [Float](repeating: 0, count: w * h)
        src.withUnsafeBufferPointer { s in
            tmp.withUnsafeMutableBufferPointer { t in
                for y in 0..<h {
                    let row = s.baseAddress! + y * w
                    let dst = t.baseAddress! + y * w
                    for x in 0..<w {
                        var acc: Float = 0
                        for j in 0..<k.count { acc += row[reflect(x + j - r, w)] * k[j] }
                        dst[x] = acc
                    }
                }
            }
        }

        // Vertical pass row-wise so each tap is one contiguous vector op --
        // a column-major gather here costs several seconds on a 3200px plate.
        var out = [Float](repeating: 0, count: w * h)
        tmp.withUnsafeBufferPointer { t in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0..<h {
                    let dst = o.baseAddress! + y * w
                    for j in 0..<k.count {
                        var wgt = k[j]
                        let sy = reflect(y + j - r, h)
                        vDSP_vsma(t.baseAddress! + sy * w, 1, &wgt, dst, 1, dst, 1, vDSP_Length(w))
                    }
                }
            }
        }
        return out
    }

    /// make_dog_blend: add a difference of Gaussians of the luminance back onto
    /// every colour channel, sharpening colony edges without changing hue.
    static func dogBlend(_ bmp: Bitmap, sigma1: Double = 2, sigma2: Double = 12,
                         strength: Float = 1.0) -> Bitmap {
        let g = grey(bmp)
        let g1 = gaussianBlur(g, w: bmp.width, h: bmp.height, sigma: sigma1)
        let g2 = gaussianBlur(g, w: bmp.width, h: bmp.height, sigma: sigma2)
        var out = bmp
        for i in 0..<(bmp.width * bmp.height) {
            let d = strength * (g1[i] - g2[i])
            for c in 0..<3 {
                // np.clip then astype(uint8) TRUNCATES; it does not round. Using
                // rounded() here would bias every pixel up by half a unit.
                let v = Float(bmp.rgba[i * 4 + c]) + d
                out.rgba[i * 4 + c] = UInt8(max(0, min(255, Int(v))))
            }
        }
        return out
    }

    /// cv2.normalize(src, 0, 255, NORM_MINMAX) on an 8-bit plane.
    static func normalizeMinMax(_ v: [UInt8]) -> [UInt8] {
        guard let lo = v.min(), let hi = v.max(), hi > lo else {
            // OpenCV divides by (max-min); a flat plane would divide by zero, so
            // it is left as-is rather than blown up to an arbitrary constant.
            return v
        }
        let scale = 255.0 / Double(Int(hi) - Int(lo))
        let shift = -Double(lo) * scale
        return v.map { s in
            // saturate_cast<uchar> rounds half to even, not half away from zero.
            UInt8(max(0, min(255, Int((Double(s) * scale + shift).rounded(.toNearestOrEven)))))
        }
    }

    /// make_lab_ab: throw the luminance away entirely and keep only contrast-
    /// stretched chroma. The server's own notes record what this costs -- pale,
    /// low-colour-contrast colonies can vanish -- but the transform must still
    /// be reproduced exactly, because lab_ab was trained on it.
    static func labAB(_ bmp: Bitmap) -> Bitmap {
        let n = bmp.width * bmp.height
        var a = [UInt8](repeating: 0, count: n)
        var b = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            let (_, A, B) = ColorLAB.toLAB(r: bmp.rgba[i * 4], g: bmp.rgba[i * 4 + 1],
                                           b: bmp.rgba[i * 4 + 2])
            a[i] = UInt8(max(0, min(255, Int(A.rounded()))))
            b[i] = UInt8(max(0, min(255, Int(B.rounded()))))
        }
        let an = normalizeMinMax(a), bn = normalizeMinMax(b)
        var out = bmp
        for i in 0..<n {
            // cv2.merge([a, b, 128]) builds a BGR image: channel 0 is blue.
            // Ultralytics flips it to RGB before the model sees it, so in this
            // pipeline's RGB buffer that lands as R=128, G=b, B=a. Getting the
            // order wrong here would swap two channels of the model's input and
            // still produce plausible-looking counts.
            out.rgba[i * 4] = 128
            out.rgba[i * 4 + 1] = bn[i]
            out.rgba[i * 4 + 2] = an[i]
        }
        return out
    }
}
