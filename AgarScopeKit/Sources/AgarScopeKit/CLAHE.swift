import Foundation

/// Colour conversion and CLAHE, reproducing OpenCV's 8-bit behaviour.
///
/// Every calibrated threshold in this project -- circularity 0.75, the marking
/// chroma and darkness cutoffs, the area bounds -- was measured on images that
/// went through cv2.cvtColor and cv2.createCLAHE. Approximating either changes
/// the pixel values those thresholds read, so both are matched to OpenCV's
/// integer arithmetic rather than to the textbook formulas.
enum ColorLAB {
    // OpenCV's sRGB -> XYZ matrix, D65 white point.
    private static let m = [0.412453, 0.357580, 0.180423,
                            0.212671, 0.715160, 0.072169,
                            0.019334, 0.119193, 0.950227]
    private static let xn = 0.950456, yn = 1.0, zn = 1.088754

    private static let gammaLUT: [Double] = (0..<256).map { i in
        let c = Double(i) / 255.0
        return c > 0.04045 ? pow((c + 0.055) / 1.055, 2.4) : c / 12.92
    }

    private static func f(_ t: Double) -> Double {
        t > 0.008856 ? cbrt(t) : (7.787 * t + 16.0 / 116.0)
    }

    /// Returns (L, a, b) in OpenCV's 8-bit encoding: L scaled to 0-255,
    /// a and b offset by +128.
    static func toLAB(r: UInt8, g: UInt8, b: UInt8) -> (Double, Double, Double) {
        let R = gammaLUT[Int(r)], G = gammaLUT[Int(g)], B = gammaLUT[Int(b)]
        let X = (m[0] * R + m[1] * G + m[2] * B) / xn
        let Y = (m[3] * R + m[4] * G + m[5] * B) / yn
        let Z = (m[6] * R + m[7] * G + m[8] * B) / zn
        let fx = f(X), fy = f(Y), fz = f(Z)
        let L = Y > 0.008856 ? (116.0 * fy - 16.0) : (903.3 * Y)
        return (L * 255.0 / 100.0, 500.0 * (fx - fy) + 128.0, 200.0 * (fy - fz) + 128.0)
    }
}

enum CLAHE {
    /// OpenCV's CLAHE with clipLimit 3.0 and an 8x8 tile grid, applied to the
    /// L channel only -- which is what make_clahe() in the server does, leaving
    /// a and b untouched so the marking colour test still sees true chroma.
    static func applyToLuminance(_ l: [UInt8], width: Int, height: Int,
                                 clipLimit: Double = 3.0, tiles: Int = 8) -> [UInt8] {
        // When the image does not divide evenly into the tile grid, OpenCV does
        // NOT shrink the edge tiles -- it grows the image with BORDER_REFLECT_101
        // so every tile is full size, and normalises each LUT by that full size.
        //
        // This port originally truncated instead. On a plate whose height was
        // 1922 the two agreed on 99.8% of pixels when the height was cropped to
        // 1920, and on only 73% when it was not: a whole eighth of the image
        // was being equalised against a different pixel population and a
        // different divisor. The README's "0.11% of pixels" figure had been
        // measured on a divisible size and did not generalise.
        //
        // Note the padding is applied to BOTH axes as soon as EITHER is
        // indivisible -- so an exactly-divisible width still gains a full extra
        // tile of columns. That looks like an oddity of OpenCV's condition
        // rather than a design decision, but it is what the calibrated
        // thresholds were measured against.
        let divisible = width % tiles == 0 && height % tiles == 0
        let padX = divisible ? 0 : tiles - width % tiles
        let padY = divisible ? 0 : tiles - height % tiles
        let extW = width + padX, extH = height + padY
        let tw = extW / tiles, th = extH / tiles
        let tileArea = tw * th

        /// BORDER_REFLECT_101: the edge pixel is not repeated, the reflection
        /// starts one in.
        func srcAt(_ x: Int, _ y: Int) -> UInt8 {
            let sx = x < width ? x : 2 * (width - 1) - x
            let sy = y < height ? y : 2 * (height - 1) - y
            return l[max(0, min(height - 1, sy)) * width + max(0, min(width - 1, sx))]
        }

        // OpenCV scales the limit by the tile's pixel count over 256 bins, then
        // rounds to at least 1 -- a fractional limit would clip nothing.
        var limit = Int(clipLimit * Double(tileArea) / 256.0)
        limit = max(1, limit)

        var luts = [[UInt8]](repeating: [UInt8](repeating: 0, count: 256),
                             count: tiles * tiles)
        for ty in 0..<tiles {
            for tx in 0..<tiles {
                var hist = [Int](repeating: 0, count: 256)
                let x0 = tx * tw, y0 = ty * th
                for y in y0..<(y0 + th) {
                    for x in x0..<(x0 + tw) { hist[Int(srcAt(x, y))] += 1 }
                }
                let n = tileArea
                // Clip, then hand the excess back evenly; the remainder is
                // spread one bin at a time exactly as OpenCV does.
                var excess = 0
                for i in 0..<256 where hist[i] > limit {
                    excess += hist[i] - limit
                    hist[i] = limit
                }
                let inc = excess / 256
                let rest = excess % 256
                for i in 0..<256 { hist[i] += inc }
                if rest > 0 {
                    let step = max(1, 256 / rest)
                    var i = 0, given = 0
                    while i < 256 && given < rest { hist[i] += 1; given += 1; i += step }
                    var j = 0
                    while given < rest && j < 256 { hist[j] += 1; given += 1; j += 1 }
                }
                var cum = 0
                let scale = 255.0 / Double(n)
                for i in 0..<256 {
                    cum += hist[i]
                    // saturate_cast<uchar> is cvRound, which breaks ties to
                    // even, not away from zero.
                    luts[ty * tiles + tx][i] = UInt8(max(0, min(255,
                        Int((Double(cum) * scale).rounded(.toNearestOrEven)))))
                }
            }
        }

        // Bilinear blend between the four surrounding tile LUTs.
        var out = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            let gy = Double(y) / Double(th) - 0.5
            let ty0 = max(0, min(tiles - 1, Int(floor(gy))))
            let ty1 = max(0, min(tiles - 1, ty0 + 1))
            let fy = max(0.0, min(1.0, gy - Double(ty0)))
            for x in 0..<width {
                let gx = Double(x) / Double(tw) - 0.5
                let tx0 = max(0, min(tiles - 1, Int(floor(gx))))
                let tx1 = max(0, min(tiles - 1, tx0 + 1))
                let fx = max(0.0, min(1.0, gx - Double(tx0)))
                let v = Int(l[y * width + x])
                let a = Double(luts[ty0 * tiles + tx0][v])
                let b = Double(luts[ty0 * tiles + tx1][v])
                let c = Double(luts[ty1 * tiles + tx0][v])
                let d = Double(luts[ty1 * tiles + tx1][v])
                let top = a + (b - a) * fx
                let bot = c + (d - c) * fx
                out[y * width + x] = UInt8(max(0, min(255,
                    Int((top + (bot - top) * fy).rounded(.toNearestOrEven)))))
            }
        }
        return out
    }
}

extension ColorLAB {
    private static let invGammaLUT: [Double] = (0..<4096).map { i in
        let c = Double(i) / 4095.0
        return c > 0.0031308 ? 1.055 * pow(c, 1 / 2.4) - 0.055 : 12.92 * c
    }

    /// Inverse of toLAB, so the CLAHE-adjusted L can be written back to RGB.
    static func toRGB(L: Double, a: Double, b: Double) -> (UInt8, UInt8, UInt8) {
        let Ls = L * 100.0 / 255.0
        let fy = (Ls + 16.0) / 116.0
        let fx = fy + (a - 128.0) / 500.0
        let fz = fy - (b - 128.0) / 200.0
        func finv(_ t: Double) -> Double {
            t > 0.206893 ? t * t * t : (t - 16.0 / 116.0) / 7.787
        }
        let X = finv(fx) * 0.950456, Y = finv(fy), Z = finv(fz) * 1.088754
        let R =  3.240479 * X - 1.537150 * Y - 0.498535 * Z
        let G = -0.969256 * X + 1.875992 * Y + 0.041556 * Z
        let B =  0.055648 * X - 0.204043 * Y + 1.057311 * Z
        func enc(_ v: Double) -> UInt8 {
            let c = max(0.0, min(1.0, v))
            let g = c > 0.0031308 ? 1.055 * pow(c, 1 / 2.4) - 0.055 : 12.92 * c
            return UInt8(max(0, min(255, Int((g * 255.0).rounded()))))
        }
        return (enc(R), enc(G), enc(B))
    }
}
