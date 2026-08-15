import Foundation

/// Petri-dish boundary detection, reproducing find_dish_circle().
///
/// The Python version runs cv2.HoughCircles with HOUGH_GRADIENT on a 1200px
/// downscale after a 9x9 blur. The radius it returns feeds two things that
/// matter: adaptive_imgsz, which is 581 * longSide / radius, and the rim filter
/// that discards detections beyond the dish edge. Both tolerate a percent or so
/// of error -- imgsz is rounded to the nearest exported size anyway -- but not a
/// wrong circle, so the accumulator below follows the same gradient-voting
/// scheme rather than substituting a different circle fit.
enum DishDetect {
    struct Circle { var cx: Double; var cy: Double; var r: Double }

    static func find(_ bmp: Bitmap) -> Circle? {
        let scale = 1200.0 / Double(max(bmp.width, bmp.height))
        let w = Int(Double(bmp.width) * scale)
        let h = Int(Double(bmp.height) * scale)
        guard w > 16, h > 16 else { return nil }

        // Grey, downscaled by area averaging to match cv2.resize's default.
        var grey = [Double](repeating: 0, count: w * h)
        let sx = Double(bmp.width) / Double(w), sy = Double(bmp.height) / Double(h)
        for y in 0..<h {
            let y0 = Int(Double(y) * sy), y1 = min(bmp.height, Int(Double(y + 1) * sy))
            for x in 0..<w {
                let x0 = Int(Double(x) * sx), x1 = min(bmp.width, Int(Double(x + 1) * sx))
                var sum = 0.0, n = 0
                for yy in y0..<max(y0 + 1, y1) {
                    for xx in x0..<max(x0 + 1, x1) {
                        let i = (yy * bmp.width + xx) * 4
                        // OpenCV's BGR->GRAY weights.
                        sum += 0.114 * Double(bmp.rgba[i + 2]) + 0.587 * Double(bmp.rgba[i + 1])
                             + 0.299 * Double(bmp.rgba[i])
                        n += 1
                    }
                }
                grey[y * w + x] = sum / Double(max(1, n))
            }
        }
        grey = gaussianBlur(grey, w: w, h: h, sigma: 2.0, radius: 4)

        // Sobel gradients; edge strength gates which pixels get to vote.
        var gx = [Double](repeating: 0, count: w * h)
        var gy = [Double](repeating: 0, count: w * h)
        var mag = [Double](repeating: 0, count: w * h)
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                let tl = grey[i - w - 1], tc = grey[i - w], tr = grey[i - w + 1]
                let ml = grey[i - 1],                      mr = grey[i + 1]
                let bl = grey[i + w - 1], bc = grey[i + w], br = grey[i + w + 1]
                let dx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)
                let dy = (bl + 2 * bc + br) - (tl + 2 * tc + tr)
                gx[i] = dx; gy[i] = dy
                mag[i] = (dx * dx + dy * dy).squareRoot()
            }
        }
        // param1=60 is Canny's high threshold in OpenCV; edges above it vote.
        let voteThreshold = 60.0
        let minR = Int(Double(w) * 0.25), maxR = Int(Double(w) * 0.5)
        guard maxR > minR else { return nil }

        // Vote for centres along each edge's gradient line, both directions.
        var acc = [Int](repeating: 0, count: w * h)
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                let m = mag[i]
                if m < voteThreshold { continue }
                let ux = gx[i] / m, uy = gy[i] / m
                for sign in [-1.0, 1.0] {
                    var r = Double(minR)
                    while r <= Double(maxR) {
                        let cx = Int((Double(x) + sign * ux * r).rounded())
                        let cy = Int((Double(y) + sign * uy * r).rounded())
                        if cx >= 0 && cy >= 0 && cx < w && cy < h { acc[cy * w + cx] += 1 }
                        r += 1
                    }
                }
            }
        }

        var bestIdx = 0, bestVal = 0
        for i in 0..<(w * h) where acc[i] > bestVal { bestVal = acc[i]; bestIdx = i }
        if bestVal == 0 { return nil }
        let ccx = Double(bestIdx % w), ccy = Double(bestIdx / w)

        // Radius: the distance at which the most edge pixels sit from that centre.
        var hist = [Int](repeating: 0, count: maxR + 2)
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                if mag[i] < voteThreshold { continue }
                let d = ((Double(x) - ccx) * (Double(x) - ccx)
                         + (Double(y) - ccy) * (Double(y) - ccy)).squareRoot()
                let ri = Int(d.rounded())
                if ri >= minR && ri <= maxR { hist[ri] += 1 }
            }
        }
        // Smooth, then divide by radius before picking the peak. A raw count
        // favours large circles for a purely geometric reason -- their
        // circumference is longer, so more stray edge pixels land at that
        // distance by chance. On one test plate that bias chose 877 where
        // OpenCV chose 794, a 10% error that shifted adaptive_imgsz by a whole
        // step. Support per unit of circumference is what actually indicates a
        // real edge.
        var smooth = [Double](repeating: 0, count: maxR + 2)
        for r in minR...maxR {
            var s = 0.0
            for k in -2...2 {
                let rr = r + k
                if rr >= 0 && rr <= maxR { s += Double(hist[rr]) }
            }
            smooth[r] = s / Double(r)
        }
        var bestR = minR
        var bestScore = 0.0
        for r in minR...maxR where smooth[r] > bestScore { bestScore = smooth[r]; bestR = r }
        if bestScore == 0 { return nil }

        return Circle(cx: ccx / scale, cy: ccy / scale, r: Double(bestR) / scale)
    }

    static func gaussianBlur(_ src: [Double], w: Int, h: Int,
                             sigma: Double, radius: Int) -> [Double] {
        var kernel = [Double](repeating: 0, count: 2 * radius + 1)
        var sum = 0.0
        for i in -radius...radius {
            let v = exp(-Double(i * i) / (2 * sigma * sigma))
            kernel[i + radius] = v; sum += v
        }
        for i in 0..<kernel.count { kernel[i] /= sum }

        var tmp = [Double](repeating: 0, count: w * h)
        var out = [Double](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                var acc = 0.0
                for k in -radius...radius {
                    let xx = max(0, min(w - 1, x + k))
                    acc += src[y * w + xx] * kernel[k + radius]
                }
                tmp[y * w + x] = acc
            }
        }
        for y in 0..<h {
            for x in 0..<w {
                var acc = 0.0
                for k in -radius...radius {
                    let yy = max(0, min(h - 1, y + k))
                    acc += tmp[yy * w + x] * kernel[k + radius]
                }
                out[y * w + x] = acc
            }
        }
        return out
    }
}
