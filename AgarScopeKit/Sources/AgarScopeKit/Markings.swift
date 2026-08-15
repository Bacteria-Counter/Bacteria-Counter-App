import Foundation

/// Pen and printed-label rejection, ported from looks_like_marking /
/// find_markings in fastsam_colony_count.py.
///
/// Detections are judged against the OTHER detections on the same plate, never
/// against fixed thresholds. Fixed cutoffs were tried in Python and were badly
/// wrong: tuned on a lab plate whose colonies sit at chroma ~1 against blue
/// marker at 12.2, they went on to delete 99% of the PCA benchmark's colonies,
/// which are themselves chroma 24 against their own agar. A colony on one
/// medium can be more chromatic than ink on another.
enum Markings {
    /// Chroma and darkness of one detection relative to the plate's agar, at the
    /// 75th percentile -- pen strokes are thin, so a mask straddling one keeps a
    /// majority of ordinary agar pixels and its median stays innocent.
    static func features(_ bmp: Bitmap, mask: [UInt8], width: Int, height: Int,
                        dish: DishDetect.Circle) -> (Double, Double) {
        // Agar reference: the median well inside the dish, so the rim and the
        // bench never skew it.
        var bgL: [Double] = [], bgA: [Double] = [], bgB: [Double] = []
        let inner = dish.r * 0.75
        let step = max(1, min(width, height) / 300)   // sampled; the median is stable
        var y = 0
        while y < height {
            var x = 0
            while x < width {
                let dx = Double(x) - dish.cx, dy = Double(y) - dish.cy
                if dx * dx + dy * dy <= inner * inner {
                    let i = (y * width + x) * 4
                    let (L, A, B) = ColorLAB.toLAB(r: bmp.rgba[i], g: bmp.rgba[i + 1],
                                                   b: bmp.rgba[i + 2])
                    bgL.append(L); bgA.append(A); bgB.append(B)
                }
                x += step
            }
            y += step
        }
        guard !bgL.isEmpty else { return (0, 0) }
        let mL = median(&bgL), mA = median(&bgA), mB = median(&bgB)

        var chroma: [Double] = [], dark: [Double] = []
        for yy in 0..<height {
            for xx in 0..<width where mask[yy * width + xx] != 0 {
                let i = (yy * width + xx) * 4
                let (L, A, B) = ColorLAB.toLAB(r: bmp.rgba[i], g: bmp.rgba[i + 1],
                                               b: bmp.rgba[i + 2])
                chroma.append(((A - mA) * (A - mA) + (B - mB) * (B - mB)).squareRoot())
                dark.append(mL - L)
            }
        }
        guard !chroma.isEmpty else { return (0, 0) }
        return (percentile(&chroma, 75), percentile(&dark, 75))
    }

    /// Indices whose chroma or darkness is an outlier among the plate's own
    /// detections. Median absolute deviation, so a handful of marking
    /// detections cannot drag the centre out to meet themselves.
    static func findOutliers(_ feats: [(Double, Double)], z: Double = 20.0,
                             minDetections: Int = 12) -> Set<Int> {
        // Below this there is no population to be an outlier from, and wrongly
        // dropping a near-empty plate's only colony costs more than keeping a
        // stray ink blob.
        guard feats.count >= minDetections else { return [] }
        var flagged = Set<Int>()
        for col in 0..<2 {
            var v = feats.map { col == 0 ? $0.0 : $0.1 }
            let med = median(&v)
            var dev = v.map { abs($0 - med) }
            let mad = median(&dev)
            if mad <= 1e-6 { continue }
            // 1.4826 scales MAD to a standard-deviation equivalent.
            let cut = med + z * 1.4826 * mad
            for (i, value) in (col == 0 ? feats.map { $0.0 } : feats.map { $0.1 }).enumerated()
            where value > cut { flagged.insert(i) }
        }
        return flagged
    }

    static func median(_ v: inout [Double]) -> Double {
        guard !v.isEmpty else { return 0 }
        v.sort()
        let n = v.count
        // numpy.median averages the middle pair on even counts; matching it
        // keeps the agar reference identical to the Python side.
        return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2
    }

    static func percentile(_ v: inout [Double], _ p: Double) -> Double {
        guard !v.isEmpty else { return 0 }
        v.sort()
        // numpy's default linear interpolation between order statistics.
        let idx = (p / 100.0) * Double(v.count - 1)
        let lo = Int(idx), hi = min(v.count - 1, lo + 1)
        return v[lo] + (v[hi] - v[lo]) * (idx - Double(lo))
    }
}
