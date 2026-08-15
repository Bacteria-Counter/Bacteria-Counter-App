import CoreML
import Foundation

/// CSRNet density-map counting, from run_csrnet() in server.py.
///
/// A different paradigm from everything else here: no boxes, no NMS, no masks.
/// The model predicts a density heatmap and the count is its sum, so there is
/// no per-colony position to report -- and the server's notes are explicit that
/// inventing one from density peaks was tried and looked visibly wrong, because
/// a 3120x4160 photo becomes a 96x96 map where one cell spans 32x43 real
/// pixels. This port keeps that limit rather than quietly papering over it.
enum CSRNet {
    static let workingSize = 768
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    /// cv2.resize(..., INTER_AREA), which is NOT box-filter-then-sample: each
    /// destination pixel averages the source pixels its footprint covers, with
    /// fractional weights at both ends. Bilinear here would alias a 4x
    /// downscale badly, and the count is a sum over exactly these pixels.
    static func resizeArea(_ bmp: Bitmap, newW: Int, newH: Int) -> Bitmap {
        /// One axis of OpenCV's computeResizeAreaTab: for each destination
        /// index, the source indices it draws from and their weights.
        func tab(_ srcN: Int, _ dstN: Int) -> [[(Int, Double)]] {
            let scale = Double(srcN) / Double(dstN)
            var out = [[(Int, Double)]](repeating: [], count: dstN)
            for d in 0..<dstN {
                let f1 = Double(d) * scale
                let f2 = f1 + scale
                let cell = min(scale, Double(srcN) - f1)
                var s1 = Int(f1.rounded(.up))
                var s2 = Int(f2.rounded(.down))
                s2 = min(s2, srcN - 1)
                s1 = min(s1, s2)
                if Double(s1) - f1 > 1e-3 {
                    out[d].append((s1 - 1, (Double(s1) - f1) / cell))
                }
                var s = s1
                while s < s2 { out[d].append((s, 1.0 / cell)); s += 1 }
                if f2 - Double(s2) > 1e-3 {
                    out[d].append((s2, min(min(f2 - Double(s2), 1.0), cell) / cell))
                }
            }
            return out
        }

        let tx = tab(bmp.width, newW), ty = tab(bmp.height, newH)
        var out = [UInt8](repeating: 255, count: newW * newH * 4)
        // Horizontal first into a float buffer, then vertical -- the same order
        // and the same float accumulation OpenCV uses, so the single rounding
        // happens once at the end.
        var rows = [Float](repeating: 0, count: bmp.height * newW * 3)
        for y in 0..<bmp.height {
            for x in 0..<newW {
                var acc: (Double, Double, Double) = (0, 0, 0)
                for (s, w) in tx[x] {
                    let i = (y * bmp.width + s) * 4
                    acc.0 += Double(bmp.rgba[i]) * w
                    acc.1 += Double(bmp.rgba[i + 1]) * w
                    acc.2 += Double(bmp.rgba[i + 2]) * w
                }
                let o = (y * newW + x) * 3
                rows[o] = Float(acc.0); rows[o + 1] = Float(acc.1); rows[o + 2] = Float(acc.2)
            }
        }
        for y in 0..<newH {
            for x in 0..<newW {
                var acc: (Double, Double, Double) = (0, 0, 0)
                for (s, w) in ty[y] {
                    let o = (s * newW + x) * 3
                    acc.0 += Double(rows[o]) * w
                    acc.1 += Double(rows[o + 1]) * w
                    acc.2 += Double(rows[o + 2]) * w
                }
                let d = (y * newW + x) * 4
                // saturate_cast<uchar>, which rounds half to even.
                out[d] = UInt8(max(0, min(255, Int(acc.0.rounded(.toNearestOrEven)))))
                out[d + 1] = UInt8(max(0, min(255, Int(acc.1.rounded(.toNearestOrEven)))))
                out[d + 2] = UInt8(max(0, min(255, Int(acc.2.rounded(.toNearestOrEven)))))
            }
        }
        return Bitmap(width: newW, height: newH, rgba: out)
    }

    struct Result {
        /// The raw density map, 1/8 of the working size on each axis.
        var density: [Float]
        var mapW: Int, mapH: Int
        /// run_csrnet() rounds the summed density to report a colony count.
        ///
        /// Clamped at zero, which the Python is not. Nothing constrains the
        /// predicted density to be positive, and on an empty plate the sum
        /// lands just either side of zero -- the float32 weights give -0.4,
        /// which rounds to 0 and hides the issue, while the quantised weights
        /// give about -0.6, which rounds to -1. A count of -1 colonies is not
        /// a wrong answer, it is not an answer; run_csrnet() would have shown
        /// it too and simply never met the input that produced it.
        var count: Int { max(0, Int(density.reduce(0, +).rounded())) }
    }

    /// OpenCV's COLORMAP_JET, as a 256-entry RGB table.
    ///
    /// Generated from the same piecewise-linear anchors OpenCV builds it from,
    /// rather than eyeballed: the heatmap is the only thing CSRNet gives a user
    /// to look at, and a different colour ramp would make the same prediction
    /// look like a different one next to a screenshot from the server.
    static let jetLUT: [(UInt8, UInt8, UInt8)] = {
        // OpenCV's colormap_jet control points, in BGR order in its source;
        // written here in the R, G, B order this pipeline uses.
        let r: [Double] = [0, 0, 0, 0, 0.5, 1, 1, 1, 0.5]
        let g: [Double] = [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0]
        let b: [Double] = [0.5, 1, 1, 1, 0.5, 0, 0, 0, 0]
        func sample(_ xs: [Double], _ t: Double) -> Double {
            let pos = t * Double(xs.count - 1)
            let i = min(xs.count - 2, max(0, Int(pos)))
            let f = pos - Double(i)
            return xs[i] + (xs[i + 1] - xs[i]) * f
        }
        return (0..<256).map { i in
            let t = Double(i) / 255.0
            return (UInt8(max(0, min(255, (sample(r, t) * 255).rounded()))),
                    UInt8(max(0, min(255, (sample(g, t) * 255).rounded()))),
                    UInt8(max(0, min(255, (sample(b, t) * 255).rounded()))))
        }
    }()

    /// run_csrnet()'s overlay: normalise the density map, colour it, stretch it
    /// back to the photo's size and blend 55/45. PNG rather than the server's
    /// JPEG -- this one is handed straight to an image view instead of crossing
    /// an HTTP boundary, so there is no reason to lose anything to compression.
    static func heatmapPNG(density r: Result, over bmp: Bitmap) -> Data? {
        guard !r.density.isEmpty, let lo = r.density.min(), let hi = r.density.max()
        else { return nil }
        let span = hi > lo ? Double(hi - lo) : 1
        var out = [UInt8](repeating: 255, count: bmp.width * bmp.height * 4)
        for y in 0..<bmp.height {
            // Nearest-neighbour up from a 96x96 map would band visibly; bilinear
            // matches cv2.resize(INTER_LINEAR), which is what the server used.
            let sy = (Double(y) + 0.5) * Double(r.mapH) / Double(bmp.height) - 0.5
            let y0 = max(0, min(r.mapH - 1, Int(sy.rounded(.down))))
            let y1 = min(r.mapH - 1, y0 + 1), fy = max(0, sy - Double(y0))
            for x in 0..<bmp.width {
                let sx = (Double(x) + 0.5) * Double(r.mapW) / Double(bmp.width) - 0.5
                let x0 = max(0, min(r.mapW - 1, Int(sx.rounded(.down))))
                let x1 = min(r.mapW - 1, x0 + 1), fx = max(0, sx - Double(x0))
                func v(_ xi: Int, _ yi: Int) -> Double {
                    (Double(r.density[yi * r.mapW + xi]) - Double(lo)) / span * 255
                }
                let top = v(x0, y0) + (v(x1, y0) - v(x0, y0)) * fx
                let bot = v(x0, y1) + (v(x1, y1) - v(x0, y1)) * fx
                let heat = jetLUT[max(0, min(255, Int((top + (bot - top) * fy).rounded())))]
                let i = (y * bmp.width + x) * 4
                out[i]     = UInt8(min(255, Int(Double(bmp.rgba[i]) * 0.55 + Double(heat.0) * 0.45)))
                out[i + 1] = UInt8(min(255, Int(Double(bmp.rgba[i+1]) * 0.55 + Double(heat.1) * 0.45)))
                out[i + 2] = UInt8(min(255, Int(Double(bmp.rgba[i+2]) * 0.55 + Double(heat.2) * 0.45)))
            }
        }
        guard let cg = ImageOps.cgImage(Bitmap(width: bmp.width, height: bmp.height, rgba: out))
        else { return nil }
        return ImageOps.pngData(cg)
    }

    static func count(image: Bitmap, modelDir: String) throws -> Result {
        // Square resize, aspect deliberately not preserved -- the checkpoint was
        // trained this way, and its measured numbers (MAE 6.26 on AGAR, 9.25 on
        // the PCA holdout) are numbers for the distorted input.
        let small = resizeArea(image, newW: workingSize, newH: workingSize)

        // ToTensor then Normalize: channel-first, /255, ImageNet statistics.
        // The export takes a raw tensor rather than an image input, so none of
        // this is folded into the model.
        let n = workingSize * workingSize
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: workingSize),
                                           NSNumber(value: workingSize)], dataType: .float32)
        let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<3 {
            for i in 0..<n {
                p[c * n + i] = (Float(small.rgba[i * 4 + c]) / 255 - mean[c]) / std[c]
            }
        }

        let model = try Inference.load(dir: modelDir, name: "csrnet")
        let outs = try Inference.outputs(model, tensor: arr)
        guard let d = outs.first(where: { $0.shape.count == 4 }) else {
            return Result(density: [], mapW: 0, mapH: 0)
        }
        let s = d.shape.map { $0.intValue }
        let h = s[2], w = s[3]
        let dp = d.dataPointer.assumingMemoryBound(to: Float.self)
        return Result(density: Array(UnsafeBufferPointer(start: dp, count: w * h)),
                      mapW: w, mapH: h)
    }
}
