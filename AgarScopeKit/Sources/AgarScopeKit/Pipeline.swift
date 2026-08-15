import CoreML
import Foundation

/// The complete sam_tuned / sam_micro count, assembled from the verified parts.
///
/// Mirrors run_sam() in server.py: CLAHE, dish detection, adaptive input size,
/// FastSAM, then the same filter chain and the same escalation rule. Every
/// constant here is quoted from the Python source rather than re-derived, since
/// each was fixed by a measurement recorded there.
enum Pipeline {
    // From server.py's SAM_TUNED_* block.
    static let conf: Float = 0.2
    static let minCircularity = 0.75
    static let dishMarginRatio = 1.0
    static let minAreaFrac = 0.0000001
    static let maxAreaFrac = 0.005
    // From SAM_ESCALATE_*.
    static let escalateImgsz = 4480
    static let escalateMinCount = 8
    static let escalateMaxColonyPct = 1.5
    // Exported Core ML sizes; adaptive_imgsz is rounded to the nearest.
    static let availableSizes = [1280, 1920, 2560, 3200, 4480]
    static let dishReference = 581.0
    static let imgszMin = 1280, imgszMax = 3200

    /// Circularity is carried alongside the mask because _sam_response()
    /// reports the mean of it as the pipeline's confidence -- FastSAM has no
    /// class score to average, so shape agreement is what stands in for one.
    struct Colony {
        var mask: [UInt8]; var area: Double; var cx: Double; var cy: Double
        var circularity: Double
    }

    struct Result {
        var count: Int
        var colonies: [Colony]
        var imgsz: Int
        var escalated: Bool
        var dish: DishDetect.Circle?
    }

    /// CLAHE on the L channel only, leaving a and b untouched -- make_clahe().
    static func clahe(_ bmp: Bitmap) -> Bitmap {
        let n = bmp.width * bmp.height
        var l = [UInt8](repeating: 0, count: n)
        // a and b are quantised to 8 bits, NOT carried as Doubles. cv2.cvtColor
        // returns an 8-bit LAB image, so the server's pipeline has always
        // rounded here before converting back. Keeping full precision sounds
        // harmless but changes the reconstructed RGB by a fraction of a unit,
        // and that was measured to shift a detection's confidence from 0.7722
        // to 0.7340 -- enough to move which candidates survive NMS.
        var a = [UInt8](repeating: 0, count: n)
        var b = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            let (L, A, B) = ColorLAB.toLAB(r: bmp.rgba[i * 4], g: bmp.rgba[i * 4 + 1],
                                           b: bmp.rgba[i * 4 + 2])
            l[i] = UInt8(max(0, min(255, Int(L.rounded()))))
            a[i] = UInt8(max(0, min(255, Int(A.rounded()))))
            b[i] = UInt8(max(0, min(255, Int(B.rounded()))))
        }
        let eq = CLAHE.applyToLuminance(l, width: bmp.width, height: bmp.height)
        var out = bmp
        for i in 0..<n {
            let (r, g, bl) = ColorLAB.toRGB(L: Double(eq[i]), a: Double(a[i]),
                                            b: Double(b[i]))
            out.rgba[i * 4] = r; out.rgba[i * 4 + 1] = g; out.rgba[i * 4 + 2] = bl
        }
        return out
    }

    static func adaptiveImgsz(_ bmp: Bitmap, dish: DishDetect.Circle?) -> Int {
        guard let d = dish, d.r > 0 else { return 1536 }
        let longSide = Double(max(bmp.width, bmp.height))
        let required = (dishReference * longSide / d.r / 32).rounded() * 32
        return max(imgszMin, min(imgszMax, Int(required)))
    }

    static func nearestSize(_ target: Int) -> Int {
        availableSizes.min(by: { abs($0 - target) < abs($1 - target) })!
    }

    /// Circularity, area and rim filters, then the marking outlier test.
    static func filter(masks: [[UInt8]], width: Int, height: Int,
                       original: Bitmap, dish: DishDetect.Circle?) -> [Colony] {
        let imgArea = Double(width * height)
        var kept: [Colony] = []
        for m in masks {
            let (circ, area) = Contours.circularity(m, width: width, height: height)
            if circ < minCircularity { continue }
            let frac = area / imgArea
            if frac < minAreaFrac || frac > maxAreaFrac { continue }
            var sx = 0.0, sy = 0.0, n = 0.0
            for y in 0..<height {
                for x in 0..<width where m[y * width + x] != 0 {
                    sx += Double(x); sy += Double(y); n += 1
                }
            }
            if n == 0 { continue }
            let cx = sx / n, cy = sy / n
            if let d = dish {
                let dist = ((cx - d.cx) * (cx - d.cx) + (cy - d.cy) * (cy - d.cy)).squareRoot()
                if dist > d.r * dishMarginRatio { continue }
            }
            kept.append(Colony(mask: m, area: area, cx: cx, cy: cy, circularity: circ))
        }
        guard !kept.isEmpty, let d = dish else { return kept }

        // Marking rejection, judged against this plate's own detections --
        // fixed thresholds were shown to delete 99% of a benchmark's colonies.
        let feats = kept.map { Markings.features(original, mask: $0.mask,
                                                 width: width, height: height,
                                                 dish: d) }
        let flagged = Markings.findOutliers(feats)
        return kept.enumerated().filter { !flagged.contains($0.offset) }.map { $0.element }
    }

    static func count(image: Bitmap, modelDir: String, micro: Bool) throws -> Result {
        let dish = DishDetect.find(image)
        let target = adaptiveImgsz(image, dish: dish)
        let size = nearestSize(target)
        let processed = clahe(image)

        func run(_ s: Int) throws -> [Colony] {
            let model = try Inference.load(dir: modelDir, name: "fastsam_\(s)")
            let lb = ImageOps.letterbox(processed, targetW: s, targetH: s)
            guard let raw = try Inference.run(model, lb) else { return [] }
            let dets = PostProcess.nms(raw.det, confThreshold: conf)
            let masks = PostProcess.masks(proto: raw.proto, detections: dets,
                                          letterboxW: s, letterboxH: s,
                                          originalW: image.width, originalH: image.height)
            return filter(masks: masks, width: image.width, height: image.height,
                          original: image, dish: dish)
        }

        var colonies = try run(size)
        var escalated = false
        if micro, let d = dish, colonies.count >= escalateMinCount {
            let areas = colonies.map { $0.area }.sorted()
            let median = areas[areas.count / 2]
            let pct = (median / Double.pi).squareRoot() / d.r * 100
            if pct < escalateMaxColonyPct {
                colonies = try run(escalateImgsz)
                escalated = true
            }
        }
        return Result(count: colonies.count, colonies: colonies,
                      imgsz: size, escalated: escalated, dish: dish)
    }
}
