import Accelerate
import CoreML
import Foundation

/// One surviving detection: box in letterboxed coordinates, plus the 32
/// coefficients that weight the prototype planes into this instance's mask.
struct Detection {
    var x1: Float, y1: Float, x2: Float, y2: Float
    var conf: Float
    var coeffs: [Float]
}

enum PostProcess {
    /// Ultralytics' non_max_suppression for the single-class segmentation case.
    ///
    /// The raw tensor is (1, 37, N) laid out channel-major: rows 0-3 are the box
    /// as centre-x, centre-y, width, height; row 4 is the score; rows 5-36 are
    /// the mask coefficients. Written out rather than using Vision's NMS because
    /// the ordering and tie-breaking have to match the Python reference exactly
    /// -- a differently ordered suppression keeps a different member of each
    /// overlapping pair, which changes the mask that survives.
    static func nms(_ det: MLMultiArray, confThreshold: Float = 0.2,
                    iouThreshold: Float = 0.7, maxDet: Int = 3000) -> [Detection] {
        let shape = det.shape.map { $0.intValue }
        guard shape.count == 3, shape[1] == 37 else { return [] }
        let n = shape[2]
        let p = det.dataPointer.assumingMemoryBound(to: Float.self)

        var candidates: [Detection] = []
        candidates.reserveCapacity(1024)
        for i in 0..<n {
            let conf = p[4 * n + i]
            if conf <= confThreshold { continue }
            let cx = p[0 * n + i], cy = p[1 * n + i]
            let w = p[2 * n + i], h = p[3 * n + i]
            var coeffs = [Float](repeating: 0, count: 32)
            for c in 0..<32 { coeffs[c] = p[(5 + c) * n + i] }
            candidates.append(Detection(x1: cx - w / 2, y1: cy - h / 2,
                                        x2: cx + w / 2, y2: cy + h / 2,
                                        conf: conf, coeffs: coeffs))
        }
        if candidates.isEmpty { return [] }
        // torchvision.ops.nms sorts by score descending; ties keep the earlier
        // index, which is what a stable sort on the original order reproduces.
        candidates.sort { $0.conf > $1.conf }
        if candidates.count > 30000 { candidates = Array(candidates.prefix(30000)) }

        var keep: [Detection] = []
        var suppressed = [Bool](repeating: false, count: candidates.count)
        for i in 0..<candidates.count {
            if suppressed[i] { continue }
            let a = candidates[i]
            keep.append(a)
            if keep.count >= maxDet { break }
            let areaA = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
            for j in (i + 1)..<candidates.count {
                if suppressed[j] { continue }
                let b = candidates[j]
                let ix1 = max(a.x1, b.x1), iy1 = max(a.y1, b.y1)
                let ix2 = min(a.x2, b.x2), iy2 = min(a.y2, b.y2)
                let iw = max(0, ix2 - ix1), ih = max(0, iy2 - iy1)
                let inter = iw * ih
                if inter <= 0 { continue }
                let areaB = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
                if inter / (areaA + areaB - inter) > iouThreshold { suppressed[j] = true }
            }
        }
        return keep
    }

    /// Build each detection's binary mask at the ORIGINAL image resolution.
    ///
    /// Mirrors ops.process_mask(upsample=True) followed by scale_masks: weight
    /// the 32 prototype planes by the detection's coefficients, sigmoid, crop to
    /// the detection's own box so neighbouring colonies cannot bleed in, then
    /// resample to the source frame with the letterbox padding removed.
    static func masks(proto: MLMultiArray, detections: [Detection],
                      letterboxW: Int, letterboxH: Int,
                      originalW: Int, originalH: Int) -> [[UInt8]] {
        let ps = proto.shape.map { $0.intValue }
        guard ps.count == 4, ps[1] == 32 else { return [] }
        let mh = ps[2], mw = ps[3]
        let pp = proto.dataPointer.assumingMemoryBound(to: Float.self)
        let plane = mh * mw

        // Letterbox geometry, so padding can be cropped before the final resize.
        let scale = min(Float(letterboxW) / Float(originalW),
                        Float(letterboxH) / Float(originalH))
        let newW = Int((Float(originalW) * scale).rounded())
        let newH = Int((Float(originalH) * scale).rounded())
        let padLeft = Float(letterboxW - newW) / 2
        let padTop = Float(letterboxH - newH) / 2

        var out: [[UInt8]] = []
        out.reserveCapacity(detections.count)
        var acc = [Float](repeating: 0, count: plane)

        for d in detections {
            // Weighted sum of prototype planes -- BLAS keeps this from dominating
            // the runtime when a dense plate yields hundreds of detections.
            vDSP_vclr(&acc, 1, vDSP_Length(plane))
            for c in 0..<32 {
                var w = d.coeffs[c]
                if w == 0 { continue }
                vDSP_vsma(pp + c * plane, 1, &w, acc, 1, &acc, 1, vDSP_Length(plane))
            }

            // Box in prototype-grid coordinates, used to crop.
            let sx = Float(mw) / Float(letterboxW), sy = Float(mh) / Float(letterboxH)
            let bx1 = d.x1 * sx, bx2 = d.x2 * sx
            let by1 = d.y1 * sy, by2 = d.y2 * sy

            // The order below is ultralytics' and must not be collapsed into a
            // single sample. process_mask upsamples the logits to letterbox
            // resolution and BINARISES there; scale_masks then interpolates
            // those 0/1 values again on the way down to the source frame, which
            // widens every edge. Binarising once at the end instead produced
            // masks about 16% smaller -- enough to push detections under the
            // area filter and lose roughly ten colonies per plate.

            /// Binary value at letterbox resolution: proto logit bilinearly
            /// upsampled, thresholded at 0, and cropped to this detection's box.
            func binaryAtLetterbox(_ lx: Float, _ ly: Float) -> Float {
                if lx < d.x1 || lx > d.x2 || ly < d.y1 || ly > d.y2 { return 0 }
                // F.interpolate defaults to align_corners=False, which samples
                // at pixel CENTRES: (dst + 0.5) * ratio - 0.5. Dropping the
                // half-pixel terms offsets every sample by 0.375 of a prototype
                // cell here, and a prototype cell covers 4 letterbox pixels.
                let gx = max(0, (lx + 0.5) * sx - 0.5)
                let gy = max(0, (ly + 0.5) * sy - 0.5)
                let x0 = max(0, min(mw - 1, Int(gx))), x1i = min(mw - 1, x0 + 1)
                let y0 = max(0, min(mh - 1, Int(gy))), y1i = min(mh - 1, y0 + 1)
                let fx = gx - Float(x0), fy = gy - Float(y0)
                let v00 = acc[y0 * mw + x0], v01 = acc[y0 * mw + x1i]
                let v10 = acc[y1i * mw + x0], v11 = acc[y1i * mw + x1i]
                let top = v00 + (v01 - v00) * fx
                let bot = v10 + (v11 - v10) * fx
                return (top + (bot - top) * fy) > 0 ? 1 : 0
            }

            // scale_masks crops the padding away FIRST, then interpolates the
            // cropped region to the source size -- so the mapping is over the
            // cropped extent, again with align_corners=False, and the pad offset
            // is added afterwards.
            let cropW = Float(letterboxW) - 2 * padLeft
            let cropH = Float(letterboxH) - 2 * padTop
            let rx = cropW / Float(originalW), ry = cropH / Float(originalH)

            var mask = [UInt8](repeating: 0, count: originalW * originalH)
            for y in 0..<originalH {
                let ly = padTop + (Float(y) + 0.5) * ry - 0.5
                if ly < d.y1 - 1 || ly > d.y2 + 1 { continue }
                let ly0 = ly.rounded(.down), fy = ly - ly0
                for x in 0..<originalW {
                    let lx = padLeft + (Float(x) + 0.5) * rx - 0.5
                    if lx < d.x1 - 1 || lx > d.x2 + 1 { continue }
                    let lx0 = lx.rounded(.down), fx = lx - lx0
                    // Second bilinear pass, over the binarised values.
                    let b00 = binaryAtLetterbox(lx0, ly0)
                    let b01 = binaryAtLetterbox(lx0 + 1, ly0)
                    let b10 = binaryAtLetterbox(lx0, ly0 + 1)
                    let b11 = binaryAtLetterbox(lx0 + 1, ly0 + 1)
                    let top = b00 + (b01 - b00) * fx
                    let bot = b10 + (b11 - b10) * fx
                    if top + (bot - top) * fy > 0.5 { mask[y * originalW + x] = 1 }
                }
            }
            out.append(mask)
        }
        return out
    }
}
