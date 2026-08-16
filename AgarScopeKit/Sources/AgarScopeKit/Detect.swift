import CoreML
import Foundation

/// One YOLO box, in the coordinates of the ORIGINAL photo.
struct Box {
    var x1: Double, y1: Double, x2: Double, y2: Double
    var conf: Double
    var cx: Double { (x1 + x2) / 2 }
    var cy: Double { (y1 + y2) / 2 }
    /// run_yolo() reports max(w, h) / 2, not the inscribed radius -- the app
    /// draws a circle that covers the box rather than one that fits inside it.
    var radius: Double { max(x2 - x1, y2 - y1) / 2 }
}

/// The YOLO detection pipeline: the seven box-detector options the server
/// exposes, all sharing one architecture and differing only in checkpoint and
/// in the preprocessing that was baked into their training.
enum Detect {
    /// server.py's YOLO_CONF / max_det. iou is ultralytics' own default, which
    /// the server never overrides.
    static let conf: Float = 0.4
    static let iou: Float = 0.7
    static let maxDet = 1000
    /// The YOLO checkpoints were exported at four sizes; 4480 exists only for
    /// FastSAM's sam_micro escalation. adaptive_imgsz already clamps to 3200 so
    /// this list can never be overshot, but stating it separately keeps that
    /// from being an accident of the clamp.
    static let sizes = [1280, 1920, 2560, 3200]

    static func nearestSize(_ target: Int) -> Int {
        sizes.min(by: { abs($0 - target) < abs($1 - target) })!
    }

    /// Which transform each checkpoint expects, from YOLO_PREPROCESS.
    static func preprocess(_ key: String, _ bmp: Bitmap) -> Bitmap {
        switch key {
        case "dog_blend": return Preprocess.dogBlend(bmp)
        case "clahe": return Pipeline.clahe(bmp)
        case "lab_ab": return Preprocess.labAB(bmp)
        default: return bmp
        }
    }

    /// Ultralytics' non_max_suppression for a single-class detect head.
    ///
    /// The raw tensor is (1, 5, N) channel-major: rows 0-3 are the box as
    /// centre-x, centre-y, width, height in letterbox pixels, row 4 is the
    /// score. Deliberately a separate function from PostProcess.nms rather than
    /// a generalisation of it -- that one is verified against torchvision on
    /// identical boxes and is not worth disturbing to save fifteen lines.
    static func nms(_ det: MLMultiArray, confThreshold: Float = conf,
                    iouThreshold: Float = iou, maxDet: Int = maxDet) -> [(Float, Float, Float, Float, Float)] {
        let shape = det.shape.map { $0.intValue }
        guard shape.count == 3, shape[1] == 5 else { return [] }
        let n = shape[2]
        let p = det.dataPointer.assumingMemoryBound(to: Float.self)

        var cand: [(Float, Float, Float, Float, Float)] = []
        for i in 0..<n {
            let c = p[4 * n + i]
            if c <= confThreshold { continue }
            let cx = p[0 * n + i], cy = p[1 * n + i], w = p[2 * n + i], h = p[3 * n + i]
            cand.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, c))
        }
        if cand.isEmpty { return [] }
        cand.sort { $0.4 > $1.4 }
        // max_nms, ultralytics' cap on what reaches torchvision at all.
        if cand.count > 30000 { cand = Array(cand.prefix(30000)) }

        var keep: [(Float, Float, Float, Float, Float)] = []
        var dead = [Bool](repeating: false, count: cand.count)
        for i in 0..<cand.count {
            if dead[i] { continue }
            let a = cand[i]
            keep.append(a)
            if keep.count >= maxDet { break }
            let areaA = max(0, a.2 - a.0) * max(0, a.3 - a.1)
            for j in (i + 1)..<cand.count {
                if dead[j] { continue }
                let b = cand[j]
                let iw = max(0, min(a.2, b.2) - max(a.0, b.0))
                let ih = max(0, min(a.3, b.3) - max(a.1, b.1))
                let inter = iw * ih
                if inter <= 0 { continue }
                let areaB = max(0, b.2 - b.0) * max(0, b.3 - b.1)
                if inter / (areaA + areaB - inter) > iouThreshold { dead[j] = true }
            }
        }
        return keep
    }

    /// ops.scale_boxes: undo the letterbox, then clip to the frame.
    static func scaleBoxes(_ raw: [(Float, Float, Float, Float, Float)],
                           letterbox: Int, originalW: Int, originalH: Int) -> [Box] {
        let gain = min(Double(letterbox) / Double(originalH),
                       Double(letterbox) / Double(originalW))
        // Rounded exactly as ultralytics does, including the -0.1 nudge, so the
        // offset removed here is the one LetterBox actually added.
        let padX = ((Double(letterbox) - (Double(originalW) * gain).rounded()) / 2 - 0.1).rounded()
        let padY = ((Double(letterbox) - (Double(originalH) * gain).rounded()) / 2 - 0.1).rounded()
        return raw.map { b in
            Box(x1: max(0, min(Double(originalW), (Double(b.0) - padX) / gain)),
                y1: max(0, min(Double(originalH), (Double(b.1) - padY) / gain)),
                x2: max(0, min(Double(originalW), (Double(b.2) - padX) / gain)),
                y2: max(0, min(Double(originalH), (Double(b.3) - padY) / gain)),
                conf: Double(b.4))
        }
    }

    struct Result {
        var boxes: [Box]
        var imgsz: Int
        var dish: DishDetect.Circle?
        var count: Int { boxes.count }
        /// run_yolo() reports the mean box score as a percentage.
        var confidence: Double {
            boxes.isEmpty ? 0 : boxes.reduce(0) { $0 + $1.conf } / Double(boxes.count) * 100
        }
    }

    /// `forcedSize` defaults to the adaptive rule, so no shipping caller
    /// changes; it exists so the cross-pipeline comparison can hold input size
    /// fixed and ask what the checkpoint alone contributes.
    static func count(image: Bitmap, modelDir: String, key: String,
                      forcedSize: Int? = nil) throws -> Result {
        // adaptive_imgsz() measures the dish on the ORIGINAL frame, before any
        // preprocessing -- make_lab_ab in particular discards luminance, and the
        // dish edge is a luminance edge.
        let dish = DishDetect.find(image)
        let size = forcedSize ?? nearestSize(Pipeline.adaptiveImgsz(image, dish: dish))
        let src = preprocess(key, image)
        let lb = ImageOps.letterbox(src, targetW: size, targetH: size)

        let model = try Inference.load(dir: modelDir, name: "\(key)_\(size)")
        let outs = try Inference.outputs(model, image: lb)
        // (1, 5, N): 4 box + 1 class score. Identified by shape, as elsewhere.
        guard let det = outs.first(where: { $0.shape.count == 3 && $0.shape[1].intValue == 5 })
        else { return Result(boxes: [], imgsz: size, dish: dish) }

        let boxes = scaleBoxes(nms(det), letterbox: size,
                               originalW: image.width, originalH: image.height)
        return Result(boxes: boxes, imgsz: size, dish: dish)
    }
}
