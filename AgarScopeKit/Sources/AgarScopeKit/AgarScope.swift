import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The public entry point: one call that replaces the local Python server.
///
/// Shaped to match what `POST /analyze` used to return, field for field, so the
/// app's view layer does not have to be rewritten alongside the transport. The
/// server's own response is the specification here -- if a field looked odd
/// there it is reproduced odd, because the UI was built against it and the
/// stored results a user is comparing against came out of it.
public enum AgarScope {

    /// Which pipeline to run. Nine of the server's twelve options are gone.
    /// Two could not run on-device:
    ///
    /// - `gsam2` never converted. 1.1 GB, a separate dependency tree, and it
    ///   failed the empty-plate check outright at 36/36 photos hallucinating
    ///   colonies out of paper texture.
    /// - `sam` -- the ORIGINAL frozen FastSAM settings -- would need a Core ML
    ///   export at 3840 that was never made, plus the square-cell filter. Its
    ///   only purpose was reproducing results from before August 2026, and it
    ///   is by far the least accurate option (MAE 35.11 against sam_tuned's
    ///   3.86). Dropped on the user's decision; Server/server.py still exists
    ///   if an old number ever has to be reproduced exactly.
    ///
    /// The other seven were cut as redundant after measuring all of them on
    /// 126 bright plates, 34 empty plates and the lab photos: `mac1` beat or
    /// matched the six other YOLO variants nearly everywhere, and `sam_micro`
    /// is `sam_tuned` plus an automatic high-resolution pass, which costs
    /// +0.14 MAE on bright plates and turns a 4x undercount into a close
    /// match on a pinpoint plate.
    /// Note this is the APP's list, and it is shorter than what the engine can
    /// run. `AgarScopeCLI` still accepts all seven YOLO checkpoints and
    /// `sam_tuned`, because the benchmark harnesses have to be able to measure
    /// the models that were cut in order to justify cutting them. Shipping a
    /// picker entry and being able to measure something are different needs.
    public enum Model: String, CaseIterable, Sendable {
        case mac1
        case samMicro = "sam_micro"
        case csrnet

        var isSAM: Bool { self == .samMicro }
    }

    /// One colony, in the ORIGINAL photo's pixel space. The overlay view scales
    /// these; nothing here knows how the image is displayed.
    public struct Detection: Sendable {
        public let cx: Double, cy: Double, radius: Double
    }

    public struct Output: Sendable {
        public let totalColonies: Int
        public let averageConfidence: Double
        public let modelUsed: String
        public let detections: [Detection]
        public let imageWidth: Double
        public let imageHeight: Double
        public let countability: CFU.Countability
        /// CSRNet only: a density heatmap blended over the photo, PNG-encoded.
        /// Empty for every other model, and the emptiness is meaningful --
        /// CSRNet produces no per-colony position, so there are no circles to
        /// draw and this is shown instead. See the note in CSRNet.swift.
        public let heatmapPNG: Data?
        /// Which input size the adaptive rule chose, for the status bar.
        public let imgsz: Int
        /// Whether sam_micro's high-resolution second pass fired.
        public let escalated: Bool
    }

    public enum AnalyzeError: Error, LocalizedError {
        case modelsNotFound(String)
        case imageUnreadable

        public var errorDescription: String? {
            switch self {
            case .modelsNotFound(let path):
                return "Berkas model tidak ditemukan di \(path)."
            case .imageUnreadable:
                return "Foto tidak bisa dibaca."
            }
        }
    }

    /// Where the .mlpackage files live. Bundled resources first, so a shipped
    /// app needs no setup; the environment variable is for the CLI and for
    /// running against the quantised directory without rebuilding.
    public static func defaultModelDirectory(bundle: Bundle = .main) -> URL? {
        if let override = ProcessInfo.processInfo.environment["AGARSCOPE_MODELS"] {
            return URL(fileURLWithPath: override)
        }
        if let url = bundle.url(forResource: "coreml_models", withExtension: nil) {
            return url
        }
        return bundle.resourceURL
    }

    public static func analyze(imageAt path: String, model: Model,
                               modelDirectory: URL) throws -> Output {
        guard let bmp = ImageOps.load(path) else { throw AnalyzeError.imageUnreadable }
        return try analyze(bitmap: bmp, model: model, modelDirectory: modelDirectory)
    }

    public static func analyze(image: CGImage, model: Model,
                               modelDirectory: URL) throws -> Output {
        guard let bmp = ImageOps.from(image) else { throw AnalyzeError.imageUnreadable }
        return try analyze(bitmap: bmp, model: model, modelDirectory: modelDirectory)
    }

    static func analyze(bitmap bmp: Bitmap, model: Model,
                        modelDirectory: URL) throws -> Output {
        let dir = modelDirectory.path
        guard FileManager.default.fileExists(atPath: dir) else {
            throw AnalyzeError.modelsNotFound(dir)
        }
        let w = Double(bmp.width), h = Double(bmp.height)

        if model == .csrnet {
            let r = try CSRNet.count(image: bmp, modelDir: dir)
            return Output(totalColonies: r.count, averageConfidence: 0,
                          modelUsed: model.rawValue, detections: [],
                          imageWidth: w, imageHeight: h,
                          countability: CFU.assess(count: r.count),
                          heatmapPNG: CSRNet.heatmapPNG(density: r, over: bmp),
                          imgsz: CSRNet.workingSize, escalated: false)
        }

        if model.isSAM {
            let r = try Pipeline.count(image: bmp, modelDir: dir, micro: model == .samMicro)
            // _sam_response(): the overlay circle is the minimum enclosing
            // circle of the mask's largest contour, and confidence is the mean
            // circularity as a percentage.
            let detections = r.colonies.compactMap { c -> Detection? in
                let pts = Contours.outerBoundary(c.mask, width: bmp.width, height: bmp.height)
                guard !pts.isEmpty else { return nil }
                let circle = Contours.minEnclosingCircle(pts)
                return Detection(cx: circle.cx, cy: circle.cy, radius: circle.r)
            }
            let conf = r.colonies.isEmpty ? 0
                : r.colonies.reduce(0) { $0 + $1.circularity } / Double(r.colonies.count) * 100
            return Output(totalColonies: r.count, averageConfidence: (conf * 10).rounded() / 10,
                          modelUsed: model.rawValue, detections: detections,
                          imageWidth: w, imageHeight: h,
                          countability: CFU.assess(count: r.count), heatmapPNG: nil,
                          imgsz: r.imgsz, escalated: r.escalated)
        }

        let r = try Detect.count(image: bmp, modelDir: dir, key: model.rawValue)
        return Output(totalColonies: r.count,
                      averageConfidence: (r.confidence * 10).rounded() / 10,
                      modelUsed: model.rawValue,
                      detections: r.boxes.map {
                          Detection(cx: $0.cx, cy: $0.cy, radius: $0.radius)
                      },
                      imageWidth: w, imageHeight: h,
                      countability: CFU.assess(count: r.count), heatmapPNG: nil,
                      imgsz: r.imgsz, escalated: false)
    }
}
