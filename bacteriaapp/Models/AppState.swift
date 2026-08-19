import AppKit
import Foundation

enum AppState: Equatable {
    case disconnected
    case connected
    case analyzing
    case complete
    case cropping
}

/// Which counting model to run.
///
/// Six models from two codebases, listed flat. The two halves reach the model
/// by different routes -- AgarScopeKit runs its own CLAHE and picks an input
/// size per photo, the lab YOLO models get a grayscale pass at a size fixed in
/// their Core ML export -- but the dish crop in front of both is now shared, so
/// there is one preprocessing decision a technician could care about and it has
/// already been made for them. Showing an engine picker would ask them to know
/// which team wrote which model, which is not a question about plates.
/// YOLOv26n is deliberately absent. It reports zero colonies on dense plates --
/// six of the 126 PCA plates and both dense lab photos -- and the cause is in
/// the checkpoint, not the integration: on an image where YOLOv26s finds 220
/// colonies at 0.61-0.90 confidence from the identical crop, v26n's highest
/// confidence is 0.0005. Its authors are investigating. The model file is still
/// in the bundle, so putting the case back is a one-line change.
/// The raw values are the model FILE names on disk (`mac1_1280.mlpackage` and
/// friends) and the keys AgarScopeKit dispatches on, so they are not display
/// text and must not be renamed casually. `mac1` is a YOLOv8n, confirmed twice:
/// its checkpoint holds 3,108,116 parameters and its int8 Core ML weights are
/// 3.3 MB, both of which are the v8n scale and nowhere near v8s's ~11 M.
/// Renaming the files, the benchmark scripts and the reports to match the
/// displayed names is a separate pass.
enum ModelChoice: String, CaseIterable, Identifiable {
    case v26s_new
    case samMicro = "sam_micro"
    case mac1
    case csrnet
    case v11s
    case v26s

    var id: String { rawValue }

    /// Which of the two pipelines runs this model. Not shown in the UI; the
    /// view model needs it to decide which route to take after the crop.
    enum Engine { case agarScope, labYOLO }

    var engine: Engine {
        switch self {
        case .samMicro, .mac1, .csrnet: .agarScope
        case .v11s, .v26s, .v26s_new: .labYOLO
        }
    }

    /// CSRNet estimates a density map and never locates individual colonies, so
    /// there is nothing to draw a box around. It shows a heatmap instead. This
    /// is a property of the method, not a gap in the implementation.
    var producesBoxes: Bool { self != .csrnet }

    /// Every model counts on the cropped dish.
    ///
    /// FastSAM was briefly taken off the crop while counting appeared to stall
    /// on real captures. That turned out to be the build, not the pipeline: the
    /// same photo takes 3.1 s in Release and 255.8 s in Debug, for identical
    /// counts. The crop is back for it.
    ///
    /// For the lab YOLO models this was never optional -- they were trained on
    /// cropped plates and lose 5 to 19 MAE without it.
    var usesCrop: Bool { true }

    var displayName: String {
        switch self {
        case .v26s_new: "YOLOv26s"
        case .samMicro: "FastSAM"
        case .mac1: "YOLOv8n"
        case .csrnet: "CSRNet"
        case .v11s: "YOLOv11s"
        case .v26s: "YOLOv26s"
        }
    }

    /// Full label for the picker and the status bar. Named by the job rather
    /// than the architecture: naming them by architecture is what led to models
    /// being picked by the wrong criterion once already.
    var fullDisplayName: String { displayName }
    
    static var visibleCases: [ModelChoice] {
        allCases.filter { $0 != .mac1 && $0 != .csrnet && $0 != .v11s && $0 != .v26s }
    }
}

struct CaptureSettings: Equatable {
    var resolution: String = "-"
    var flash: String = "-"
    var zoom: String = "-"
    var focus: String = "-"

    static let unavailable = CaptureSettings()
}

/// One detected colony as a box, in the pixel space of the image the pipeline
/// actually measured -- which after cropping is the cropped dish, not the
/// original photo. The overlay view scales these onto wherever that image ends
/// up being laid out; nothing here knows how it is displayed.
///
/// A box rather than a circle because colonies are not reliably round. A circle
/// over an irregular or merged colony either clips it or claims agar around it.
struct ColonyDetection: Equatable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct AnalysisResult: Equatable {
    let totalColonies: Int
    let averageConfidence: Int
    let modelUsed: ModelChoice
    let detections: [ColonyDetection]
    /// Size of the image the detections were measured against.
    let imageWidth: Double
    let imageHeight: Double
    /// Only set for models with no discrete per-colony locations (currently
    /// CSRNet) -- a density heatmap to show instead of the box overlay.
    let heatmapImage: NSImage?
    /// True when the dish segmentation failed and the AgarScope models fell
    /// back to the uncropped photo, so the status bar can say so. The lab YOLO
    /// models cannot fall back; they error instead.
    let usedFullFrame: Bool

    static func == (lhs: AnalysisResult, rhs: AnalysisResult) -> Bool {
        lhs.totalColonies == rhs.totalColonies
            && lhs.averageConfidence == rhs.averageConfidence
            && lhs.modelUsed == rhs.modelUsed
            && lhs.detections == rhs.detections
            && lhs.imageWidth == rhs.imageWidth
            && lhs.imageHeight == rhs.imageHeight
            && lhs.heatmapImage === rhs.heatmapImage
            && lhs.usedFullFrame == rhs.usedFullFrame
    }
}
