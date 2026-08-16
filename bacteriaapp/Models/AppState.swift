import AppKit
import Foundation

enum AppState: Equatable {
    case disconnected
    case connected
    case analyzing
    case complete
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
enum ModelChoice: String, CaseIterable, Identifiable {
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
        case .v11s, .v26s: .labYOLO
        }
    }

    /// CSRNet estimates a density map and never locates individual colonies, so
    /// there is nothing to draw a box around. It shows a heatmap instead. This
    /// is a property of the method, not a gap in the implementation.
    var producesBoxes: Bool { self != .csrnet }

    var displayName: String {
        switch self {
        case .samMicro: "SAM"
        case .mac1: "Mac1"
        case .csrnet: "CSRNet"
        case .v11s: "YOLOv11s"
        case .v26s: "YOLOv26s"
        }
    }

    /// Full label for the picker and the status bar. Named by the job rather
    /// than the architecture: naming them by architecture is what led to models
    /// being picked by the wrong criterion once already.
    var fullDisplayName: String {
        switch self {
        case .samMicro: "SAM — penghitung utama"
        case .mac1: "Mac1 — pembanding"
        case .csrnet: "CSRNet — pembanding"
        case .v11s: "YOLOv11s"
        case .v26s: "YOLOv26s"
        }
    }

    /// One line under the picker, about USING the model rather than about how it
    /// scored. A technician needs to know when to distrust the number in front
    /// of them; benchmark figures answer a question they did not ask, and a
    /// paragraph of them stops being read at all.
    var caveat: String? {
        switch self {
        case .samMicro:
            "Pakai ini untuk menghitung. Periksa ulang bila latar fotonya berubah."
        case .mac1:
            "Pembanding. Zoom untuk memeriksa kotak yang meragukan."
        case .csrnet:
            "Pembanding angka, tanpa kotak. Bisa melapor nol di cawan yang sangat sepi."
        case .v11s:
            "Melewatkan koloni yang sangat kecil."
        case .v26s:
            "Berhenti di 300 koloni. Cawan lebih padat akan dilaporkan kurang."
        }
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
