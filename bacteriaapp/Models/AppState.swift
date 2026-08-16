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
enum ModelChoice: String, CaseIterable, Identifiable {
    case samMicro = "sam_micro"
    case mac1
    case csrnet
    case v11s
    case v26s
    case v26n

    var id: String { rawValue }

    /// Which of the two pipelines runs this model. Not shown in the UI; the
    /// view model needs it to decide which route to take after the crop.
    enum Engine { case agarScope, labYOLO }

    var engine: Engine {
        switch self {
        case .samMicro, .mac1, .csrnet: .agarScope
        case .v11s, .v26s, .v26n: .labYOLO
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
        case .v26n: "YOLOv26n"
        }
    }

    /// Full label for the picker and the status bar. Named by the job rather
    /// than the architecture: naming them by architecture is what led to models
    /// being picked by the wrong criterion once already.
    var fullDisplayName: String {
        switch self {
        case .samMicro: "SAM — penghitung utama"
        case .mac1: "Mac1 — pembanding visual"
        case .csrnet: "CSRNet — pembanding angka"
        case .v11s: "YOLOv11s — lab"
        case .v26s: "YOLOv26s — lab"
        case .v26n: "YOLOv26n — lab"
        }
    }

    /// One-line caveat under the picker.
    ///
    /// Every figure is from 126 bright PCA plates with ground truth, each model
    /// through its own chain, all with the shared dish crop in front
    /// (`coreml_tools/eval_cross.py`). The three lab photos are quoted only
    /// where their count was confirmed by eye, and they are still estimates
    /// until the measured ground truth lands.
    var caveat: String? {
        switch self {
        case .samMicro:
            "Penghitung utama, paling akurat di cawan terang (MAE 4,5). Otomatis mengulang di resolusi tinggi bila koloninya kecil. RAPUH terhadap latar baru: pada latar yang belum pernah ia lihat, ia bisa menghitung tekstur sebagai koloni tanpa memberi tanda."
        case .mac1:
            "Pembanding visual. Menggambar kotak per koloni, jadi kalau angkanya jauh berbeda dari SAM kamu bisa zoom dan menilai sendiri. Jangan dipakai sebagai angka utama."
        case .csrnet:
            "Pembanding angka, satu-satunya tanpa bias sistematis. Tampil sebagai heatmap karena metode ini tidak menghasilkan posisi per koloni, jadi selisihnya tidak bisa diperiksa dengan mata. Melapor NOL pada sebagian cawan yang isinya di bawah 5 koloni."
        case .v11s:
            "Model lab. Terbaik di data AGAR (MAE 0,6) tapi turun jauh di cawan PCA (11,0). Melewatkan koloni pinpoint."
        case .v26s:
            "Model lab, terbaik dari ketiganya di cawan PCA (MAE 9,7). TIDAK BISA melapor lebih dari 300 koloni: batas itu terkunci di berkas modelnya, dan cawan yang lebih padat akan berhenti di 300 tanpa memberi tanda."
        case .v26n:
            "Model lab. BELUM ANDAL: melapor NOL koloni pada sebagian cawan padat, termasuk dua foto lab. Sedang diperiksa oleh penulisnya."
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
