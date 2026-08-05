import AppKit
import Foundation

enum AppState: Equatable {
    case disconnected
    case connected
    case analyzing
    case complete
}

/// Which colony-counting pipeline the local inference server should run.
/// See server.py in the bacterial-colony-detection repo for what each does.
enum ModelChoice: String, CaseIterable, Identifiable {
    case yoloOld = "yolo_old"
    case yoloNew = "yolo_new"
    case sam
    case dogBlend = "dog_blend"
    case clahe
    case labAb = "lab_ab"
    case gsam2
    case csrnet

    var id: String { rawValue }

    /// Short label for tight spaces.
    var displayName: String {
        switch self {
        case .yoloOld: "Lama"
        case .yoloNew: "Baru"
        case .sam: "SAM"
        case .dogBlend: "DoG"
        case .clahe: "CLAHE"
        case .labAb: "LAB a/b"
        case .gsam2: "GroundedSAM2"
        case .csrnet: "CSRNet"
        }
    }

    /// Full label for places with more room (status bar, result summaries).
    var fullDisplayName: String {
        switch self {
        case .yoloOld: "YOLO (Lama)"
        case .yoloNew: "YOLO (Baru)"
        case .sam: "SAM"
        case .dogBlend: "YOLO + DoG-blend"
        case .clahe: "YOLO + CLAHE"
        case .labAb: "YOLO + LAB a/b"
        case .gsam2: "Colony Grounded SAM2"
        case .csrnet: "CSRNet (density map)"
        }
    }

    /// One-line caveat shown next to the picker so real-world testing goes
    /// in with eyes open, since ground-truth results don't tell the whole
    /// story on real photos (see server.py for the full validation notes).
    var caveat: String? {
        switch self {
        case .yoloOld, .yoloNew, .sam: nil
        case .dogBlend: "Eksperimental — mirip performa YOLO Baru"
        case .clahe: "Eksperimental — perbaikan sedang, belum divalidasi penuh"
        case .labAb: "Eksperimental — akurat di data uji, tapi berisiko meleset di koloni pucat"
        case .gsam2: "⚠️ Belum pernah dilatih ke data kita — pernah berhalusinasi di background kosong (36/36 foto). Uji dengan sangat hati-hati."
        case .csrnet: "⚠️ Training belum selesai — cenderung overcounting di foto padat/kompleks. Tidak ada lingkaran deteksi (cuma angka total), karena pendekatannya beda total dari yang lain."
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

/// A single detected colony, in the ORIGINAL captured image's pixel space
/// (not view/screen coordinates — the overlay view is responsible for
/// scaling these to wherever the image is actually displayed).
struct ColonyDetection: Equatable {
    let cx: Double
    let cy: Double
    let radius: Double
}

struct AnalysisResult: Equatable {
    let totalColonies: Int
    let averageConfidence: Int
    let modelUsed: ModelChoice
    let detections: [ColonyDetection]
    let imageWidth: Double
    let imageHeight: Double
    /// Only set for models with no discrete per-colony locations (currently
    /// CSRNet) -- a density-map heatmap to show instead of box/circle
    /// overlays.
    let heatmapImage: NSImage?

    static func == (lhs: AnalysisResult, rhs: AnalysisResult) -> Bool {
        lhs.totalColonies == rhs.totalColonies
            && lhs.averageConfidence == rhs.averageConfidence
            && lhs.modelUsed == rhs.modelUsed
            && lhs.detections == rhs.detections
            && lhs.imageWidth == rhs.imageWidth
            && lhs.imageHeight == rhs.imageHeight
            && lhs.heatmapImage === rhs.heatmapImage
    }
}
