import AppKit
import Foundation

enum AppState: Equatable {
    case disconnected
    case connected
    case analyzing
    case complete
}

/// Which colony-counting pipeline to run. All of these now run on-device
/// through AgarScopeKit; there is no server to start.
///
/// Two options the server used to offer are gone, and both were deliberate:
///   - "sam" (the frozen ORIGINAL FastSAM settings) needed a Core ML export at
///     3840 that was never made. It existed only to reproduce pre-August 2026
///     numbers and was the least accurate option here (MAE 35.11 against
///     sam_tuned's 3.86). The FastSAM we actually rely on -- sam_tuned and
///     sam_micro -- is untouched.
///   - "gsam2" was never converted: 1.1 GB, a separate dependency tree, and it
///     failed the empty-plate safety check at 36/36 photos.
/// Server/server.py still has both if an old figure ever needs reproducing.
enum ModelChoice: String, CaseIterable, Identifiable {
    /// Three models, chosen from the ten the server offered by measuring all of
    /// them on 126 bright plates with ground truth, 34 real empty plates, and
    /// the lab photos (`coreml_tools/eval_yolo.py`). Six YOLO variants were cut
    /// as redundant: mac1 beat or matched every one of them on almost every
    /// axis. `sam_tuned` was cut in favour of `sam_micro`, which is identical
    /// except that it automatically re-runs at higher resolution when the
    /// colonies are pinpoint.
    ///
    /// What is left is three genuinely different jobs, not three rankings of
    /// the same job.
    case mac1
    case samMicro = "sam_micro"
    case csrnet

    var id: String { rawValue }

    /// Short label for tight spaces.
    var displayName: String {
        switch self {
        case .mac1: "Mac1"
        case .samMicro: "SAM"
        case .csrnet: "CSRNet"
        }
    }

    /// Full label for places with more room (status bar, result summaries).
    /// Named by the JOB each one does, because naming them by architecture is
    /// what led to them being picked by the wrong criterion.
    var fullDisplayName: String {
        switch self {
        case .mac1: "Mac1 — pastikan cawan bersih"
        case .samMicro: "SAM — hitung koloni"
        case .csrnet: "CSRNet — pendapat kedua"
        }
    }

    /// One-line caveat shown next to the picker. Written around the JOB each
    /// model does rather than its architecture, because naming them by
    /// architecture is what led to them being picked by the wrong criterion.
    ///
    /// Figures come from three benchmarks measured the same way
    /// (`coreml_tools/eval_yolo.py`, `eval_by_colony_size.py`,
    /// `eval_agar_bright.py`): 126 bright PCA plates, 34 real empty plates,
    /// AGAR's 99 held-out bright images, and the three lab photos whose true
    /// counts were confirmed by eye.
    var caveat: String? {
        switch self {
        case .samMicro:
            "Penghitung utama. Paling akurat di cawan terang (MAE 5,4 vs 8,3 Mac1) dan satu-satunya yang cocok dengan foto lab: 274 pada cawan yang benarnya ~280, dan 120 pada yang benarnya ~104. Otomatis mengulang di resolusi tinggi bila koloninya kecil. RAPUH terhadap latar baru: pada latar yang belum pernah ia lihat, ia bisa menghitung tekstur sebagai koloni tanpa memberi tanda."
        case .mac1:
            "Pembanding visual. Menggambar lingkaran per koloni, jadi kalau angkanya jauh berbeda dari SAM kamu bisa zoom dan menilai sendiri. Nol deteksi palsu di 34 cawan kosong (SAM: 64), jadi selisih besar antara keduanya adalah sinyal bahwa setup foto berubah. JANGAN dipakai sebagai angka utama: di foto lab ia melapor 428 untuk cawan berisi ~280, dan 21 untuk cawan berisi ~104."
        case .csrnet:
            "Pembanding angka. Satu-satunya tanpa bias sistematis (-0,02; yang lain overcount +4 sampai +10) dan nol deteksi palsu di cawan kosong. Tampil sebagai heatmap karena metode ini tidak menghasilkan posisi per koloni -- itu batasnya, bukan fiturnya, jadi selisih dengannya tidak bisa diperiksa dengan mata. Training-nya belum selesai."
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
