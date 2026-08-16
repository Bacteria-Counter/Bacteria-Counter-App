import AgarScopeKit
import AppKit
import Foundation

/// Runs the colony-counting pipelines on this machine, through AgarScopeKit.
///
/// This used to POST the photo to a local Python server on port 8721, which had
/// to be started by hand before the app was any use. Everything it did now runs
/// in-process through Core ML: same models, same preprocessing, same filters,
/// verified plate for plate against the Python path before the change (see
/// AgarScopeKit/README.md). Server/server.py is still in the repo, but the app
/// no longer needs it running.
///
/// Two of the server's twelve options did not come across, and their absence is
/// deliberate rather than pending -- see AgarScope.Model.
struct InferenceService {
    enum InferenceError: LocalizedError {
        case modelsMissing(String)
        case invalidImage
        case failed(String)

        var errorDescription: String? {
            switch self {
            case .modelsMissing(let path):
                "Berkas model tidak ditemukan di \(path). Pasang ulang aplikasi atau setel "
                    + "AGARSCOPE_MODELS ke folder coreml_models."
            case .invalidImage:
                "Foto hasil tangkapan tidak bisa disiapkan untuk dianalisis."
            case .failed(let message):
                "Analisis gagal: \(message)"
            }
        }
    }

    /// Where the .mlpackage / .mlmodelc files live. Bundled with the app, so
    /// there is nothing to install; AGARSCOPE_MODELS overrides it, which is how
    /// the quantised set was measured against the float one.
    var modelDirectory: URL? = AgarScope.defaultModelDirectory()

    func analyze(image: NSImage, model: ModelChoice) async throws -> AnalysisResult {
        guard let cgImage = Self.cgImage(from: image) else {
            throw InferenceError.invalidImage
        }
        guard let dir = modelDirectory else {
            throw InferenceError.modelsMissing("(bundel aplikasi)")
        }
        guard let engineModel = AgarScope.Model(rawValue: model.rawValue) else {
            throw InferenceError.failed("Model \(model.rawValue) tidak tersedia on-device.")
        }

        // Off the main actor: a dense plate at imgsz 3200 takes seconds, and
        // this used to be a network call that never blocked the UI.
        let output: AgarScope.Output
        do {
            output = try await Task.detached(priority: .userInitiated) {
                try AgarScope.analyze(image: cgImage, model: engineModel, modelDirectory: dir)
            }.value
        } catch let error as AgarScope.AnalyzeError {
            if case .modelsNotFound(let path) = error { throw InferenceError.modelsMissing(path) }
            throw InferenceError.failed(error.localizedDescription)
        } catch {
            throw InferenceError.failed(error.localizedDescription)
        }

        return AnalysisResult(
            totalColonies: output.totalColonies,
            averageConfidence: Int(output.averageConfidence.rounded()),
            modelUsed: model,
            detections: output.detections.map {
                ColonyDetection(cx: $0.cx, cy: $0.cy, radius: $0.radius)
            },
            imageWidth: output.imageWidth,
            imageHeight: output.imageHeight,
            heatmapImage: output.heatmapPNG.flatMap { NSImage(data: $0) }
        )
    }

    /// Turn the count into a reportable CFU figure. Was POST /calculate-cfu;
    /// the rules themselves are ported in AgarScopeKit's CFU.
    func calculateCFU(plates: [CFU.Plate], dishAreaCm2: Double = CFU.defaultDishAreaCm2,
                      method: String = "pour", unit: String = "CFU/ml") throws -> CFU.Result {
        try CFU.calculate(plates, dishAreaCm2: dishAreaCm2, method: method, unit: unit)
    }

    private static func cgImage(from image: NSImage) -> CGImage? {
        // Straight from the NSImage, with no JPEG round trip. The old path had
        // to encode to JPEG to put the photo in an HTTP body, and that cost a
        // measured 1-4 units per pixel; nothing needs to pay it now.
        var rect = CGRect(origin: .zero, size: image.size)
        return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }
}
