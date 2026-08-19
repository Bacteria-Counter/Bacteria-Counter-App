import AgarScopeKit
import AppKit
import CoreGraphics
import Foundation


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

  
    var modelDirectory: URL? = AgarScope.defaultModelDirectory()

    private let clahePreprocessor = CLAHEGrayscalePreprocessor()
    private let yoloDetector = YOLODetector()

    func analyze(image: CGImage, model: ModelChoice,
                 usedFullFrame: Bool) async throws -> AnalysisResult {
        switch model.engine {
        case .agarScope: try await runAgarScope(image, model, usedFullFrame)
        case .labYOLO: try await runLabYOLO(image, model, usedFullFrame)
        }
    }

    // MARK: - AgarScope

    private func runAgarScope(_ image: CGImage, _ model: ModelChoice,
                              _ usedFullFrame: Bool) async throws -> AnalysisResult {
        guard let dir = modelDirectory else {
            throw InferenceError.modelsMissing("(bundel aplikasi)")
        }
        guard let engineModel = AgarScope.Model(rawValue: model.rawValue) else {
            throw InferenceError.failed("Model \(model.rawValue) tidak tersedia on-device.")
        }

        let output: AgarScope.Output
        do {
            output = try await Task.detached(priority: .userInitiated) {
                try AgarScope.analyze(image: image, model: engineModel, modelDirectory: dir)
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
                ColonyDetection(x: $0.x, y: $0.y, width: $0.width, height: $0.height)
            },
            imageWidth: output.imageWidth,
            imageHeight: output.imageHeight,
            heatmapImage: output.heatmapPNG.flatMap { NSImage(data: $0) },
            usedFullFrame: usedFullFrame
        )
    }

    // MARK: - Lab YOLO

    private func runLabYOLO(_ image: CGImage, _ model: ModelChoice,
                            _ usedFullFrame: Bool) async throws -> AnalysisResult {
        guard let variant = YOLOModelVariant(rawValue: model.rawValue) else {
            throw InferenceError.failed("Varian \(model.rawValue) tidak dikenal.")
        }

        let boxes: [BoundingBox]
        do {
            let prepared = try clahePreprocessor.preprocess(image)
            boxes = try await yoloDetector.detect(in: prepared, variant: variant)
        } catch {
            throw InferenceError.failed(error.localizedDescription)
        }

        let size = CGSize(width: image.width, height: image.height)
        let confidences = boxes.map { Double($0.confidence) }
        let meanConfidence = confidences.isEmpty
            ? 0 : confidences.reduce(0, +) / Double(confidences.count) * 100

        return AnalysisResult(
            totalColonies: boxes.count,
            averageConfidence: Int(meanConfidence.rounded()),
            modelUsed: model,
            detections: boxes.map {
                let r = $0.rect(in: size)
                return ColonyDetection(x: r.minX, y: r.minY, width: r.width, height: r.height)
            },
            imageWidth: size.width,
            imageHeight: size.height,
            heatmapImage: nil,
            usedFullFrame: usedFullFrame
        )
    }
}
