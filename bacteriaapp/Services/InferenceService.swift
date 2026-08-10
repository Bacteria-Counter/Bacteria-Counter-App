import AppKit
import Foundation

/// Talks to the local Python inference server (see server.py in the
/// bacterial-colony-detection repo). The server must be started manually
/// before analyzing a plate — see README for `start_server.sh`.
struct InferenceService {
    enum InferenceError: LocalizedError {
        case serverUnreachable
        case invalidImage
        case badResponse
        case serverError(String)

        var errorDescription: String? {
            switch self {
            case .serverUnreachable:
                "Can't reach the local model server. Start it with start_server.sh, then try again."
            case .invalidImage:
                "Could not prepare the captured photo for analysis."
            case .badResponse:
                "The model server returned an unexpected response."
            case .serverError(let message):
                "Analysis failed: \(message)"
            }
        }
    }

    private struct DetectionDTO: Decodable {
        let cx: Double
        let cy: Double
        let radius: Double
    }

    private struct CountabilityDTO: Decodable {
        let status: String
        let regulation: Int
        let reliable: Bool
        let densityPerCm2: Double
        let advisory: String
    }

    private struct AnalyzeResponse: Decodable {
        let totalColonies: Int
        let averageConfidence: Double
        let modelUsed: String
        let detections: [DetectionDTO]
        let imageWidth: Double
        let imageHeight: Double
        /// Only present for models with no discrete per-colony locations
        /// (currently CSRNet) -- a density-map heatmap blended over the
        /// original photo, base64-encoded JPEG, in place of box/circle
        /// overlays.
        let heatmapImage: String?
        /// APHA 2002 reliability labelling of the count (25-250 countable,
        /// outside that estimate-only). Optional so older servers still work.
        let countability: CountabilityDTO?
    }

    private struct ServerErrorBody: Decodable {
        let error: String
    }

    var baseURL = URL(string: "http://127.0.0.1:8721")!

    func analyze(image: NSImage, model: ModelChoice) async throws -> AnalysisResult {
        guard let jpegData = Self.jpegData(from: image) else {
            throw InferenceError.invalidImage
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("analyze"))
        request.httpMethod = "POST"
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = Self.makeMultipartBody(jpegData: jpegData, model: model, boundary: boundary)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw InferenceError.serverUnreachable
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw InferenceError.badResponse
        }

        guard httpResponse.statusCode == 200 else {
            if let errorBody = try? JSONDecoder().decode(ServerErrorBody.self, from: data) {
                throw InferenceError.serverError(errorBody.error)
            }
            throw InferenceError.badResponse
        }

        guard let decoded = try? JSONDecoder().decode(AnalyzeResponse.self, from: data),
              let modelUsed = ModelChoice(rawValue: decoded.modelUsed) else {
            throw InferenceError.badResponse
        }

        let heatmapImage: NSImage? = decoded.heatmapImage
            .flatMap { Data(base64Encoded: $0) }
            .flatMap { NSImage(data: $0) }

        return AnalysisResult(
            totalColonies: decoded.totalColonies,
            averageConfidence: Int(decoded.averageConfidence.rounded()),
            modelUsed: modelUsed,
            detections: decoded.detections.map { ColonyDetection(cx: $0.cx, cy: $0.cy, radius: $0.radius) },
            imageWidth: decoded.imageWidth,
            imageHeight: decoded.imageHeight,
            countability: decoded.countability.map {
                Countability(status: $0.status, regulation: $0.regulation,
                             reliable: $0.reliable, densityPerCm2: $0.densityPerCm2,
                             advisory: $0.advisory)
            },
            heatmapImage: heatmapImage
        )
    }

    private static func jpegData(from image: NSImage) -> Data? {
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData) else {
            return nil
        }
        return bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.9])
    }

    private static func makeMultipartBody(jpegData: Data, model: ModelChoice, boundary: String) -> Data {
        var body = Data()

        func appendField(name: String, value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }

        appendField(name: "model", value: model.rawValue)

        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"plate.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(jpegData)
        body.append("\r\n".data(using: .utf8)!)

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        return body
    }
}
