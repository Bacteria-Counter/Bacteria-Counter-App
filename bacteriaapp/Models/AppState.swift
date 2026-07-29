import Foundation

enum AppState: Equatable {
    case disconnected
    case connected
    case analyzing
    case complete
}

struct CaptureSettings: Equatable {
    var resolution: String = "-"
    var flash: String = "-"
    var zoom: String = "-"
    var focus: String = "-"

    static let unavailable = CaptureSettings()
}

struct AnalysisResult: Equatable {
    let totalColonies: Int
    let speciesCount: Int
    let averageConfidence: Int
    let plateType: String

    static let sample = AnalysisResult(
        totalColonies: 29,
        speciesCount: 4,
        averageConfidence: 84,
        plateType: "LB Agar 90mm"
    )
}
