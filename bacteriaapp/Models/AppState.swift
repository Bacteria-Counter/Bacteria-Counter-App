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
    let averageConfidence: Int
}
