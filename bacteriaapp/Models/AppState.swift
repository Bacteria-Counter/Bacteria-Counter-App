import Foundation

enum AppState: Equatable {
    case disconnected
    case connected
    case analyzing
    case complete
}

struct CaptureSettings {
    var resolution: String = "12 MP"
    var flash: String = "Auto"
    var zoom: String = "1.0×"
    var focus: String = "Macro"
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
