import SwiftUI

enum AppTheme {
    static let background = Color(red: 0.04, green: 0.04, blue: 0.04)
    static let sidebarBackground = Color(red: 0.06, green: 0.06, blue: 0.06)
    static let viewportBackground = Color.black

    static let accentGreen = Color(red: 0.22, green: 0.85, blue: 0.45)
    static let accentGreenDim = Color(red: 0.15, green: 0.55, blue: 0.30)
    static let accentOrange = Color(red: 0.95, green: 0.65, blue: 0.25)
    static let accentYellow = Color(red: 0.90, green: 0.82, blue: 0.35)

    static let textPrimary = Color.white.opacity(0.92)
    static let textSecondary = Color.white.opacity(0.45)
    static let textMuted = Color(red: 0.35, green: 0.55, blue: 0.40)

    static let border = Color.white.opacity(0.08)
    static let cardBackground = Color.white.opacity(0.04)

    static let sidebarWidth: CGFloat = 240
    static let resultsPanelWidth: CGFloat = 220

    static let monoFont = Font.system(.body, design: .monospaced)
    static let monoSmall = Font.system(.caption, design: .monospaced)
    static let monoLarge = Font.system(size: 72, weight: .light, design: .monospaced)
}
