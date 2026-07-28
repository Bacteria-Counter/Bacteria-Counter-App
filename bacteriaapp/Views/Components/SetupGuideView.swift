import SwiftUI

struct SetupGuideView: View {
    private let steps = [
        "Enable Continuity Camera on your iPhone.",
        "Ensure both devices share the same Apple ID.",
        "Connect to the same Wi-Fi network.",
        "Click Connect Camera. The Mac camera is used as fallback."
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("SETUP GUIDE")
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textMuted)
                .tracking(1.2)

            VStack(alignment: .leading, spacing: 14) {
                ForEach(Array(steps.enumerated()), id: \.offset) { index, step in
                    StatusBadge(number: index + 1, text: step)
                }
            }
        }
        .padding(20)
        .frame(maxWidth: 420)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(AppTheme.cardBackground)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(AppTheme.border, lineWidth: 1)
                )
        )
    }
}

#Preview {
    SetupGuideView()
        .padding(40)
        .background(Color.black)
}
