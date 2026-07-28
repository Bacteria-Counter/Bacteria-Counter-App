import SwiftUI

struct InfoRow: View {
    let label: String
    let value: String
    var valueColor: Color = AppTheme.textPrimary

    var body: some View {
        HStack {
            Text(label)
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textMuted)

            Spacer()

            Text(value)
                .font(AppTheme.monoSmall)
                .foregroundStyle(valueColor)
        }
    }
}

#Preview {
    VStack(spacing: 8) {
        InfoRow(label: "Resolution", value: "12 MP")
        InfoRow(label: "Flash", value: "Auto", valueColor: AppTheme.accentOrange)
        InfoRow(label: "Focus", value: "Macro", valueColor: AppTheme.accentGreen)
    }
    .padding()
    .background(AppTheme.sidebarBackground)
}
