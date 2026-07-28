import SwiftUI

struct StatusBarView: View {
    let message: String
    var icon: String = "waveform.path.ecg"

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 10))
                .foregroundStyle(AppTheme.textMuted)

            Text(message)
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textMuted)
                .lineLimit(1)

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(AppTheme.background)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(AppTheme.border)
                .frame(height: 1)
        }
    }
}

#Preview {
    StatusBarView(message: "Live preview · 12 MP · Macro · iPhone 15 Pro")
}
