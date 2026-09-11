import SwiftUI

struct HeaderBarView: View {
    let deviceName: String?
    let isConnected: Bool

    var body: some View {
        HStack {
            Spacer()

            HStack(spacing: 6) {
                Image(systemName: "camera.macro")
                    .font(.system(size: 11))
                    .foregroundStyle(AppTheme.textMuted)

                Text("AgarScope — Bacterial Colony Counter")
                    .font(AppTheme.monoSmall)
                    .foregroundStyle(AppTheme.textSecondary)
            }

            Spacer()

            StatusIndicator(deviceName: deviceName, isConnected: isConnected)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(AppTheme.background)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(AppTheme.border)
                .frame(height: 1)
        }
    }
}

#Preview {
    HeaderBarView(deviceName: "iPhone 15 Pro", isConnected: true)
}
