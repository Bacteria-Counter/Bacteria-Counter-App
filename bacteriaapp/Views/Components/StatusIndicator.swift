import SwiftUI

struct StatusIndicator: View {
    let deviceName: String?
    let isConnected: Bool

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(isConnected ? AppTheme.accentGreen : Color.red.opacity(0.7))
                .frame(width: 7, height: 7)

            Text(isConnected ? (deviceName ?? "Connected") : "No Device")
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textSecondary)

            Image(systemName: "gearshape")
                .font(.system(size: 12))
                .foregroundStyle(AppTheme.textMuted)
                .padding(.leading, 4)
        }
    }
}

#Preview {
    HStack(spacing: 24) {
        StatusIndicator(deviceName: nil, isConnected: false)
        StatusIndicator(deviceName: "iPhone 15 Pro", isConnected: true)
    }
    .padding()
    .background(AppTheme.background)
}
