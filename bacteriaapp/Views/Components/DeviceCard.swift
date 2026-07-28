import SwiftUI

struct DeviceCard: View {
    let deviceName: String?
    let isConnected: Bool

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: isConnected ? "iphone.radiowaves.left.and.right" : "wifi")
                .font(.system(size: 18))
                .foregroundStyle(isConnected ? AppTheme.accentGreen : AppTheme.textMuted)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 2) {
                Text(isConnected ? deviceName ?? "iPhone" : "Not Connected")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(isConnected ? AppTheme.textPrimary : AppTheme.textSecondary)

                Text(isConnected ? "Continuity Camera" : "Connect Your Phone")
                    .font(AppTheme.monoSmall)
                    .foregroundStyle(AppTheme.textMuted)
            }

            Spacer()
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isConnected ? AppTheme.accentGreen.opacity(0.08) : AppTheme.cardBackground)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(
                            isConnected ? AppTheme.accentGreenDim.opacity(0.6) : AppTheme.border,
                            lineWidth: 1
                        )
                )
        )
    }
}

#Preview {
    VStack(spacing: 16) {
        DeviceCard(deviceName: nil, isConnected: false)
        DeviceCard(deviceName: "iPhone 15 Pro", isConnected: true)
    }
    .padding()
    .frame(width: 240)
    .background(AppTheme.sidebarBackground)
}
