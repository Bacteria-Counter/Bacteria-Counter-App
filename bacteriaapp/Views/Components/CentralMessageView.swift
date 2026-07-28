import SwiftUI

struct CentralMessageView: View {
    let icon: String
    let title: String
    let description: String

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(AppTheme.accentGreenDim)
                .frame(width: 64, height: 64)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(AppTheme.accentGreenDim.opacity(0.4), lineWidth: 1)
                )

            Text(title)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(AppTheme.textPrimary)

            Text(description)
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)
        }
    }
}

#Preview {
    CentralMessageView(
        icon: "camera",
        title: "No Camera Connected",
        description: "Connect your iPhone via Continuity Camera to begin capturing agar plate images for ML analysis."
    )
    .padding()
    .background(Color.black)
}
