import SwiftUI

struct SecondaryButton: View {
    let title: String
    let icon: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .medium))

                Text(title)
                    .font(AppTheme.monoSmall)
            }
            .foregroundStyle(AppTheme.accentGreen)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(AppTheme.accentGreenDim, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    SecondaryButton(title: "New Capture", icon: "arrow.clockwise") {}
        .padding()
        .frame(width: 220)
        .background(AppTheme.sidebarBackground)
}
