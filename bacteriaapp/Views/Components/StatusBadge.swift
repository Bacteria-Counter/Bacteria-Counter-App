import SwiftUI

struct StatusBadge: View {
    let number: Int
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(number)")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(AppTheme.accentGreen)
                .frame(width: 22, height: 22)
                .background(
                    Circle()
                        .stroke(AppTheme.accentGreenDim, lineWidth: 1)
                )

            Text(text)
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

#Preview {
    StatusBadge(number: 1, text: "Enable Continuity Camera on your iPhone.")
        .padding()
        .background(AppTheme.background)
}
