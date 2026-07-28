import SwiftUI

struct PrimaryButton: View {
    let title: String
    let icon: String
    var isLoading: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(AppTheme.accentGreen)

                HStack(spacing: 8) {
                    if isLoading {
                        ProgressView()
                            .controlSize(.small)
                            .tint(.black)
                    } else {
                        Image(systemName: icon)
                            .font(.system(size: 13, weight: .semibold))
                    }

                    Text(title)
                        .font(.system(size: 13, weight: .semibold))
                }
                .foregroundStyle(.black)
                .allowsHitTesting(false)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 41)
            .contentShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity)
        .contentShape(RoundedRectangle(cornerRadius: 8))
        .disabled(isLoading)
    }
}

#Preview {
    PrimaryButton(title: "Connect iPhone", icon: "wifi") {}
        .padding()
        .frame(width: 220)
        .background(AppTheme.sidebarBackground)
}
