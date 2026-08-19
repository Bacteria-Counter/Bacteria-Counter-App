import SwiftUI

struct PrimaryButton: View {
    let title: String
    let icon: String
    var isLoading: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .controlSize(.small)
                        .tint(.black)
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 15, weight: .semibold))
                }

                Text(title)
                    .font(.system(size: 15, weight: .semibold))
            }
            .foregroundStyle(.black)
            .frame(maxWidth: .infinity)
            .frame(height: 48) // Sedikit dinaikkan agar proporsinya mirip seperti di screenshot
            
            // KUNCI: Taruh background dan contentShape DI DALAM label Button
            .contentShape(Rectangle())
            .background(AppTheme.accentGreen)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .disabled(isLoading)
        // Efek opacity saat tombol sedang loading
        .opacity(isLoading ? 0.7 : 1.0)
    }
}
