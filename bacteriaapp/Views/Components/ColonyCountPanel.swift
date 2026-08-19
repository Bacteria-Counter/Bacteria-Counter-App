import SwiftUI

struct ColonyCountPanel: View {
    let count: Int
    let isAnalyzing: Bool
    let progress: Double
    let isComplete: Bool

    private var isUncountable: Bool {
        count > 250
    }

    var body: some View {
        SidebarSection(title: "COLONY COUNT") {
            VStack(spacing: 12) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(count)")
                        .font(AppTheme.monoLarge)
                        .foregroundStyle(AppTheme.textPrimary)
                        .contentTransition(.numericText())
                        .animation(.easeOut(duration: 0.15), value: count)

                    if isUncountable {
                        Text("TNTC")
                            .font(AppTheme.monoSmall)
                            .foregroundStyle(.orange)
                            .padding(.bottom, 12)
                    } else if isComplete {
                        Text("total")
                            .font(AppTheme.monoSmall)
                            .foregroundStyle(AppTheme.textMuted)
                            .padding(.bottom, 12)
                    }
                }
                .frame(maxWidth: .infinity)

                if isAnalyzing {
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(AppTheme.accentGreenDim.opacity(0.3))
                                .frame(height: 3)

                            RoundedRectangle(cornerRadius: 2)
                                .fill(AppTheme.accentGreen)
                                .frame(width: geometry.size.width * progress, height: 3)
                                .animation(.easeInOut(duration: 0.1), value: progress)
                        }
                    }
                    .frame(height: 3)
                }

                if isComplete {
                    HStack(spacing: 6) {
                        Image(systemName: isUncountable ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(isUncountable ? .orange : AppTheme.accentGreen)

                        Text(isUncountable ? "Too Numerous to Count" : "Counting Done")
                            .font(AppTheme.monoSmall)
                            .foregroundStyle(isUncountable ? .orange : AppTheme.accentGreen)
                    }
                    .padding(.top, 8)
                }
            }
        }
    }
}

#Preview {
    HStack(spacing: 0) {
        Color.black
        ColonyCountPanel(count: 312, isAnalyzing: false, progress: 1.0, isComplete: true)
    }
    .frame(width: 500, height: 400)
}
