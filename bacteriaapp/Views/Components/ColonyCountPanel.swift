import SwiftUI

struct ColonyCountPanel: View {
    let count: Int
    let isAnalyzing: Bool
    let progress: Double
    let isComplete: Bool

    var body: some View {
        SidebarSection(title: "COLONY COUNT") {
            VStack(spacing: 12) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(count)")
                        .font(AppTheme.monoLarge)
                        .foregroundStyle(AppTheme.textPrimary)
                        .contentTransition(.numericText())
                        .animation(.easeOut(duration: 0.15), value: count)

                    if isComplete {
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
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 12))
                            .foregroundStyle(AppTheme.accentGreen)

                        Text("Counting Done")
                            .font(AppTheme.monoSmall)
                            .foregroundStyle(AppTheme.accentGreen)
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
        ColonyCountPanel(count: 18, isAnalyzing: true, progress: 0.6, isComplete: false)
    }
    .frame(width: 500, height: 400)
}
