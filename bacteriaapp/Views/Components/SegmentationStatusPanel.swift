import SwiftUI

struct SegmentationStatusPanel: View {
    let isAnalyzing: Bool
    let progress: Double
    let isComplete: Bool
    let maskCoverage: Double?

    var body: some View {
        SidebarSection(title: "PETRI DISH") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Image(
                        systemName: isComplete
                            ? "checkmark.circle.fill"
                            : "viewfinder.circle"
                    )
                    .foregroundStyle(AppTheme.accentGreen)

                    Text(isComplete ? "Dish Segmented" : "Segmenting Dish")
                        .font(AppTheme.monoSmall)
                        .foregroundStyle(
                            isComplete
                                ? AppTheme.accentGreen
                                : AppTheme.textPrimary
                        )
                }

                if isAnalyzing {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                        .tint(AppTheme.accentGreen)
                }

                if isComplete, let maskCoverage {
                    Text(
                        "Detected area: \(maskCoverage.formatted(.percent.precision(.fractionLength(1))))"
                    )
                    .font(AppTheme.monoSmall)
                    .foregroundStyle(AppTheme.textPrimary)

                    Text("Green area = segmented petri dish")
                        .font(AppTheme.monoSmall)
                        .foregroundStyle(AppTheme.textSecondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

#Preview {
    SegmentationStatusPanel(
        isAnalyzing: false,
        progress: 1,
        isComplete: true,
        maskCoverage: 0.47
    )
    .frame(width: 240)
    .padding()
    .background(AppTheme.sidebarBackground)
}
