import SwiftUI

struct ResultsPanelView: View {
    @ObservedObject var viewModel: MainViewModel

    var body: some View {
        VStack(spacing: 0) {
            SegmentationStatusPanel(
                isAnalyzing: viewModel.appState == .analyzing,
                progress: viewModel.analysisProgress,
                isComplete: viewModel.appState == .complete,
                maskCoverage: viewModel.segmentationCoverage
            )
        }
        .frame(width: AppTheme.resultsPanelWidth)
        .background(AppTheme.sidebarBackground)
    }
}

#Preview {
    ResultsPanelView(viewModel: {
        let vm = MainViewModel()
        return vm
    }())
    .frame(height: 500)
}
