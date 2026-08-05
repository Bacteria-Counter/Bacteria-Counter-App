import SwiftUI

struct ResultsPanelView: View {
    @ObservedObject var viewModel: MainViewModel

    var body: some View {
        VStack(spacing: 0) {
            ColonyCountPanel(
                count: viewModel.colonyCount,
                isAnalyzing: viewModel.appState == .analyzing,
                progress: viewModel.analysisProgress,
                isComplete: viewModel.appState == .complete
            )

//            if viewModel.appState == .complete {
//                SecondaryButton(
//                    title: "Export Report",
//                    icon: "square.and.arrow.down",
//                    action: viewModel.exportReport
//                )
//                .padding(.horizontal, 20)
//                .padding(.bottom, 24)
//            }
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
