import SwiftUI

struct MainViewportView: View {
    @ObservedObject var viewModel: MainViewModel

    var body: some View {
        ZStack {
            AppTheme.viewportBackground

            cameraContent

            if viewModel.appState == .disconnected {
                disconnectedOverlay
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var cameraContent: some View {
        // Every model is shown against the same picture -- the dish crop -- so
        // switching models changes the boxes and nothing else. Giving each
        // pipeline its own preprocessed image instead would make two models look
        // different for a reason that has nothing to do with the counting.
        if let plate = viewModel.preparedImage ?? viewModel.capturedImage {
            // CSRNet has no per-colony positions, so there are no boxes to draw;
            // its density heatmap goes here instead. That is the method's limit,
            // not a missing feature.
            if viewModel.appState == .complete,
               let heatmap = viewModel.analysisResult?.heatmapImage {
                CapturedImageView(image: heatmap)
            } else {
                CapturedImageView(
                    image: plate,
                    detections: viewModel.appState == .complete
                        ? (viewModel.analysisResult?.detections ?? []) : [],
                    detectionImageSize: viewModel.analysisResult.map {
                        CGSize(width: $0.imageWidth, height: $0.imageHeight)
                    }
                )
            }
        } else if viewModel.appState == .connected,
                  let session = viewModel.cameraService.previewSession {
            CameraPreviewView(session: session)
        } else {
            EmptyView()
        }
    }

    private var disconnectedOverlay: some View {
        VStack(spacing: 32) {
            CentralMessageView(
                icon: "camera",
                title: "No Camera Connected",
                description: "Connect your iPhone via Continuity Camera. If unavailable, the Mac camera will be used automatically."
            )

            SetupGuideView()
        }
    }
}

#Preview("Disconnected") {
    MainViewportView(viewModel: MainViewModel())
        .frame(width: 700, height: 500)
}
