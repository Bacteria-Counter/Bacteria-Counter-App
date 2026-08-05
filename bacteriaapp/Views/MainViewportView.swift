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
        if let image = viewModel.capturedImage {
            // CSRNet has no discrete per-colony boxes -- when its density
            // heatmap comes back, show that instead of the raw photo +
            // circle overlay (which would just be empty circles anyway).
            if viewModel.appState == .complete,
               let heatmap = viewModel.analysisResult?.heatmapImage {
                CapturedImageView(image: heatmap, detections: [], detectionImageSize: nil)
            } else {
                CapturedImageView(
                    image: image,
                    detections: viewModel.appState == .complete ? (viewModel.analysisResult?.detections ?? []) : [],
                    detectionImageSize: viewModel.analysisResult.map {
                        CGSize(width: $0.imageWidth, height: $0.imageHeight)
                    }
                )
            }
            CapturedImageView(
                image: image,
                segmentationMask: viewModel.segmentationMask,
                imageSize: viewModel.analysisImageSize
            )
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
