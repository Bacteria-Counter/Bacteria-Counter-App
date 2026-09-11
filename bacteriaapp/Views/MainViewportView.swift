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
        if viewModel.appState == .cropping, let pending = viewModel.pendingCropImage {
            CropImageView(
                image: pending,
                onConfirm: { viewModel.confirmCrop($0) },
                onCancel: { viewModel.cancelCrop() }
            )
        } else if let plate = viewModel.preparedImage ?? viewModel.capturedImage {
            if viewModel.appState == .complete,
               let heatmap = viewModel.analysisResult?.heatmapImage {
                CapturedImageView(image: heatmap)
            } else {
                let isComplete = viewModel.appState == .complete
                CapturedImageView(
                    image: plate,
                    detections: isComplete ? viewModel.detections : [],
                    detectionImageSize: viewModel.analysisResult.map {
                        CGSize(width: $0.imageWidth, height: $0.imageHeight)
                    },
                    onRemove: isComplete ? { viewModel.removeDetection($0) } : nil,
                    onAdd: isComplete ? { viewModel.addDetection($0) } : nil
                )
            }
        } else if viewModel.appState == .connected,
                  let session = viewModel.cameraService.previewSession {
            CameraPreviewView(session: session)
                .aspectRatio(1, contentMode: .fit)
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
