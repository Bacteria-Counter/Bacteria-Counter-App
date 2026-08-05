import SwiftUI

struct SidebarView: View {
    @ObservedObject var viewModel: MainViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 24) {
                SidebarSection(title: "DEVICE") {
                    DeviceCard(
                        deviceName: viewModel.deviceName,
                        deviceType: viewModel.deviceType,
                        isConnected: viewModel.isDeviceConnected
                    )
                }

                SidebarSection(title: "MODEL") {
                    VStack(alignment: .leading, spacing: 6) {
                        Picker("Model", selection: Binding(
                            get: { viewModel.selectedModel },
                            set: { viewModel.selectModel($0) }
                        )) {
                            ForEach(ModelChoice.allCases) { choice in
                                Text(choice.fullDisplayName).tag(choice)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                        .disabled(viewModel.appState == .analyzing)

                        if let caveat = viewModel.selectedModel.caveat {
                            Text(caveat)
                                .font(AppTheme.monoSmall)
                                .foregroundStyle(AppTheme.accentOrange)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

                SidebarSection(title: "CAPTURE") {
                    VStack(spacing: 8) {
                        InfoRow(label: "Resolution", value: viewModel.captureSettings.resolution)
                        InfoRow(
                            label: "Flash",
                            value: viewModel.captureSettings.flash,
                            valueColor: AppTheme.accentOrange
                        )
                        InfoRow(
                            label: "Zoom",
                            value: viewModel.captureSettings.zoom,
                            valueColor: AppTheme.accentYellow
                        )
                        InfoRow(
                            label: "Focus",
                            value: viewModel.captureSettings.focus,
                            valueColor: AppTheme.accentGreen
                        )
                    }
                }
                
                if viewModel.showSegmentationStatus {
                    SegmentationStatusPanel(
                        isAnalyzing: viewModel.appState == .analyzing,
                        progress: viewModel.analysisProgress,
                        isComplete: viewModel.appState == .complete,
                        maskCoverage: viewModel.segmentationCoverage
                    )
                }

            }
            .padding(.horizontal, 20)
            .padding(.top, 24)

            Spacer()

            VStack(spacing: 12) {
                if let error = viewModel.connectionError {
                    Text(error)
                        .font(AppTheme.monoSmall)
                        .foregroundStyle(.red.opacity(0.8))
                        .multilineTextAlignment(.center)
                }

                actionButton
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 24)
        }
        .frame(width: AppTheme.sidebarWidth)
        .background(AppTheme.sidebarBackground)
        .overlay(alignment: .trailing) {
            Rectangle()
                .fill(AppTheme.border)
                .frame(width: 1)
        }
    }

    @ViewBuilder
    private var actionButton: some View {
        if viewModel.capturedImage != nil {
            PrimaryButton(
                title: "Recapture",
                icon: "arrow.clockwise",
                action: viewModel.newCapture
            )
        } else {
            VStack(spacing: 8) {
                switch viewModel.appState {
                case .disconnected:
            switch viewModel.appState {
            case .disconnected:
                VStack(spacing: 12) {
                    photoUploadButton

                    PrimaryButton(
                        title: "Connect Camera",
                        icon: "wifi",
                        isLoading: viewModel.isConnecting,
                        action: viewModel.connectDevice
                    )

                case .connected:
                }

            case .connected:
                VStack(spacing: 12) {
                    photoUploadButton

                    PrimaryButton(
                        title: "Capture Plate",
                        icon: "camera.fill",
                        isLoading: viewModel.isCapturing,
                        action: viewModel.capturePlate
                    )

                case .analyzing, .complete:
                    EmptyView()
                }

                if viewModel.appState == .disconnected || viewModel.appState == .connected {
                    SecondaryButton(
                        title: "Upload Image",
                        icon: "square.and.arrow.up",
                        action: viewModel.uploadImage
                    )
                }
            }
        }
    }

    private var photoUploadButton: some View {
        PhotoUploadButton(
            onPhotoSelected: { viewModel.uploadPhoto(from: $0) },
            onError: viewModel.handlePhotoUploadError
        )
    }
}

#Preview {
    SidebarView(viewModel: MainViewModel())
        .frame(height: 600)
}
