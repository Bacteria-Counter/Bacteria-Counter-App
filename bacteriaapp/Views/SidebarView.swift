import SwiftUI

struct SidebarView: View {
    @ObservedObject var viewModel: MainViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 24) {
                SidebarSection(title: "DEVICE") {
                    DeviceCard(
                        deviceName: viewModel.deviceName,
                        isConnected: viewModel.isDeviceConnected
                    )
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
                
                if viewModel.showColonyCount {
                    VStack(spacing: 16) {
                        ColonyCountPanel(
                            count: viewModel.colonyCount,
                            isAnalyzing: viewModel.appState == .analyzing,
                            progress: viewModel.analysisProgress,
                            isComplete: viewModel.appState == .complete
                        )

//                        if viewModel.appState == .complete {
//                            SecondaryButton(
//                                title: "Export Report",
//                                icon: "square.and.arrow.down",
//                                action: viewModel.exportReport
//                            )
//                        }
                    }
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
            switch viewModel.appState {
            case .disconnected:
                PrimaryButton(
                    title: "Connect iPhone",
                    icon: "wifi",
                    isLoading: viewModel.isConnecting,
                    action: viewModel.connectDevice
                )

            case .connected:
                PrimaryButton(
                    title: "Capture Plate",
                    icon: "camera.fill",
                    isLoading: viewModel.isCapturing,
                    action: viewModel.capturePlate
                )

            case .analyzing, .complete:
                EmptyView()
            }
        }
    }
}

#Preview {
    SidebarView(viewModel: MainViewModel())
        .frame(height: 600)
}
