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
                        // One flat list of all six. Which codebase a model came
                        // from is not a question about plates, and the one
                        // preprocessing step a technician could reasonably care
                        // about -- the dish crop -- is shared by all of them.
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

                // Shown only when segmentation failed, because that changes what
                // the numbers mean: the AgarScope models are counting the whole
                // photo, and the lab models cannot run at all.
                if viewModel.usedFullFrame {
                    SidebarSection(title: "CAWAN") {
                        Text("Cawan tidak terdeteksi. SAM, Mac1, dan CSRNet menghitung dari foto utuh; model lab butuh potongan cawan dan tidak bisa dipakai pada foto ini.")
                            .font(AppTheme.monoSmall)
                            .foregroundStyle(AppTheme.accentOrange)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
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

                if viewModel.showColonyCount {
                    ColonyCountPanel(
                        count: viewModel.colonyCount,
                        isAnalyzing: viewModel.appState == .analyzing,
                        progress: viewModel.analysisProgress,
                        isComplete: viewModel.appState == .complete
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
                    PrimaryButton(
                        title: "Connect Camera",
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
}

#Preview {
    SidebarView(viewModel: MainViewModel())
        .frame(height: 600)
}
