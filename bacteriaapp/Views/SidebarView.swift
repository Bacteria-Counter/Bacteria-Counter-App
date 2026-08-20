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
                    // One flat list. Which codebase a model came from is not a
                    // question about plates, and the one preprocessing step a
                    // technician could reasonably care about -- the dish crop --
                    // is shared by all of them.
                    Picker("Model", selection: Binding(
                        get: { viewModel.selectedModel },
                        set: { viewModel.selectModel($0) }
                    )) {
                        ForEach(ModelChoice.visibleCases) { choice in
                            Text(choice.displayName).tag(choice)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .disabled(viewModel.appState == .analyzing)
                }
                
                if viewModel.appState == .cropping {
                    SidebarSection(title: "CROP") {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Adjust the cropping area")
                                .font(AppTheme.monoFont)
                                .foregroundStyle(AppTheme.textPrimary)

                            Text("Move or resize the square to fit the petri dish. Line up the dish edge with the dashed circle guide, then confirm to start counting.")
                                .font(AppTheme.monoSmall)
                                .foregroundStyle(AppTheme.textSecondary)
                        }
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }


                if viewModel.usedFullFrame {
                    SidebarSection(title: "PETRI DISH") {
                        Text("The Petri dish was not detected. SAM, Mac1, and CSRNet count colonies directly from the full image, while the lab model requires a cropped Petri dish and cannot be used on this image.")
                            .font(AppTheme.monoSmall)
                            .foregroundStyle(AppTheme.accentOrange)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
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

                case .analyzing, .complete, .cropping:
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
