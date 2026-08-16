import AppKit
import Combine
import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class MainViewModel: ObservableObject {
    @Published private(set) var appState: AppState = .disconnected
    @Published private(set) var deviceName: String?
    @Published private(set) var deviceType: String?
    @Published private(set) var isConnecting = false
    @Published private(set) var isCapturing = false
    @Published private(set) var capturedImage: NSImage?
    @Published private(set) var colonyCount: Int = 0
    @Published private(set) var analysisProgress: Double = 0
    @Published private(set) var analysisResult: AnalysisResult?
    @Published private(set) var captureSettings = CaptureSettings.unavailable
    @Published var connectionError: String?
    // Defaults to the counter rather than the sterility check: counting is
    // what the app is opened for, and SAM is the most accurate option on
    // bright plates by a wide margin (MAE 5.4 against Mac1's 8.3).
    @Published var selectedModel: ModelChoice = .samMicro

    let cameraService = CameraService()
    private let inferenceService = InferenceService()

    var showColonyCount: Bool {
        appState == .analyzing || appState == .complete
    }

    var statusBarMessage: String {
        switch appState {
        case .disconnected:
            return "Waiting for device connection"
        case .connected:
            return "Live preview · \(captureSettings.resolution) · \(captureSettings.focus) · \(deviceName ?? "Camera")"
        case .analyzing:
            return "Analyzing · \(Int(analysisProgress * 100))% · \(colonyCount) colonies detected"
        case .complete:
            guard let result = analysisResult else { return "Complete" }
            return "Complete · \(result.totalColonies) colonies · avg conf \(result.averageConfidence)% · \(result.modelUsed.fullDisplayName) model"
        }
    }

    var headerStatusText: String {
        deviceName ?? "No Device"
    }

    var isDeviceConnected: Bool {
        deviceName != nil
    }

    private var analysisTask: Task<Void, Never>?

    func connectDevice() {
        guard appState == .disconnected else { return }
        isConnecting = true
        connectionError = nil

        Task {
            do {
                try await cameraService.startSession()
                guard let connectedDeviceName = cameraService.connectedDeviceName else {
                    throw CameraService.CameraError.noDeviceFound
                }
                deviceName = connectedDeviceName
                deviceType = cameraService.connectedDeviceType
                captureSettings = cameraService.captureSettings
                appState = .connected
            } catch {
                connectionError = error.localizedDescription
            }
            isConnecting = false
        }
    }

    func capturePlate() {
        guard appState == .connected, !isCapturing else { return }

        isCapturing = true
        connectionError = nil

        Task {
            defer { isCapturing = false }

            do {
                let image = try await cameraService.capturePhoto()
                capturedImage = image
                startAnalysis()
            } catch {
                connectionError = error.localizedDescription
            }
        }
    }

    /// Called by the sidebar model picker. Switching models after a plate
    /// has already been analyzed re-runs analysis on the SAME image with the
    /// new model — otherwise the displayed count/label would silently stay
    /// from whichever model ran last, which reads as "the picker doesn't do
    /// anything."
    func selectModel(_ model: ModelChoice) {
        guard selectedModel != model else { return }
        selectedModel = model
        if capturedImage != nil, appState != .analyzing {
            startAnalysis()
        }
    }

    func newCapture() {
        analysisTask?.cancel()
        analysisTask = nil
        capturedImage = nil
        colonyCount = 0
        analysisProgress = 0
        analysisResult = nil
        appState = isDeviceConnected ? .connected : .disconnected
    }

    /// Lets the user analyze a plate photo from disk instead of the camera —
    /// works regardless of whether a camera is connected.
    func uploadImage() {
        guard appState != .analyzing else { return }
        connectionError = nil

        let panel = NSOpenPanel()
        panel.title = "Select a Plate Photo"
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true

        guard panel.runModal() == .OK, let url = panel.url else { return }

        guard let image = NSImage(contentsOf: url) else {
            connectionError = "Could not load the selected image."
            return
        }

        capturedImage = image
        startAnalysis()
    }

    private func startAnalysis() {
        guard let image = capturedImage else { return }

        appState = .analyzing
        colonyCount = 0
        analysisProgress = 0

        let model = selectedModel

        analysisTask = Task {
            // Indeterminate progress while inference runs. The pipeline has
            // no meaningful intermediate percentage to report — it is one
            // Core ML pass plus filtering — so this is visual feedback, not
            // a measurement of how far along it is.
            let progressTask = Task {
                while !Task.isCancelled && analysisProgress < 0.9 {
                    analysisProgress += 0.03
                    try? await Task.sleep(for: .milliseconds(120))
                }
            }

            do {
                let result = try await inferenceService.analyze(image: image, model: model)
                progressTask.cancel()
                guard !Task.isCancelled else { return }

                colonyCount = result.totalColonies
                analysisProgress = 1.0
                analysisResult = result
                appState = .complete
            } catch {
                progressTask.cancel()
                guard !Task.isCancelled else { return }
                connectionError = error.localizedDescription
                appState = isDeviceConnected ? .connected : .disconnected
            }
        }
    }

    func exportReport() {
        // Placeholder for export functionality
    }
}
