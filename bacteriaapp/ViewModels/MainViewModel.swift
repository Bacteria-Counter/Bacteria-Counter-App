import AppKit
import Combine
import SwiftUI

@MainActor
final class MainViewModel: ObservableObject {
    @Published private(set) var appState: AppState = .disconnected
    @Published private(set) var deviceName: String?
    @Published private(set) var isConnecting = false
    @Published private(set) var isCapturing = false
    @Published private(set) var capturedImage: NSImage?
    @Published private(set) var colonyCount: Int = 0
    @Published private(set) var analysisProgress: Double = 0
    @Published private(set) var analysisResult: AnalysisResult?
    @Published var captureSettings = CaptureSettings()
    @Published var connectionError: String?

    let cameraService = CameraService()

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
            let result = analysisResult ?? .sample
            return "Complete · \(result.totalColonies) colonies · \(result.speciesCount) species · avg conf \(result.averageConfidence)% · \(result.plateType)"
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
                deviceName = cameraService.connectedDeviceName ?? cameraService.discoverContinuityCamera()
                if deviceName == nil {
                    deviceName = "iPhone"
                }
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

    func newCapture() {
        analysisTask?.cancel()
        analysisTask = nil
        capturedImage = nil
        colonyCount = 0
        analysisProgress = 0
        analysisResult = nil
        appState = .connected
    }

    private func startAnalysis() {
        appState = .analyzing
        colonyCount = 0
        analysisProgress = 0

        let targetCount = AnalysisResult.sample.totalColonies

        analysisTask = Task {
            let steps = 40
            for step in 0...steps {
                guard !Task.isCancelled else { return }

                let progress = Double(step) / Double(steps)
                analysisProgress = progress
                colonyCount = Int(Double(targetCount) * progress)

                try? await Task.sleep(for: .milliseconds(80))
            }

            guard !Task.isCancelled else { return }

            colonyCount = targetCount
            analysisProgress = 1.0
            analysisResult = AnalysisResult.sample
            appState = .complete
        }
    }

    func exportReport() {
        // Placeholder for export functionality
    }
}
