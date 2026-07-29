import AppKit
import Combine
import SwiftUI

@MainActor
final class MainViewModel: ObservableObject {
    @Published private(set) var appState: AppState = .disconnected
    @Published private(set) var deviceName: String?
    @Published private(set) var deviceType: String?
    @Published private(set) var isConnecting = false
    @Published private(set) var isCapturing = false
    @Published private(set) var capturedImage: NSImage?
    @Published private(set) var colonyCount: Int = 0
    @Published private(set) var colonyBoxes: [ColonyBox] = []
    @Published private(set) var analysisImageSize: CGSize = .zero
    @Published private(set) var analysisProgress: Double = 0
    @Published private(set) var analysisResult: AnalysisResult?
    @Published private(set) var captureSettings = CaptureSettings.unavailable
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
            guard let result = analysisResult else {
                return "Analysis complete"
            }
            return "Complete · \(result.totalColonies) colonies · avg conf \(result.averageConfidence)%"
        }
    }

    var headerStatusText: String {
        deviceName ?? "No Device"
    }

    var isDeviceConnected: Bool {
        deviceName != nil
    }

    private var analysisTask: Task<Void, Never>?
    private let analysisService = ColonyAnalysisService()

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

    func uploadPhoto(from url: URL) {
        guard appState == .disconnected || appState == .connected else { return }

        connectionError = nil

        let hasSecurityScopedAccess = url.startAccessingSecurityScopedResource()
        defer {
            if hasSecurityScopedAccess {
                url.stopAccessingSecurityScopedResource()
            }
        }

        guard let image = NSImage(contentsOf: url) else {
            connectionError = "The selected file could not be opened as an image."
            return
        }

        capturedImage = image
        startAnalysis()
    }

    func handlePhotoUploadError(_ error: Error) {
        connectionError = "Could not upload photo: \(error.localizedDescription)"
    }

    func newCapture() {
        analysisTask?.cancel()
        analysisTask = nil
        capturedImage = nil
        colonyCount = 0
        colonyBoxes = []
        analysisImageSize = .zero
        analysisProgress = 0
        analysisResult = nil
        appState = standbyState
    }

    private func startAnalysis() {
        guard let capturedImage,
              let cgImage = Self.cgImage(from: capturedImage) else {
            connectionError = ColonyAnalysisError.imageConversionFailed.localizedDescription
            return
        }

        appState = .analyzing
        colonyCount = 0
        colonyBoxes = []
        analysisImageSize = CGSize(
            width: cgImage.width,
            height: cgImage.height
        )
        analysisProgress = 0

        analysisTask = Task {
            do {
                let result = try await analysisService.analyze(
                    image: cgImage
                ) { [weak self] progress, count in
                    self?.analysisProgress = progress
                    self?.colonyCount = count
                }

                guard !Task.isCancelled else { return }
                colonyCount = result.totalColonies
                colonyBoxes = result.boundingBoxes
                analysisProgress = 1
                analysisResult = AnalysisResult(
                    totalColonies: result.totalColonies,
                    averageConfidence: result.averageConfidence
                )
                appState = .complete
            } catch is CancellationError {
                return
            } catch {
                connectionError = error.localizedDescription
                analysisProgress = 0
                colonyCount = 0
                colonyBoxes = []
                appState = standbyState
            }
        }
    }

    private var standbyState: AppState {
        isDeviceConnected ? .connected : .disconnected
    }

    private static func cgImage(from image: NSImage) -> CGImage? {
        var rect = NSRect(origin: .zero, size: image.size)
        return image.cgImage(
            forProposedRect: &rect,
            context: nil,
            hints: nil
        )
    }

    func exportReport() {
        // Placeholder for export functionality
    }
}
