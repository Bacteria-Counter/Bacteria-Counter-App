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
    @Published private(set) var segmentationMask: CGImage?
    @Published private(set) var segmentationCoverage: Double?
    @Published private(set) var analysisImageSize: CGSize = .zero
    @Published private(set) var analysisProgress: Double = 0
    @Published private(set) var captureSettings = CaptureSettings.unavailable
    @Published var connectionError: String?

    let cameraService = CameraService()

    var showSegmentationStatus: Bool {
        appState == .analyzing || appState == .complete
    }

    var statusBarMessage: String {
        switch appState {
        case .disconnected:
            return "Waiting for device connection"
        case .connected:
            return "Live preview · \(captureSettings.resolution) · \(captureSettings.focus) · \(deviceName ?? "Camera")"
        case .analyzing:
            return "Segmenting petri dish · \(Int(analysisProgress * 100))%"
        case .complete:
            return "Petri dish segmentation complete"
        }
    }

    var headerStatusText: String {
        deviceName ?? "No Device"
    }

    var isDeviceConnected: Bool {
        deviceName != nil
    }

    private var segmentationTask: Task<Void, Never>?
    private let segmenter = PetriDishSegmenter()

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
        segmentationTask?.cancel()
        segmentationTask = nil
        capturedImage = nil
        segmentationMask = nil
        segmentationCoverage = nil
        analysisImageSize = .zero
        analysisProgress = 0
        appState = standbyState
    }

    private func startAnalysis() {
        guard let capturedImage,
              let cgImage = Self.cgImage(from: capturedImage) else {
            connectionError = DishSegmentationError.imageConversionFailed.localizedDescription
            return
        }

        appState = .analyzing
        segmentationMask = nil
        segmentationCoverage = nil
        analysisImageSize = CGSize(
            width: cgImage.width,
            height: cgImage.height
        )
        analysisProgress = 0.1

        segmentationTask = Task {
            do {
                let result = try await segmenter.makeMask(from: cgImage)

                guard !Task.isCancelled else { return }
                segmentationMask = result.mask
                segmentationCoverage = result.foregroundFraction
                analysisProgress = 1
                appState = .complete
            } catch is CancellationError {
                return
            } catch {
                connectionError = error.localizedDescription
                analysisProgress = 0
                segmentationMask = nil
                segmentationCoverage = nil
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
