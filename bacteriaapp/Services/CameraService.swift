@preconcurrency
import AVFoundation
import AppKit
import Combine
import CoreImage

@MainActor
final class CameraService: NSObject, ObservableObject {
    @Published private(set) var isSessionRunning = false
    @Published private(set) var connectedDeviceName: String?
    @Published private(set) var connectedDeviceType: String?
    @Published private(set) var previewSession: AVCaptureSession?

    private var captureSession: AVCaptureSession?
    private var photoOutput: AVCapturePhotoOutput?
    private var currentDevice: AVCaptureDevice?
    private var inFlightCaptureDelegate: PhotoCaptureDelegate?
    private let sessionQueue = DispatchQueue(label: "com.ameliacitra.bacteriaapp.camera.session")

    func startSession() async throws {
        guard captureSession == nil else {
            if let session = captureSession, !session.isRunning {
                await withCheckedContinuation { continuation in
                    sessionQueue.async {
                        session.startRunning()
                        continuation.resume()
                    }
                }
                guard session.isRunning else {
                    clearSession()
                    return try await startSession()
                }
                isSessionRunning = true
            }
            return
        }

        let candidates = cameraCandidates()
        guard !candidates.isEmpty else {
            throw CameraError.noDeviceFound
        }

        var lastError: Error?
        for candidate in candidates {
            do {
                try await startSession(with: candidate.device, type: candidate.type)
                return
            } catch {
                lastError = error
                clearSession()
            }
        }

        throw lastError ?? CameraError.noDeviceFound
    }

    private func startSession(with device: AVCaptureDevice, type: String) async throws {
        let session = AVCaptureSession()
        session.sessionPreset = .photo

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAddInput }
        session.addInput(input)

        let output = AVCapturePhotoOutput()
        guard session.canAddOutput(output) else { throw CameraError.cannotAddOutput }
        session.addOutput(output)

        await withCheckedContinuation { continuation in
            sessionQueue.async {
                session.startRunning()
                continuation.resume()
            }
        }

        guard session.isRunning else {
            throw CameraError.sessionFailed(device.localizedName)
        }

        captureSession = session
        photoOutput = output
        currentDevice = device
        connectedDeviceName = device.localizedName
        connectedDeviceType = type
        previewSession = session
        isSessionRunning = true
    }

    private func cameraCandidates() -> [(device: AVCaptureDevice, type: String)] {
        let iPhoneDiscovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.continuityCamera, .external],
            mediaType: .video,
            position: .unspecified
        )

        // Prefer a device explicitly identified by AVFoundation as Continuity
        // Camera. The name check keeps compatibility with iPhones that macOS
        // reports as a generic external camera.
        let iPhone = iPhoneDiscovery.devices.first(where: {
            $0.deviceType == .continuityCamera
        }) ?? iPhoneDiscovery.devices.first(where: {
            $0.localizedName.localizedCaseInsensitiveContains("iPhone")
        })

        let builtInDiscovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .unspecified
        )
        let builtIn = builtInDiscovery.devices.first

        return [
            iPhone.map { ($0, "Continuity Camera") },
            builtIn.map { ($0, "Mac built-in camera") }
        ].compactMap { $0 }
    }

    func stopSession() {
        sessionQueue.async { [captureSession] in
            captureSession?.stopRunning()
        }
        isSessionRunning = false
    }

    private func clearSession() {
        captureSession?.stopRunning()
        captureSession = nil
        photoOutput = nil
        currentDevice = nil
        connectedDeviceName = nil
        connectedDeviceType = nil
        previewSession = nil
        isSessionRunning = false
    }

    func capturePhoto() async throws -> NSImage {
        guard let photoOutput else { throw CameraError.notConnected }

        return try await withCheckedThrowingContinuation { continuation in
            let delegate = PhotoCaptureDelegate { [weak self] result in
                Task { @MainActor in
                    self?.inFlightCaptureDelegate = nil
                }
                continuation.resume(with: result)
            }
            inFlightCaptureDelegate = delegate

            let settings = Self.makePhotoSettings(for: photoOutput)

            sessionQueue.async {
                photoOutput.capturePhoto(with: settings, delegate: delegate)
            }
        }
    }

    private static func makePhotoSettings(for output: AVCapturePhotoOutput) -> AVCapturePhotoSettings {
        if output.availablePhotoCodecTypes.contains(.jpeg) {
            return AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.jpeg])
        }
        if let codec = output.availablePhotoCodecTypes.first {
            return AVCapturePhotoSettings(format: [AVVideoCodecKey: codec])
        }
        return AVCapturePhotoSettings()
    }

    enum CameraError: LocalizedError {
        case noDeviceFound
        case cannotAddInput
        case cannotAddOutput
        case notConnected
        case captureFailed
        case sessionFailed(String)

        var errorDescription: String? {
            switch self {
            case .noDeviceFound: "No camera found. Connect an iPhone or enable the Mac camera."
            case .cannotAddInput: "Cannot add camera input"
            case .cannotAddOutput: "Cannot add photo output"
            case .notConnected: "Camera is not connected"
            case .captureFailed: "Photo capture failed"
            case .sessionFailed(let deviceName): "Could not start \(deviceName)"
            }
        }
    }
}

private final class PhotoCaptureDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let completion: (Result<NSImage, Error>) -> Void

    init(completion: @escaping (Result<NSImage, Error>) -> Void) {
        self.completion = completion
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        guard !didComplete else { return }

        if let error {
            didComplete = true
            completion(.failure(error))
            return
        }

        if let image = Self.makeImage(from: photo) {
            didComplete = true
            completion(.success(image))
        }
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings,
        error: Error?
    ) {
        guard !didComplete else { return }

        didComplete = true
        if let error {
            completion(.failure(error))
        } else {
            completion(.failure(CameraService.CameraError.captureFailed))
        }
    }

    private var didComplete = false

    private static func makeImage(from photo: AVCapturePhoto) -> NSImage? {
        if let data = photo.fileDataRepresentation(), let image = NSImage(data: data) {
            return image
        }

        if let cgImage = photo.cgImageRepresentation() {
            return NSImage(
                cgImage: cgImage,
                size: NSSize(width: cgImage.width, height: cgImage.height)
            )
        }

        if let pixelBuffer = photo.pixelBuffer {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let rep = NSCIImageRep(ciImage: ciImage)
            let image = NSImage(size: rep.size)
            image.addRepresentation(rep)
            return image
        }

        return nil
    }
}
