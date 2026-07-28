import AVFoundation
import AppKit
import Combine
import CoreImage

@MainActor
final class CameraService: NSObject, ObservableObject {
    @Published private(set) var isSessionRunning = false
    @Published private(set) var connectedDeviceName: String?
    @Published private(set) var previewSession: AVCaptureSession?

    private var captureSession: AVCaptureSession?
    private var photoOutput: AVCapturePhotoOutput?
    private var currentDevice: AVCaptureDevice?
    private var inFlightCaptureDelegate: PhotoCaptureDelegate?
    private let sessionQueue = DispatchQueue(label: "com.ameliacitra.bacteriaapp.camera.session")

    func discoverContinuityCamera() -> String? {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [
                .external,
                .continuityCamera,
                .builtInWideAngleCamera
            ],
            mediaType: .video,
            position: .unspecified
        )

        let external = discovery.devices.first { device in
            device.deviceType == .continuityCamera ||
            device.deviceType == .external ||
            device.localizedName.localizedCaseInsensitiveContains("iPhone")
        }

        return external?.localizedName
    }

    func startSession() async throws {
        guard captureSession == nil else {
            if let session = captureSession, !session.isRunning {
                await withCheckedContinuation { continuation in
                    sessionQueue.async {
                        session.startRunning()
                        continuation.resume()
                    }
                }
                isSessionRunning = true
            }
            return
        }

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.continuityCamera, .external, .builtInWideAngleCamera],
            mediaType: .video,
            position: .unspecified
        )

        guard let device = discovery.devices.first(where: {
            $0.deviceType == .continuityCamera ||
            $0.deviceType == .external ||
            $0.localizedName.localizedCaseInsensitiveContains("iPhone")
        }) ?? discovery.devices.first else {
            throw CameraError.noDeviceFound
        }

        let session = AVCaptureSession()
        session.sessionPreset = .photo

        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAddInput }
        session.addInput(input)

        let output = AVCapturePhotoOutput()
        guard session.canAddOutput(output) else { throw CameraError.cannotAddOutput }
        session.addOutput(output)

        captureSession = session
        photoOutput = output
        currentDevice = device
        connectedDeviceName = device.localizedName
        previewSession = session

        await withCheckedContinuation { continuation in
            sessionQueue.async {
                session.startRunning()
                continuation.resume()
            }
        }
        isSessionRunning = true
    }

    func stopSession() {
        sessionQueue.async { [captureSession] in
            captureSession?.stopRunning()
        }
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

        var errorDescription: String? {
            switch self {
            case .noDeviceFound: "No Continuity Camera device found"
            case .cannotAddInput: "Cannot add camera input"
            case .cannotAddOutput: "Cannot add photo output"
            case .notConnected: "Camera is not connected"
            case .captureFailed: "Photo capture failed"
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
