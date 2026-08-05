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
    @Published private(set) var captureSettings = CaptureSettings.unavailable

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

        if let maximumDimensions = Self.maximumPhotoDimensions(for: device.activeFormat) {
            output.maxPhotoDimensions = maximumDimensions
        }

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
        captureSettings = Self.captureSettings(for: device, output: output)
        isSessionRunning = true
    }

    private func cameraCandidates() -> [(device: AVCaptureDevice, type: String)] {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.continuityCamera, .external, .builtInWideAngleCamera],
            mediaType: .video,
            position: .unspecified
        )

        // isContinuityCamera remains reliable even if an older configuration
        // reports the iPhone using a generic or built-in device type.
        let iPhone = discovery.devices.first(where: {
            $0.isContinuityCamera
        })

        let builtIn = discovery.devices.first(where: {
            !$0.isContinuityCamera && $0.deviceType == .builtInWideAngleCamera
        })

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
        captureSettings = .unavailable
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

            let settings = Self.makePhotoSettings(
                for: photoOutput,
                device: currentDevice
            )

            sessionQueue.async {
                photoOutput.capturePhoto(with: settings, delegate: delegate)
            }
        }
    }

    private static func makePhotoSettings(
        for output: AVCapturePhotoOutput,
        device: AVCaptureDevice?
    ) -> AVCapturePhotoSettings {
        let settings: AVCapturePhotoSettings

        if output.availablePhotoCodecTypes.contains(.jpeg) {
            settings = AVCapturePhotoSettings(
                format: [AVVideoCodecKey: AVVideoCodecType.jpeg]
            )
        } else if let codec = output.availablePhotoCodecTypes.first {
            settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: codec])
        } else {
            settings = AVCapturePhotoSettings()
        }

        let dimensions = output.maxPhotoDimensions
        if dimensions.width > 0, dimensions.height > 0 {
            settings.maxPhotoDimensions = dimensions
        }

        if let device, let flashMode = preferredFlashMode(for: device, output: output) {
            settings.flashMode = flashMode
        }

        return settings
    }

    private static func maximumPhotoDimensions(
        for format: AVCaptureDevice.Format
    ) -> CMVideoDimensions? {
        format.supportedMaxPhotoDimensions
            .filter { $0.width > 0 && $0.height > 0 }
            .max {
                Int64($0.width) * Int64($0.height)
                    < Int64($1.width) * Int64($1.height)
            }
    }

    private static func captureSettings(
        for device: AVCaptureDevice,
        output: AVCapturePhotoOutput
    ) -> CaptureSettings {
        guard device.isContinuityCamera else {
            return .unavailable
        }

        let dimensions = output.maxPhotoDimensions
        let resolution: String
        if dimensions.width > 0, dimensions.height > 0 {
            let sideLength = min(dimensions.width, dimensions.height)
            resolution = "\(sideLength)×\(sideLength)"
        } else {
            resolution = "-"
        }

        let flash = preferredFlashMode(for: device, output: output)
            .map(flashDescription) ?? "-"

        let supportedFocusModes: [AVCaptureDevice.FocusMode] = [
            .locked,
            .autoFocus,
            .continuousAutoFocus
        ]
        let focus = supportedFocusModes.contains(where: device.isFocusModeSupported)
            ? focusDescription(device.focusMode)
            : "-"

        // AVFoundation doesn't expose AVCaptureDevice.videoZoomFactor on macOS.
        return CaptureSettings(
            resolution: resolution,
            flash: flash,
            zoom: "-",
            focus: focus
        )
    }

    private static func preferredFlashMode(
        for device: AVCaptureDevice,
        output: AVCapturePhotoOutput
    ) -> AVCaptureDevice.FlashMode? {
        guard device.hasFlash, device.isFlashAvailable else {
            return nil
        }

        if output.supportedFlashModes.contains(.auto) {
            return .auto
        }
        if output.supportedFlashModes.contains(.on) {
            return .on
        }
        if output.supportedFlashModes.contains(.off) {
            return .off
        }
        return nil
    }

    private static func flashDescription(
        _ mode: AVCaptureDevice.FlashMode
    ) -> String {
        switch mode {
        case .off: "Off"
        case .on: "On"
        case .auto: "Auto"
        @unknown default: "-"
        }
    }

    private static func focusDescription(
        _ mode: AVCaptureDevice.FocusMode
    ) -> String {
        switch mode {
        case .locked: "Locked"
        case .autoFocus: "Auto"
        case .continuousAutoFocus: "Continuous"
        @unknown default: "-"
        }
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
    private static let imageContext = CIContext()

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
        if let data = photo.fileDataRepresentation(),
           let image = CIImage(
               data: data,
               options: [.applyOrientationProperty: true]
           ),
           let squareImage = squareImage(from: image) {
            return squareImage
        }

        if let cgImage = photo.cgImageRepresentation(),
           let squareCGImage = squareCGImage(from: cgImage) {
            return makeNSImage(from: squareCGImage)
        }

        if let pixelBuffer = photo.pixelBuffer {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            return squareImage(from: ciImage)
        }

        return nil
    }

    private static func squareImage(from image: CIImage) -> NSImage? {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else { return nil }

        let sideLength = floor(min(extent.width, extent.height))
        let cropRect = CGRect(
            x: floor(extent.midX - sideLength * 0.5),
            y: floor(extent.midY - sideLength * 0.5),
            width: sideLength,
            height: sideLength
        )

        guard let cgImage = imageContext.createCGImage(image, from: cropRect) else {
            return nil
        }
        return makeNSImage(from: cgImage)
    }

    private static func squareCGImage(from image: CGImage) -> CGImage? {
        let sideLength = min(image.width, image.height)
        let cropRect = CGRect(
            x: (image.width - sideLength) / 2,
            y: (image.height - sideLength) / 2,
            width: sideLength,
            height: sideLength
        )
        return image.cropping(to: cropRect)
    }

    private static func makeNSImage(from image: CGImage) -> NSImage {
        NSImage(
            cgImage: image,
            size: NSSize(width: image.width, height: image.height)
        )
    }
}
