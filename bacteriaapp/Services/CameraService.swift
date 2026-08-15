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
    private var notificationObservers: [NSObjectProtocol] = []
    private var shouldRetryContinuityCamera = false

    private let continuityCameraRetryCount = 5
    private let continuityCameraRetryDelay: UInt64 = 400_000_000

    var onConnectionLost: (() -> Void)?

    override init() {
        super.init()

        let notificationCenter = NotificationCenter.default
        notificationObservers.append(
            notificationCenter.addObserver(
                forName: AVCaptureDevice.wasDisconnectedNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                let disconnectedDeviceID = (notification.object as? AVCaptureDevice)?.uniqueID
                Task { @MainActor [weak self, disconnectedDeviceID] in
                    self?.handleDeviceDisconnected(deviceID: disconnectedDeviceID)
                }
            }
        )
        notificationObservers.append(
            notificationCenter.addObserver(
                forName: AVCaptureSession.runtimeErrorNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                let sessionID = (notification.object as? AVCaptureSession).map(ObjectIdentifier.init)
                Task { @MainActor [weak self, sessionID] in
                    self?.handleSessionFailure(sessionID: sessionID)
                }
            }
        )
        notificationObservers.append(
            notificationCenter.addObserver(
                forName: AVCaptureSession.wasInterruptedNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                let sessionID = (notification.object as? AVCaptureSession).map(ObjectIdentifier.init)
                Task { @MainActor [weak self, sessionID] in
                    self?.handleSessionFailure(sessionID: sessionID)
                }
            }
        )
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
                guard session.isRunning else {
                    clearSession()
                    return try await startSession()
                }
                isSessionRunning = true
                shouldRetryContinuityCamera = false
            }
            return
        }

        let retryContinuityCamera = shouldRetryContinuityCamera
        var lastError: Error?

        // After a Continuity Camera disconnects, macOS can take a short time
        // to publish the iPhone device again. Refresh discovery for each retry
        // before falling back to the Mac camera.
        let continuityAttempts = retryContinuityCamera ? continuityCameraRetryCount : 1
        for attempt in 0..<continuityAttempts {
            guard let candidate = cameraCandidates().first(where: { $0.isContinuityCamera }) else {
                if !retryContinuityCamera {
                    break
                }

                if attempt + 1 < continuityAttempts {
                    try await Task.sleep(nanoseconds: continuityCameraRetryDelay)
                }
                continue
            }

            do {
                try await startSession(with: candidate.device, type: candidate.type)
                shouldRetryContinuityCamera = false
                return
            } catch {
                lastError = error
                clearSession()
            }

            if attempt + 1 < continuityAttempts {
                try await Task.sleep(nanoseconds: continuityCameraRetryDelay)
            }
        }

        // Keep the MacBook camera as the final fallback. Discover it again so
        // this candidate is also a fresh AVCaptureDevice instance.
        guard let fallback = cameraCandidates().first(where: { !$0.isContinuityCamera }) else {
            throw lastError ?? CameraError.noDeviceFound
        }

        do {
            try await startSession(with: fallback.device, type: fallback.type)
            shouldRetryContinuityCamera = false
            return
        } catch {
            lastError = error
            clearSession()
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

    private struct CameraCandidate {
        let device: AVCaptureDevice
        let type: String
        let isContinuityCamera: Bool
    }

    private func cameraCandidates() -> [CameraCandidate] {
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.continuityCamera, .external, .builtInWideAngleCamera],
            mediaType: .video,
            position: .unspecified
        )

        // isContinuityCamera remains reliable even if an older configuration
        // reports the iPhone using a generic or built-in device type.
        let iPhone = discovery.devices.first(where: {
            isContinuityCameraCandidate($0)
        })

        let builtIn = discovery.devices.first(where: {
            !isContinuityCameraCandidate($0) && $0.deviceType == .builtInWideAngleCamera
        })

        return [
            iPhone.map { CameraCandidate(device: $0, type: "Continuity Camera", isContinuityCamera: true) },
            builtIn.map { CameraCandidate(device: $0, type: "Mac built-in camera", isContinuityCamera: false) }
        ].compactMap { $0 }
    }

    private func isContinuityCameraCandidate(_ device: AVCaptureDevice) -> Bool {
        device.isContinuityCamera
            || device.deviceType == .continuityCamera
            || device.localizedName.localizedCaseInsensitiveContains("iPhone")
    }

    func stopSession() {
        let session = captureSession
        sessionQueue.async {
            session?.stopRunning()
        }
        isSessionRunning = false
    }

    private func clearSession() {
        let session = captureSession
        sessionQueue.sync {
            session?.stopRunning()
        }
        clearSessionReferences()
    }

    private func clearSessionReferences() {
        captureSession = nil
        photoOutput = nil
        currentDevice = nil
        connectedDeviceName = nil
        connectedDeviceType = nil
        previewSession = nil
        captureSettings = .unavailable
        isSessionRunning = false
    }

    private func handleDeviceDisconnected(deviceID: String?) {
        guard let deviceID,
              let currentDevice,
              deviceID == currentDevice.uniqueID else {
            return
        }

        handleConnectionLost()
    }

    private func handleSessionFailure(sessionID: ObjectIdentifier?) {
        guard let sessionID,
              let captureSession,
              ObjectIdentifier(captureSession) == sessionID else {
            return
        }

        handleConnectionLost()
    }

    private func handleConnectionLost() {
        guard captureSession != nil || photoOutput != nil || connectedDeviceName != nil else {
            return
        }

        shouldRetryContinuityCamera = true

        let session = captureSession
        let delegate = inFlightCaptureDelegate
        inFlightCaptureDelegate = nil

        delegate?.cancel(with: CameraError.notConnected)

        sessionQueue.sync {
            session?.stopRunning()
        }
        clearSessionReferences()
        onConnectionLost?()
    }

    func capturePhoto() async throws -> NSImage {
        guard let photoOutput,
              let captureSession else {
            throw CameraError.notConnected
        }

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
                let hasActiveVideoConnection = photoOutput.connections.contains(where: {
                    $0.isEnabled && $0.isActive
                })

                guard captureSession.isRunning, hasActiveVideoConnection else {
                    delegate.cancel(with: CameraError.notConnected)
                    Task { @MainActor [weak self] in
                        self?.handleConnectionLost()
                    }
                    return
                }

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

    func cancel(with error: Error) {
        guard !didComplete else { return }

        didComplete = true
        completion(.failure(error))
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
