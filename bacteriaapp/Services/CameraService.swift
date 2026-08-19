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
                // Without this, a failed iPhone/Continuity Camera attempt falls
                // back to the Mac's built-in camera silently -- there was no way
                // to tell afterward whether the iPhone was never detected or was
                // detected but failed to start.
                print("[CameraService] \(candidate.type) (\(candidate.device.localizedName)) failed to start: \(error)")
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

        if #available(macOS 12.3, *) {
            AVCaptureDevice.centerStageControlMode = .cooperative
            AVCaptureDevice.isCenterStageEnabled = false
        }
        

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

        // Printed so a camera that is labelled wrongly can be read off the
        // system's own answers. isContinuityCamera and deviceType come from
        // AVFoundation; the name match is this app's own fallback and is the
        // part most likely to misfire, since a Mac named after its owner
        // produces a built-in camera called "<Owner> Camera".
        for d in discovery.devices {
            print("[AgarScope] kamera: \"\(d.localizedName)\" | tipe \(d.deviceType.rawValue) "
                  + "| isContinuityCamera \(d.isContinuityCamera) "
                  + "| dianggap iPhone: \(isContinuityCameraCandidate(d))")
        }

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

        // photoOutput.maxPhotoDimensions was only set once, at session-start
        // time, from whatever format was active then. Continuity Camera can
        // renegotiate to a higher-resolution format afterward, so the
        // settings computed below (from the device's CURRENT active format)
        // can end up larger than this stale ceiling -- refresh it here too,
        // right before every capture, so output and settings always agree.
        if let currentDevice, let dimensions = Self.maximumPhotoDimensions(for: currentDevice.activeFormat) {
            photoOutput.maxPhotoDimensions = dimensions
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

        // Re-derive from the device's CURRENT active format rather than
        // trusting output.maxPhotoDimensions -- the active format can
        // change after the session starts (Continuity Camera renegotiates
        // formats), and a stale value here crashes
        // capturePhoto(with:delegate:) since it must match one of the
        // *current* format's supportedMaxPhotoDimensions.
        if let device, let dimensions = maximumPhotoDimensions(for: device.activeFormat) {
            settings.maxPhotoDimensions = dimensions
        }

        if let device, let flashMode = preferredFlashMode(for: device, output: output) {
            settings.flashMode = flashMode
        }

        return settings
    }

    /// The smallest square frame worth capturing, in pixels.
    ///
    /// Every camera frame is squared before counting, so what matters is the
    /// SHORT side: that becomes the side of the square the dish sits in. The
    /// dish crops in the benchmark run 1693 to 1863 pixels across, and the
    /// pipeline consumes up to 4480, so 2400 leaves headroom above the former
    /// without chasing the latter.
    private static let minimumSquareSide: Int32 = 2400

    private static func maximumPhotoDimensions(
        for format: AVCaptureDevice.Format
    ) -> CMVideoDimensions? {
        let supported = format.supportedMaxPhotoDimensions
            .filter { $0.width > 0 && $0.height > 0 }
        guard !supported.isEmpty else { return nil }

        let area: (CMVideoDimensions) -> Int64 = { Int64($0.width) * Int64($0.height) }
        let shortSide: (CMVideoDimensions) -> Int32 = { min($0.width, $0.height) }

        // The smallest option that still gives a usable square. Smallest, not
        // largest, because pixels beyond what the pipeline consumes cost time
        // and nothing else -- but the floor comes first, since a capture too
        // small to resolve colonies cannot be recovered later.
        //
        // An earlier version of this chose the largest option under a ceiling on
        // the LONG side, which reads sensibly and behaves badly: on a phone
        // offering 1920x1080 and 8064x6048 and nothing between, the ceiling
        // excluded the large option and the app captured at 2 MP. Selecting on
        // the short side against a floor cannot fail that way.
        if let best = supported
            .filter({ shortSide($0) >= minimumSquareSide })
            .min(by: { area($0) < area($1) }) {
            Self.logChoice(supported, best)
            return best
        }

        // Nothing reaches the floor: take the largest there is.
        let best = supported.max(by: { area($0) < area($1) })
        if let best { Self.logChoice(supported, best) }
        return best
    }

    /// Printed once per session so a capture that comes out unexpectedly small
    /// can be read off the camera's own list rather than guessed at.
    private static func logChoice(_ supported: [CMVideoDimensions], _ chosen: CMVideoDimensions) {
        let all = supported
            .sorted { Int64($0.width) * Int64($0.height) < Int64($1.width) * Int64($1.height) }
            .map { "\($0.width)x\($0.height)" }
            .joined(separator: ", ")
        print("[AgarScope] format menawarkan: \(all) | dipilih: \(chosen.width)x\(chosen.height)")
    }

    private static func captureSettings(
        for device: AVCaptureDevice,
        output: AVCapturePhotoOutput
    ) -> CaptureSettings {
        guard device.isContinuityCamera else {
            return .unavailable
        }

        // Derived from the device's CURRENT active format, not from
        // output.maxPhotoDimensions -- that property was assigned before
        // startRunning(), while activeFormat was still the pre-session
        // default, so reading it back reported a low resolution (640x480)
        // that never matched what capture actually produced. Capture itself
        // was always correct: makePhotoSettings() re-derives from
        // activeFormat at capture time. This was a display bug only.
        // Reported as the SQUARE that is kept, not as the sensor's own frame.
        // A plate is round, so the app squares every camera frame before
        // counting and the sides never reach a model. Printing 1920x1080 next
        // to a square preview would describe pixels the technician cannot use.
        let dimensions = maximumPhotoDimensions(for: device.activeFormat)
        let resolution = if let dimensions, dimensions.width > 0, dimensions.height > 0 {
            "\(min(dimensions.width, dimensions.height))×\(min(dimensions.width, dimensions.height))"
        } else {
            "-"
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

// AVCapturePhotoOutput invokes these delegate methods on its own internal
// queue (here, sessionQueue), never guaranteed to be the main actor -- this
// project defaults every type to @MainActor isolation
// (SWIFT_DEFAULT_ACTOR_ISOLATION), so this needs an explicit opt-out or
// calling the delegate crashes under Swift 6's strict concurrency checking.
// @unchecked Sendable: AVFoundation guarantees these delegate methods fire
// serially for a single capture, so mutable `didComplete` is never touched
// concurrently even though the compiler can't verify that on its own.
nonisolated final class PhotoCaptureDelegate: NSObject, AVCapturePhotoCaptureDelegate, @unchecked Sendable {
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

    func cancel(with error: Error) {
        guard !didComplete else { return }

        didComplete = true
        completion(.failure(error))
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
