import AppKit
import Combine
import CoreGraphics
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
    @Published private(set) var selectedModel: ModelChoice = .samMicro

    /// The dish crop every model is measured against, and what the viewport
    /// shows. Cropping is done once per photo and shared: it is the one
    /// preprocessing step both pipelines agree on, and running it per model
    /// would put the 84 MB segmentation model through six passes for one plate.
    @Published private(set) var preparedImage: NSImage?
    /// Set when segmentation could not find a dish and the AgarScope models are
    /// working from the uncropped photo instead. The lab YOLO models were
    /// trained on cropped plates and have no equivalent fallback, so they refuse.
    @Published private(set) var usedFullFrame = false

    /// Which stage is running, shown in the status bar while analyzing.
    ///
    /// Worth the two lines: a plate that takes eight seconds and a plate that
    /// has hung look identical when the only feedback is a progress bar that
    /// invents its own percentage. Each stage also prints its own duration and
    /// the pixel count it worked on, so a slow run can be reported as numbers
    /// rather than as an impression.
    @Published private(set) var analysisStage: String?

    let cameraService = CameraService()
    private let inferenceService = InferenceService()
    private let segmenter = PetriDishSegmenter()

    private var preparedCGImage: CGImage?
    private var analysisTask: Task<Void, Never>?

    init() {
        cameraService.onConnectionLost = { [weak self] in
            self?.handleCameraConnectionLost()
        }
    }

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
            return "\(analysisStage ?? "Analyzing") · \(Int(analysisProgress * 100))%"
        case .complete:
            guard let result = analysisResult else { return "Complete" }
            let base = "Complete · \(result.totalColonies) colonies · avg conf "
                + "\(result.averageConfidence)% · \(result.modelUsed.fullDisplayName)"
            return result.usedFullFrame ? base + " · cawan tidak terdeteksi, foto utuh dipakai" : base
        }
    }

    var headerStatusText: String {
        deviceName ?? "No Device"
    }

    var isDeviceConnected: Bool {
        deviceName != nil
    }

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
                startAnalysis(freshPhoto: true)
            } catch {
                // A capture that fails because the iPhone went away is the
                // disconnect case, not an analysis error -- it has to reset the
                // UI rather than leave a message next to a dead preview.
                if case CameraService.CameraError.notConnected = error {
                    handleCameraConnectionLost()
                } else {
                    connectionError = error.localizedDescription
                }
            }
        }
    }

    /// Lets the user analyze a plate photo from disk instead of the camera --
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
        startAnalysis(freshPhoto: true)
    }

    /// Called by the sidebar picker. Switching models after a plate has been
    /// analyzed re-runs on the SAME crop with the new model -- otherwise the
    /// displayed count would silently stay from whichever model ran last, which
    /// reads as "the picker doesn't do anything". The crop is not redone: it
    /// does not depend on the model, and redoing it would make two models
    /// disagree for a reason that has nothing to do with either.
    func selectModel(_ model: ModelChoice) {
        guard selectedModel != model else { return }
        selectedModel = model
        if capturedImage != nil, appState != .analyzing {
            startAnalysis(freshPhoto: false)
        }
    }

    func newCapture() {
        analysisTask?.cancel()
        analysisTask = nil
        clearAnalysis()
        capturedImage = nil
        appState = isDeviceConnected ? .connected : .disconnected
    }

    /// Everything derived from a photo. Both the crop and the result have to go:
    /// leaving the crop behind would silently analyze the previous plate, and
    /// leaving the result behind would draw the previous plate's boxes over the
    /// new one.
    private func clearAnalysis() {
        preparedImage = nil
        preparedCGImage = nil
        usedFullFrame = false
        colonyCount = 0
        analysisProgress = 0
        analysisResult = nil
    }

    private func handleCameraConnectionLost() {
        analysisTask?.cancel()
        analysisTask = nil

        isConnecting = false
        isCapturing = false
        connectionError = nil

        deviceName = nil
        deviceType = nil
        captureSettings = .unavailable

        capturedImage = nil
        clearAnalysis()
        appState = .disconnected
    }

    private func startAnalysis(freshPhoto: Bool) {
        guard let photo = capturedImage else { return }

        analysisTask?.cancel()
        appState = .analyzing
        // Cleared here rather than only at capture time, so switching from a
        // lab model that refused an uncropped photo to one that can count it
        // does not leave the refusal on screen next to a fresh result.
        connectionError = nil
        colonyCount = 0
        analysisProgress = 0
        analysisResult = nil
        if freshPhoto {
            preparedImage = nil
            preparedCGImage = nil
            usedFullFrame = false
        }

        let model = selectedModel

        analysisTask = Task {
            // Indeterminate progress while the pipeline runs. There is no
            // meaningful intermediate percentage to report -- it is a crop plus
            // one Core ML pass plus filtering -- so this is visual feedback, not
            // a measurement of how far along it is.
            let progressTask = Task {
                while !Task.isCancelled && analysisProgress < 0.9 {
                    analysisProgress += 0.03
                    try? await Task.sleep(for: .milliseconds(120))
                }
            }
            defer { progressTask.cancel() }

            do {
                let prepared = try await prepareIfNeeded(photo, for: model)
                guard !Task.isCancelled else { return }

                if usedFullFrame, model.engine == .labYOLO {
                    throw PreparationError.cropRequired
                }

                analysisStage = "Menghitung koloni"
                let t = Date()
                let result = try await inferenceService.analyze(
                    image: prepared, model: model, usedFullFrame: usedFullFrame
                )
                Self.log("hitung \(model.rawValue)", since: t,
                         pixels: prepared.width * prepared.height,
                         extra: "\(result.totalColonies) koloni")
                guard !Task.isCancelled else { return }

                colonyCount = result.totalColonies
                analysisProgress = 1.0
                analysisResult = result
                analysisStage = nil
                appState = .complete
            } catch {
                guard !Task.isCancelled else { return }
                connectionError = error.localizedDescription
                analysisProgress = 0
                analysisStage = nil
                appState = isDeviceConnected ? .connected : .disconnected
            }
        }
    }

    /// The image this model counts on, cached per photo.
    ///
    /// FastSAM takes the photo whole -- no segmentation, no crop -- which is the
    /// path it ran on before the merge and the one every figure for it was
    /// measured against. Everything else gets the dish cropped out first, and if
    /// segmentation cannot find a dish the AgarScope models carry on with the
    /// full frame while the lab models refuse, since they were trained on crops.
    ///
    /// Whatever is returned is also what the viewport shows, so the boxes always
    /// sit on the picture the model actually looked at.
    private func prepareIfNeeded(_ photo: NSImage, for model: ModelChoice) async throws -> CGImage {
        guard let full = Self.cgImage(from: photo) else {
            throw PreparationError.imageUnreadable
        }

        guard model.usesCrop else {
            usedFullFrame = false
            return show(Self.capped(full))
        }

        if let cached = preparedCGImage { return show(cached) }

        analysisStage = "Mencari cawan"
        let tSeg = Date()
        let mask = try? await segmenter.makeMask(from: full)
        Self.log("segmentasi", since: tSeg, pixels: full.width * full.height,
                 extra: mask == nil ? "GAGAL" : "ok")

        // Cropping draws a canvas the size of the dish, and on a big capture
        // that is tens of megapixels of work. Off the main actor so the window
        // keeps repainting.
        analysisStage = "Memotong cawan"
        let tCrop = Date()
        let (prepared, fellBack) = await Task.detached(priority: .userInitiated) {
            if let mask, let cropped = try? DishCropper.crop(image: full, mask: mask.mask).image {
                return (Self.capped(cropped), false)
            }
            return (Self.capped(full), true)
        }.value
        Self.log("potong", since: tCrop, pixels: prepared.width * prepared.height,
                 extra: fellBack ? "foto utuh" : "\(prepared.width)x\(prepared.height)")

        preparedCGImage = prepared
        usedFullFrame = fellBack
        return show(prepared)
    }

    @discardableResult
    private func show(_ image: CGImage) -> CGImage {
        preparedImage = NSImage(cgImage: image,
                                size: NSSize(width: image.width, height: image.height))
        return image
    }

    /// The largest input any model here consumes is 4480 -- sam_micro's
    /// escalation pass, and every other size is below it. Pixels beyond that
    /// are letterboxed away before inference, but they are NOT free: the mask
    /// post-processing allocates one buffer per colony at the input's own
    /// resolution, so cost grows with colonies times pixels. On a 48 MP photo
    /// that is 1.45 GB of allocation for detections the model never saw at that
    /// resolution anyway, and it measured 20.5 s against 1.8 s for a 12 MP photo
    /// of the same plate.
    ///
    /// The area filters are fractions of the image area and circularity is
    /// scale-free, so nothing downstream reads absolute pixels.
    /// nonisolated: this runs inside the detached task above, and a MainActor
    /// method would have to hop back to the main thread to do it -- which is the
    /// hop the detached task exists to avoid.
    nonisolated private static func capped(_ image: CGImage, longSide: Int = 4480) -> CGImage {
        let side = max(image.width, image.height)
        guard side > longSide else { return image }

        let scale = Double(longSide) / Double(side)
        let w = Int((Double(image.width) * scale).rounded())
        let h = Int((Double(image.height) * scale).rounded())
        guard let context = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
            space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return image }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return context.makeImage() ?? image
    }

    private enum PreparationError: LocalizedError {
        case imageUnreadable
        case cropRequired

        var errorDescription: String? {
            switch self {
            case .imageUnreadable:
                "Foto tidak bisa dibaca."
            case .cropRequired:
                "Cawan tidak terdeteksi di foto ini. Model lab (YOLOv11s, YOLOv26s, YOLOv26n) "
                    + "dilatih memakai gambar yang sudah dipotong dan tidak bisa dipakai tanpa "
                    + "potongan itu. Pilih SAM, Mac1, atau CSRNet, atau ambil ulang fotonya."
            }
        }
    }

    /// One line per stage in the Xcode console. Kept to stdout rather than a
    /// logging framework so it can be copied out of a run and pasted into a bug
    /// report without any setup.
    nonisolated private static func log(_ stage: String, since: Date,
                                        pixels: Int, extra: String) {
        print(String(format: "[AgarScope] %@ %.2fs %.1f MP %@",
                     stage, Date().timeIntervalSince(since),
                     Double(pixels) / 1_000_000, extra))
    }

    /// The photo at its FULL pixel size.
    ///
    /// `NSImage.size` is in points, not pixels, and a photo carrying DPI
    /// metadata reports far fewer points than it has pixels -- a capture from
    /// this app arrived as 0.2 MP that way. Passing that size to
    /// `cgImage(forProposedRect:)` does not just mislabel the image, it returns
    /// a genuinely downscaled one, so every model has been counting colonies on
    /// a few hundred pixels of plate. The representations know the real pixel
    /// dimensions even when the NSImage does not, so ask them.
    private static func cgImage(from image: NSImage) -> CGImage? {
        let pixels = image.representations.reduce(into: CGSize.zero) { size, rep in
            size.width = max(size.width, CGFloat(rep.pixelsWide))
            size.height = max(size.height, CGFloat(rep.pixelsHigh))
        }
        var rect = CGRect(origin: .zero,
                          size: pixels.width > 0 && pixels.height > 0 ? pixels : image.size)
        return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }

    func exportReport() {
        // Placeholder for export functionality
    }
}
