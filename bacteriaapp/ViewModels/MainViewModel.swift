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
    /// The model's boxes after the user's corrections. `analysisResult` keeps
    /// what the model itself returned.
    @Published private(set) var detections: [ColonyDetection] = []
    @Published private(set) var captureSettings = CaptureSettings.unavailable
    @Published var connectionError: String?
    @Published private(set) var selectedModel: ModelChoice = .v26s_new

    @Published private(set) var preparedImage: NSImage?
    @Published private(set) var usedFullFrame = false

    @Published private(set) var pendingCropImage: NSImage?
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
            let base = "Complete · \(result.totalColonies) colonies · \(result.averageConfidence)% average confidence "
                + "· \(result.modelUsed.fullDisplayName)"
            return result.usedFullFrame ? base + " · cawan tidak terdeteksi, foto utuh dipakai" : base
        case .cropping:
            return "Adjust the petri dish cropping area"
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
                capturedImage = Self.squareCropped(image)
                startAnalysis(freshPhoto: true)
            } catch {
                if case CameraService.CameraError.notConnected = error {
                    handleCameraConnectionLost()
                } else {
                    connectionError = error.localizedDescription
                }
            }
        }
    }

    func uploadImage() {
        guard appState != .analyzing, appState != .cropping else { return }
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

        pendingCropImage = image
        appState = .cropping
    }

    func selectModel(_ model: ModelChoice) {
        guard selectedModel != model else { return }
        selectedModel = model
        if capturedImage != nil, appState != .analyzing {
            startAnalysis(freshPhoto: false)
        }
    }

    func confirmCrop(_ cropped: NSImage) {
        guard appState == .cropping else { return }
        pendingCropImage = nil
        capturedImage = cropped
        startAnalysis(freshPhoto: true)
    }

    func cancelCrop() {
        guard appState == .cropping else { return }
        pendingCropImage = nil
        appState = isDeviceConnected ? .connected : .disconnected
    }

    func newCapture() {
        analysisTask?.cancel()
        analysisTask = nil
        clearAnalysis()
        capturedImage = nil
        pendingCropImage = nil
        appState = isDeviceConnected ? .connected : .disconnected
    }

    func removeDetection(_ id: ColonyDetection.ID) {
        guard appState == .complete,
              let index = detections.firstIndex(where: { $0.id == id }) else { return }
        detections.remove(at: index)
        colonyCount -= 1
    }

    func addDetection(_ detection: ColonyDetection) {
        guard appState == .complete else { return }
        detections.append(detection)
        colonyCount += 1
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
        pendingCropImage = nil
        clearAnalysis()
        appState = .disconnected
    }
    
    private func clearAnalysis() {
        preparedImage = nil
        preparedCGImage = nil
        usedFullFrame = false
        colonyCount = 0
        analysisProgress = 0
        analysisResult = nil
        detections = []
    }

    private func startAnalysis(freshPhoto: Bool) {
        guard let photo = capturedImage else { return }

        analysisTask?.cancel()
        appState = .analyzing
        
        connectionError = nil
        colonyCount = 0
        analysisProgress = 0
        analysisResult = nil
        detections = []
        if freshPhoto {
            preparedImage = nil
            preparedCGImage = nil
            usedFullFrame = false
        }

        let model = selectedModel

        analysisTask = Task {
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
                detections = result.detections
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
    
    private static func squareCropped(_ image: NSImage) -> NSImage {
        guard let cg = cgImage(from: image) else { return image }
        let cropped = squared(cg)
        return NSImage(cgImage: cropped, size: NSSize(width: cropped.width, height: cropped.height))
    }

    nonisolated private static func squared(_ image: CGImage) -> CGImage {
        let side = min(image.width, image.height)
        let rect = CGRect(x: (image.width - side) / 2, y: (image.height - side) / 2,
                          width: side, height: side)
        return image.cropping(to: rect) ?? image
    }

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

    nonisolated private static func log(_ stage: String, since: Date,
                                        pixels: Int, extra: String) {
        print(String(format: "[AgarScope] %@ %.2fs %.1f MP %@",
                     stage, Date().timeIntervalSince(since),
                     Double(pixels) / 1_000_000, extra))
    }

    private static func cgImage(from image: NSImage) -> CGImage? {
        let pixels = image.representations.reduce(into: CGSize.zero) { size, rep in
            size.width = max(size.width, CGFloat(rep.pixelsWide))
            size.height = max(size.height, CGFloat(rep.pixelsHigh))
        }
        var rect = CGRect(origin: .zero,
                          size: pixels.width > 0 && pixels.height > 0 ? pixels : image.size)
        return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }
}
