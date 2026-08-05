import AVFoundation
import AppKit
import SwiftUI

struct CameraPreviewView: NSViewRepresentable {
    let session: AVCaptureSession?

    func makeNSView(context: Context) -> CameraPreviewNSView {
        let view = CameraPreviewNSView()
        view.session = session
        return view
    }

    func updateNSView(_ nsView: CameraPreviewNSView, context: Context) {
        nsView.session = session
    }
}

final class CameraPreviewNSView: NSView {
    var session: AVCaptureSession? {
        didSet {
            previewLayer.session = session
        }
    }

    private let previewLayer = AVCaptureVideoPreviewLayer()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        previewLayer.videoGravity = .resizeAspectFill
        layer?.addSublayer(previewLayer)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        previewLayer.frame = bounds
    }
}

struct CapturedImageView: View {
    let image: NSImage
    var detections: [ColonyDetection] = []
    /// Pixel size of the image the server actually measured `detections`
    /// against (from `AnalysisResult.imageWidth/imageHeight`) — used to scale
    /// detection circles onto wherever this view ends up laying the image
    /// out, independent of `NSImage`'s own reported size.
    var detectionImageSize: CGSize?

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: geometry.size.width, height: geometry.size.height)

                if let sourceSize = detectionImageSize, sourceSize.width > 0, sourceSize.height > 0 {
                    let scale = min(
                        geometry.size.width / sourceSize.width,
                        geometry.size.height / sourceSize.height
                    )
                    let offsetX = (geometry.size.width - sourceSize.width * scale) / 2
                    let offsetY = (geometry.size.height - sourceSize.height * scale) / 2

                    ForEach(Array(detections.enumerated()), id: \.offset) { _, detection in
                        let diameter = detection.radius * 2 * scale
                        Circle()
                            .stroke(AppTheme.accentGreen, lineWidth: 2)
                            .frame(width: diameter, height: diameter)
                            .position(
                                x: detection.cx * scale + offsetX,
                                y: detection.cy * scale + offsetY
                            )
                    }
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .clipped()
        }
    }
}
