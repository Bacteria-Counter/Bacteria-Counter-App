import AVFoundation
import AppKit
import SwiftUI

struct CameraPreviewView: View {
    let session: AVCaptureSession?

    var body: some View {
        GeometryReader { geometry in
            let sideLength = min(geometry.size.width, geometry.size.height)

            CameraPreviewRepresentable(session: session)
                .frame(width: sideLength, height: sideLength)
                .clipped()
                .position(
                    x: geometry.size.width * 0.5,
                    y: geometry.size.height * 0.5
                )
        }
        .accessibilityLabel("Square camera preview")
    }
}

private struct CameraPreviewRepresentable: NSViewRepresentable {
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
    let segmentationMask: CGImage?
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geometry in
            let renderedSize = aspectFitSize(
                imageSize: imageSize,
                containerSize: geometry.size
            )

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

                if let segmentationMask {
                    Color.black
                        .opacity(0.55)
                        .frame(
                            width: renderedSize.width,
                            height: renderedSize.height
                        )

                    Image(nsImage: image)
                        .resizable()
                        .frame(
                            width: renderedSize.width,
                            height: renderedSize.height
                        )
                        .mask {
                            maskImage(
                                segmentationMask,
                                size: renderedSize
                            )
                        }

                    AppTheme.accentGreen
                        .opacity(0.28)
                        .frame(
                            width: renderedSize.width,
                            height: renderedSize.height
                        )
                        .mask {
                            maskImage(
                                segmentationMask,
                                size: renderedSize
                            )
                        }

                    Image(decorative: segmentationMask, scale: 1)
                        .resizable()
                        .interpolation(.none)
                        .frame(
                            width: renderedSize.width,
                            height: renderedSize.height
                        )
                        .colorMultiply(AppTheme.accentGreen)
                        .opacity(0.55)
                        .blendMode(.screen)
                        .accessibilityHidden(true)
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .clipped()
            .overlay(alignment: .topTrailing) {
                if segmentationMask != nil {
                    Label("SEGMENTED DISH", systemImage: "viewfinder.circle")
                        .font(AppTheme.monoSmall)
                        .foregroundStyle(AppTheme.accentGreen)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(.black.opacity(0.72))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .padding(12)
                }
            }
        }
    }

    private func maskImage(
        _ mask: CGImage,
        size: CGSize
    ) -> some View {
        Image(decorative: mask, scale: 1)
            .resizable()
            .interpolation(.none)
            .frame(width: size.width, height: size.height)
            .luminanceToAlpha()
    }

    private func aspectFitSize(
        imageSize: CGSize,
        containerSize: CGSize
    ) -> CGSize {
        guard imageSize.width > 0,
              imageSize.height > 0,
              containerSize.width > 0,
              containerSize.height > 0 else {
            return .zero
        }

        let scale = min(
            containerSize.width / imageSize.width,
            containerSize.height / imageSize.height
        )
        return CGSize(
            width: imageSize.width * scale,
            height: imageSize.height * scale
        )
    }
}
