import AppKit
import SwiftUI

/// The captured plate with its detection circles, zoomable and pannable.
///
/// Zoom exists for a specific reason: the models that matter most here find
/// pinpoint colonies barely a pixel or two across on a 3200px photo, and at
/// fit-to-window there is no way for a microbiologist to judge whether a green
/// circle is sitting on a real colony or on a speck of agar. Being able to
/// check that by eye is the whole point.
///
/// The image and the circles are scaled TOGETHER, as one composed view, rather
/// than each being scaled by its own factor. That is deliberate: a circle that
/// drifts off its colony under magnification would be worse than no zoom at
/// all, because it would look like a detection error rather than a drawing
/// one. Composing first makes that drift impossible to introduce.
struct CapturedImageView: View {
    let image: NSImage
    var detections: [ColonyDetection] = []
    /// Pixel size of the image the pipeline actually measured `detections`
    /// against (from `AnalysisResult.imageWidth/imageHeight`) — used to scale
    /// detection circles onto wherever this view ends up laying the image
    /// out, independent of `NSImage`'s own reported size.
    var detectionImageSize: CGSize?

    @State private var zoom: CGFloat = 1
    @State private var pinch: CGFloat = 1
    @State private var pan: CGSize = .zero
    @State private var drag: CGSize = .zero

    private static let minZoom: CGFloat = 1
    private static let maxZoom: CGFloat = 12
    private static let step: CGFloat = 1.5

    var body: some View {
        GeometryReader { geometry in
            let z = clampZoom(zoom * pinch)
            let limit = panLimit(viewport: geometry.size, zoom: z)
            let offset = clampPan(CGSize(width: pan.width + drag.width,
                                         height: pan.height + drag.height),
                                  to: limit)

            ZStack {
                plate(viewport: geometry.size, zoom: z)
                    .scaleEffect(z)
                    .offset(offset)
                    // Without this the magnified content paints over the
                    // sidebar and the status bar.
                    .clipped()
                    .contentShape(Rectangle())
                    .gesture(panGesture(limit: limit))
                    .simultaneousGesture(pinchGesture())
                    .onTapGesture(count: 2) { toggleZoom() }

                controls(zoom: z)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        // A new photo should not inherit the previous one's magnification.
        .onChange(of: image) { _, _ in reset() }
    }

    // MARK: - Content

    @ViewBuilder
    private func plate(viewport: CGSize, zoom z: CGFloat) -> some View {
        let source = sourceSize
        let fit = fitScale(viewport: viewport, source: source)

        ZStack {
            Image(nsImage: image)
                // Nearest-neighbour past 1:1, so a magnified colony shows the
                // pixels the model actually saw instead of a smoothed guess
                // about them. Must come before .frame(): interpolation is a
                // method on Image, not on View.
                .interpolation(z * fit > 1 ? .none : .high)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: viewport.width, height: viewport.height)

            if source.width > 0, source.height > 0 {
                let offsetX = (viewport.width - source.width * fit) / 2
                let offsetY = (viewport.height - source.height * fit) / 2

                ForEach(Array(detections.enumerated()), id: \.offset) { _, detection in
                    let diameter = detection.radius * 2 * fit
                    Circle()
                        // Divided by the zoom so the outline stays a hairline
                        // on screen. At 12x a fixed 2pt stroke becomes 24pt and
                        // swallows the very colonies the zoom was for.
                        .stroke(AppTheme.accentGreen, lineWidth: 2 / z)
                        .frame(width: diameter, height: diameter)
                        .position(x: detection.cx * fit + offsetX,
                                  y: detection.cy * fit + offsetY)
                }
            }
        }
        .frame(width: viewport.width, height: viewport.height)
    }

    private func controls(zoom z: CGFloat) -> some View {
        HStack(spacing: 2) {
            button("minus.magnifyingglass", enabled: z > Self.minZoom) {
                setZoom(z / Self.step)
            }
            Text("\(Int((z * 100).rounded()))%")
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textSecondary)
                .frame(width: 52)
            button("plus.magnifyingglass", enabled: z < Self.maxZoom) {
                setZoom(z * Self.step)
            }
            Divider().frame(height: 16).overlay(AppTheme.border)
            button("arrow.up.left.and.down.right.magnifyingglass", enabled: z > Self.minZoom) {
                reset()
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(AppTheme.border))
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
        // Buttons must not swallow the pan drag underneath them.
        .allowsHitTesting(true)
    }

    private func button(_ icon: String, enabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 12))
                .frame(width: 26, height: 22)
        }
        .buttonStyle(.plain)
        .foregroundStyle(enabled ? AppTheme.textPrimary : AppTheme.textSecondary.opacity(0.4))
        .disabled(!enabled)
    }

    // MARK: - Gestures

    private func pinchGesture() -> some Gesture {
        MagnifyGesture()
            .onChanged { pinch = $0.magnification }
            .onEnded { _ in
                zoom = clampZoom(zoom * pinch)
                pinch = 1
            }
    }

    private func panGesture(limit: CGSize) -> some Gesture {
        DragGesture()
            .onChanged { drag = $0.translation }
            .onEnded { _ in
                pan = clampPan(CGSize(width: pan.width + drag.width,
                                      height: pan.height + drag.height), to: limit)
                drag = .zero
            }
    }

    // MARK: - Geometry

    /// The pixel size the detections were measured against, which is not
    /// necessarily what NSImage reports for itself.
    private var sourceSize: CGSize {
        if let s = detectionImageSize, s.width > 0, s.height > 0 { return s }
        return image.size
    }

    /// Points per source pixel at fit-to-window.
    private func fitScale(viewport: CGSize, source: CGSize) -> CGFloat {
        guard source.width > 0, source.height > 0 else { return 1 }
        return min(viewport.width / source.width, viewport.height / source.height)
    }

    /// How far the content may be dragged before its edge would come inside
    /// the viewport, leaving a band of empty background.
    private func panLimit(viewport: CGSize, zoom z: CGFloat) -> CGSize {
        let source = sourceSize
        let fit = fitScale(viewport: viewport, source: source)
        let shown = CGSize(width: source.width * fit * z, height: source.height * fit * z)
        return CGSize(width: max(0, (shown.width - viewport.width) / 2),
                      height: max(0, (shown.height - viewport.height) / 2))
    }

    private func clampZoom(_ v: CGFloat) -> CGFloat {
        min(max(v, Self.minZoom), Self.maxZoom)
    }

    private func clampPan(_ v: CGSize, to limit: CGSize) -> CGSize {
        CGSize(width: min(max(v.width, -limit.width), limit.width),
               height: min(max(v.height, -limit.height), limit.height))
    }

    private func setZoom(_ v: CGFloat) {
        withAnimation(.easeOut(duration: 0.15)) {
            zoom = clampZoom(v)
            if zoom == Self.minZoom { pan = .zero }
        }
    }

    private func toggleZoom() {
        setZoom(zoom > Self.minZoom ? Self.minZoom : 4)
    }

    private func reset() {
        withAnimation(.easeOut(duration: 0.15)) {
            zoom = 1; pinch = 1; pan = .zero; drag = .zero
        }
    }
}
