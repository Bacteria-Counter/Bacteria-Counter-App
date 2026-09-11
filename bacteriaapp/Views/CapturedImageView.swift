import AppKit
import SwiftUI

/// The cropped plate with its detection boxes, zoomable and pannable.
///
/// Zoom exists for a specific reason: the models that matter most here find
/// pinpoint colonies barely a pixel or two across on a 3200px photo, and at
/// fit-to-window there is no way for a microbiologist to judge whether a green
/// box is sitting on a real colony or on a speck of agar. Being able to check
/// that by eye is the whole point.
///
/// The image and the boxes are scaled TOGETHER, as one composed view, rather
/// than each being scaled by its own factor. That is deliberate: a box that
/// drifts off its colony under magnification would be worse than no zoom at
/// all, because it would look like a detection error rather than a drawing
/// one. Composing first makes that drift impossible to introduce.
///
/// When editable, clicking a box selects it and shows a delete button on its
/// corner, and the add mode turns a drag into a new box instead of a pan. The
/// delete button and the box being drawn live OUTSIDE the scaled view, in
/// screen space, so they stay a fixed size at any zoom.
struct CapturedImageView: View {
    let image: NSImage
    var detections: [ColonyDetection] = []
    /// Pixel size of the image the pipeline actually measured `detections`
    /// against (from `AnalysisResult.imageWidth/imageHeight`) — used to scale
    /// detection circles onto wherever this view ends up laying the image
    /// out, independent of `NSImage`'s own reported size.
    var detectionImageSize: CGSize?
    /// Nil while the boxes cannot be edited (still analyzing, or a heatmap).
    var onRemove: ((ColonyDetection.ID) -> Void)?
    var onAdd: ((ColonyDetection) -> Void)?

    @State private var zoom: CGFloat = 1
    @State private var pinch: CGFloat = 1
    @State private var pan: CGSize = .zero
    @State private var drag: CGSize = .zero

    @State private var selectedID: ColonyDetection.ID?
    @State private var isAdding = false
    /// The box being drawn, in screen space.
    @State private var draft: CGRect?

    private static let minZoom: CGFloat = 1
    private static let maxZoom: CGFloat = 12
    private static let step: CGFloat = 1.5
    private static let space = "viewport"
    /// Screen points around a box that still count as clicking it, so a
    /// pinpoint colony's box is not a one-pixel target at fit-to-window.
    private static let hitSlop: CGFloat = 6
    /// A drawn box smaller than this on screen is treated as a stray drag.
    private static let minDraft: CGFloat = 4

    private var isEditable: Bool { onRemove != nil && onAdd != nil }

    var body: some View {
        GeometryReader { geometry in
            let z = clampZoom(zoom * pinch)
            let limit = panLimit(viewport: geometry.size, zoom: z)
            let offset = clampPan(CGSize(width: pan.width + drag.width,
                                         height: pan.height + drag.height),
                                  to: limit)
            let map = Mapping(viewport: geometry.size, source: sourceSize, zoom: z, offset: offset)

            ZStack {
                plate(map: map)
                    .scaleEffect(z)
                    .offset(offset)
                    // Without this the magnified content paints over the
                    // sidebar and the status bar.
                    .clipped()
                    .contentShape(Rectangle())
                    .gesture(dragGesture(limit: limit, map: map))
                    .simultaneousGesture(pinchGesture())
                    .onTapGesture(count: 2) { toggleZoom() }
                    .simultaneousGesture(selectGesture(map: map))

                if isEditable {
                    editOverlay(map: map)
                        .clipped()
                    addControl
                }

                controls(zoom: z)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .coordinateSpace(.named(Self.space))
        }
        // A new photo should not inherit the previous one's magnification.
        .onChange(of: image) { _, _ in
            reset()
            selectedID = nil
            isAdding = false
        }
    }

    // MARK: - Content

    @ViewBuilder
    private func plate(map: Mapping) -> some View {
        let viewport = map.viewport

        ZStack {
            Image(nsImage: image)
                // Nearest-neighbour past 1:1, so a magnified colony shows the
                // pixels the model actually saw instead of a smoothed guess
                // about them. Must come before .frame(): interpolation is a
                // method on Image, not on View.
                .interpolation(map.scale > 1 ? .none : .high)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: viewport.width, height: viewport.height)

            if map.source.width > 0, map.source.height > 0 {
                ForEach(detections) { detection in
                    let selected = detection.id == selectedID
                    Rectangle()
                        .stroke(selected ? Color.red
                                : detection.isManual ? Color.blue : Color.black,
                                lineWidth: (selected ? 3 : 2) / map.zoom)
                        .frame(width: detection.width * map.fit, height: detection.height * map.fit)
                        .position(x: (detection.x + detection.width / 2) * map.fit + map.origin.x,
                                  y: (detection.y + detection.height / 2) * map.fit + map.origin.y)
                }
            }
        }
        .frame(width: viewport.width, height: viewport.height)
    }

    private func editOverlay(map: Mapping) -> some View {
        ZStack {
            if let draft {
                Rectangle()
                    .stroke(AppTheme.accentYellow, style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                    .frame(width: draft.width, height: draft.height)
                    .position(x: draft.midX, y: draft.midY)
                    .allowsHitTesting(false)
            }

            if let selected = detections.first(where: { $0.id == selectedID }) {
                let box = map.toScreen(selected.rect)
                Button {
                    selectedID = nil
                    onRemove?(selected.id)
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 18, height: 18)
                        .background(Color.red, in: Circle())
                }
                .buttonStyle(.plain)
                .help("Remove this detection")
                .position(x: box.maxX, y: box.minY)
            }
        }
        .frame(width: map.viewport.width, height: map.viewport.height)
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

    /// Labelled rather than an icon in the zoom bar: as a bare icon it read as
    /// one more zoom control and went unnoticed.
    private var addControl: some View {
        HStack(spacing: 10) {
            Button {
                isAdding.toggle()
                draft = nil
            } label: {
                Label(isAdding ? "Done Adding" : "Add Colony",
                      systemImage: isAdding ? "checkmark" : "plus.viewfinder")
                    .font(AppTheme.monoSmall)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(isAdding ? AppTheme.accentGreen : .black.opacity(0.55),
                                in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(AppTheme.border))
            }
            .buttonStyle(.plain)
            .foregroundStyle(isAdding ? .black : AppTheme.textPrimary)

            if isAdding {
                Text("Drag a box around a missed colony")
                    .font(AppTheme.monoSmall)
                    .foregroundStyle(AppTheme.textPrimary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
                    .allowsHitTesting(false)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
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

    /// Pans normally; draws a new box while adding.
    private func dragGesture(limit: CGSize, map: Mapping) -> some Gesture {
        DragGesture(coordinateSpace: .named(Self.space))
            .onChanged { value in
                if isAdding {
                    draft = map.clampToImage(CGRect(from: value.startLocation, to: value.location))
                } else {
                    drag = value.translation
                }
            }
            .onEnded { value in
                if isAdding {
                    let box = map.clampToImage(CGRect(from: value.startLocation, to: value.location))
                    draft = nil
                    guard box.width >= Self.minDraft, box.height >= Self.minDraft else { return }
                    let r = map.toPixel(box)
                    onAdd?(ColonyDetection(x: r.minX, y: r.minY, width: r.width, height: r.height,
                                           isManual: true))
                } else {
                    pan = clampPan(CGSize(width: pan.width + drag.width,
                                          height: pan.height + drag.height), to: limit)
                    drag = .zero
                }
            }
    }

    /// Selects the box under the click, or clears the selection on empty agar.
    /// Where boxes overlap, the one whose centre is nearest wins.
    private func selectGesture(map: Mapping) -> some Gesture {
        SpatialTapGesture(coordinateSpace: .named(Self.space))
            .onEnded { value in
                guard isEditable else { return }
                let p = map.toPixel(value.location)
                let slop = Self.hitSlop / map.scale
                selectedID = detections
                    .filter { $0.rect.insetBy(dx: -slop, dy: -slop).contains(p) }
                    .min { $0.rect.distanceSquared(to: p) < $1.rect.distanceSquared(to: p) }?
                    .id
            }
    }

    // MARK: - Geometry

    /// Converts between source pixels and points in the viewport, accounting
    /// for the fit, the zoom about the centre, and the pan offset, in that
    /// order -- the same order the modifiers on `plate` apply them.
    private struct Mapping {
        let viewport: CGSize
        let source: CGSize
        let zoom: CGFloat
        let offset: CGSize

        /// Points per source pixel at fit-to-window.
        var fit: CGFloat {
            guard source.width > 0, source.height > 0 else { return 1 }
            return min(viewport.width / source.width, viewport.height / source.height)
        }

        /// Where source pixel (0, 0) sits before zoom and pan.
        var origin: CGPoint {
            CGPoint(x: (viewport.width - source.width * fit) / 2,
                    y: (viewport.height - source.height * fit) / 2)
        }

        /// Screen points per source pixel.
        var scale: CGFloat { fit * zoom }

        func toScreen(_ p: CGPoint) -> CGPoint {
            CGPoint(x: (origin.x + p.x * fit - viewport.width / 2) * zoom + viewport.width / 2 + offset.width,
                    y: (origin.y + p.y * fit - viewport.height / 2) * zoom + viewport.height / 2 + offset.height)
        }

        func toPixel(_ s: CGPoint) -> CGPoint {
            CGPoint(x: ((s.x - offset.width - viewport.width / 2) / zoom + viewport.width / 2 - origin.x) / fit,
                    y: ((s.y - offset.height - viewport.height / 2) / zoom + viewport.height / 2 - origin.y) / fit)
        }

        func toScreen(_ r: CGRect) -> CGRect {
            CGRect(from: toScreen(CGPoint(x: r.minX, y: r.minY)), to: toScreen(CGPoint(x: r.maxX, y: r.maxY)))
        }

        func toPixel(_ r: CGRect) -> CGRect {
            CGRect(from: toPixel(CGPoint(x: r.minX, y: r.minY)), to: toPixel(CGPoint(x: r.maxX, y: r.maxY)))
        }

        /// Keeps a drawn box on the visible part of the image.
        func clampToImage(_ r: CGRect) -> CGRect {
            let visible = toScreen(CGRect(origin: .zero, size: source))
                .intersection(CGRect(origin: .zero, size: viewport))
            let clamped = r.intersection(visible)
            return clamped.isNull ? .zero : clamped
        }
    }

    /// The pixel size the detections were measured against, which is not
    /// necessarily what NSImage reports for itself.
    private var sourceSize: CGSize {
        if let s = detectionImageSize, s.width > 0, s.height > 0 { return s }
        return image.size
    }

    /// How far the content may be dragged before its edge would come inside
    /// the viewport, leaving a band of empty background.
    private func panLimit(viewport: CGSize, zoom z: CGFloat) -> CGSize {
        let map = Mapping(viewport: viewport, source: sourceSize, zoom: z, offset: .zero)
        let shown = CGSize(width: map.source.width * map.scale, height: map.source.height * map.scale)
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

private extension ColonyDetection {
    var rect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
}

private extension CGRect {
    /// The rectangle spanning two corners given in any order.
    init(from a: CGPoint, to b: CGPoint) {
        self.init(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
    }

    func distanceSquared(to p: CGPoint) -> CGFloat {
        (midX - p.x) * (midX - p.x) + (midY - p.y) * (midY - p.y)
    }
}
