import SwiftUI

struct ColonyBoundingBoxOverlay: View {
    let boxes: [ColonyBox]
    let imageSize: CGSize

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, canvasSize in
                guard imageSize.width > 0, imageSize.height > 0 else { return }

                let scale = min(
                    canvasSize.width / imageSize.width,
                    canvasSize.height / imageSize.height
                )
                let renderedSize = CGSize(
                    width: imageSize.width * scale,
                    height: imageSize.height * scale
                )
                let offset = CGPoint(
                    x: (canvasSize.width - renderedSize.width) * 0.5,
                    y: (canvasSize.height - renderedSize.height) * 0.5
                )

                for box in boxes {
                    let rectangle = CGRect(
                        x: offset.x + box.x1 * scale,
                        y: offset.y + box.y1 * scale,
                        width: box.width * scale,
                        height: box.height * scale
                    )
                    context.stroke(
                        Path(rectangle),
                        with: .color(AppTheme.accentGreen),
                        lineWidth: 1.5
                    )
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

#Preview {
    ColonyBoundingBoxOverlay(
        boxes: [
            ColonyBox(x1: 100, y1: 90, x2: 180, y2: 170, score: 0.92),
            ColonyBox(x1: 320, y1: 240, x2: 380, y2: 310, score: 0.87)
        ],
        imageSize: CGSize(width: 640, height: 480)
    )
    .frame(width: 640, height: 480)
    .background(.black)
}
