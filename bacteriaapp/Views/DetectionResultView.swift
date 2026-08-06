//
//  DetectionResultsView.swift
//  bacteriaapp
//
//  Created by Regina Celine Adiwinata on 06/08/26.
//

import SwiftUI

struct DetectionResultView: View {
    let croppedImage: CGImage
    let detections: [BoundingBox]

    var body: some View {
        GeometryReader { geometry in
            let imageSize = CGSize(width: croppedImage.width, height: croppedImage.height)
            let renderedSize = aspectFitSize(imageSize: imageSize, containerSize: geometry.size)

            ZStack {
                Image(decorative: croppedImage, scale: 1)
                    .resizable()
                    .frame(width: renderedSize.width, height: renderedSize.height)

                ForEach(detections) { detection in
                    let rect = detection.rect(in: renderedSize)
                    RoundedRectangle(cornerRadius: 3)
                        .stroke(AppTheme.accentGreen, lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                }
            }
            .frame(width: renderedSize.width, height: renderedSize.height)
            .position(x: geometry.size.width * 0.5, y: geometry.size.height * 0.5)
        }
        .overlay(alignment: .topTrailing) {
            Label("\(detections.count) DETECTED", systemImage: "circle.grid.3x3")
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.accentGreen)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.black.opacity(0.72))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .padding(12)
        }
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
