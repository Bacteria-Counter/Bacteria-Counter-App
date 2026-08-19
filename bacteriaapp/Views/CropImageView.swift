//
//  CropImageView.swift
//  bacteriaapp
//
//  Created by Regina Celine Adiwinata on 18/08/26.
//

import AppKit
import SwiftUI

/// Ditampilkan begitu foto diupload, sebelum analisis jalan. User menggeser
/// kotak persegi ke area cawan -- beda dari frame kamera yang otomatis
/// di-square saat capture, foto upload bisa punya rasio dan framing apa
/// saja, jadi tidak ada crop otomatis yang aman untuk diasumsikan di sini.
struct CropImageView: View {
    let image: NSImage
    var onConfirm: (NSImage) -> Void
    var onCancel: () -> Void

    private enum Corner: CaseIterable {
        case topLeft, topRight, bottomLeft, bottomRight
    }

    @State private var box: CGRect = .zero
    @State private var dragOffset: CGSize = .zero
    @State private var activeCorner: Corner?
    @State private var resizeTranslation: CGSize = .zero
    @State private var imageFrame: CGRect = .zero

    private static let handleSize: CGFloat = 22
    private static let minBoxSide: CGFloat = 60

    var body: some View {
        GeometryReader { geometry in
            let source = image.pixelSize
            let fit = fitScale(viewport: geometry.size, source: source)
            let frame = CGRect(
                x: (geometry.size.width - source.width * fit) / 2,
                y: (geometry.size.height - source.height * fit) / 2,
                width: source.width * fit,
                height: source.height * fit
            )

            ZStack {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: geometry.size.width, height: geometry.size.height)

                dimOverlay(imageFrame: frame)
                cropBox(imageFrame: frame)

                VStack {
                    instructions
                    Spacer()
                    controls
                }
                .padding(20)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .onAppear { syncFrame(frame) }
            .onChange(of: geometry.size) { _, _ in syncFrame(frame) }
            .onChange(of: image) { _, _ in syncFrame(frame, forceReset: true) }
        }
    }

    /// Menjaga `imageFrame` tetap update, dan menaruh kotak di tengah saat
    /// pertama kali frame tersedia atau saat foto diganti.
    private func syncFrame(_ frame: CGRect, forceReset: Bool = false) {
        let firstFrame = imageFrame.width == 0
        imageFrame = frame
        guard frame.width > 0, frame.height > 0 else { return }
        box = (firstFrame || forceReset) ? Self.initialBox(in: frame) : clamp(box, to: frame)
    }

    // MARK: - Copy

    private var instructions: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Sesuaikan area potong")
                .font(AppTheme.monoSmall)
                .foregroundStyle(AppTheme.textPrimary)
            Text("Geser dan ubah ukuran kotak agar pas 1:1 dengan cawan petri, "
                 + "lalu konfirmasi untuk mulai menghitung koloni.")
            .font(AppTheme.monoFont)
                .foregroundStyle(AppTheme.textSecondary)
        }
        .padding(12)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
        .frame(maxWidth: 360, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button("Batal", action: onCancel)
                .buttonStyle(.plain)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
                .foregroundStyle(AppTheme.textPrimary)

            Button("Konfirmasi Potongan", action: confirm)
                .buttonStyle(.plain)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(AppTheme.accentGreen, in: RoundedRectangle(cornerRadius: 8))
                .foregroundStyle(.black)
        }
        .frame(maxWidth: .infinity, alignment: .trailing)
    }

    // MARK: - Gambar kotak

    private func dimOverlay(imageFrame: CGRect) -> some View {
        let displayed = displayedBox(in: imageFrame)
        return Path { path in
            path.addRect(imageFrame)
            path.addRect(displayed)
        }
        .fill(.black.opacity(0.5), style: FillStyle(eoFill: true))
        .allowsHitTesting(false)
    }

    private func cropBox(imageFrame: CGRect) -> some View {
        let displayed = displayedBox(in: imageFrame)

        return Rectangle()
            .stroke(AppTheme.accentGreen, lineWidth: 2)
            .frame(width: displayed.width, height: displayed.height)
            .contentShape(Rectangle())
            .position(x: displayed.midX, y: displayed.midY)
            .gesture(moveGesture(imageFrame: imageFrame))
            .overlay(
                ForEach(Corner.allCases, id: \.self) { corner in
                    Circle()
                        .fill(AppTheme.accentGreen)
                        .frame(width: Self.handleSize, height: Self.handleSize)
                        .position(cornerPoint(corner, of: displayed))
                        .gesture(resizeGesture(corner: corner, imageFrame: imageFrame))
                }
            )
    }

    /// `box` ditambah drag/resize yang sedang berjalan, diklem ke batas foto.
    private func displayedBox(in imageFrame: CGRect) -> CGRect {
        if let corner = activeCorner {
            return resize(from: box, corner: corner, translation: resizeTranslation, in: imageFrame)
        }
        if dragOffset != .zero {
            return clamp(box.offsetBy(dx: dragOffset.width, dy: dragOffset.height), to: imageFrame)
        }
        return box
    }

    // MARK: - Gestures

    private func moveGesture(imageFrame: CGRect) -> some Gesture {
        DragGesture()
            .onChanged { dragOffset = $0.translation }
            .onEnded { value in
                box = clamp(box.offsetBy(dx: value.translation.width, dy: value.translation.height),
                            to: imageFrame)
                dragOffset = .zero
            }
    }

    private func resizeGesture(corner: Corner, imageFrame: CGRect) -> some Gesture {
        DragGesture()
            .onChanged { value in
                activeCorner = corner
                resizeTranslation = value.translation
            }
            .onEnded { value in
                box = resize(from: box, corner: corner, translation: value.translation, in: imageFrame)
                activeCorner = nil
                resizeTranslation = .zero
            }
    }
 
    private func resize(from rect: CGRect, corner: Corner, translation: CGSize, in imageFrame: CGRect) -> CGRect {
        let anchor = anchorPoint(corner, of: rect)
        let draggedPoint = CGPoint(x: cornerPoint(corner, of: rect).x + translation.width,
                                   y: cornerPoint(corner, of: rect).y + translation.height)

        let dx = draggedPoint.x - anchor.x
        let dy = draggedPoint.y - anchor.y
        let signX: CGFloat = dx >= 0 ? 1 : -1
        let signY: CGFloat = dy >= 0 ? 1 : -1

        // Batas sisi tergantung arah tarik: menarik ke kanan dibatasi oleh sisi
        // kanan imageFrame, menarik ke kiri oleh sisi kiri, dst -- bukan satu
        // batas tunggal untuk kedua arah.
        let maxSideX = signX > 0 ? (imageFrame.maxX - anchor.x) : (anchor.x - imageFrame.minX)
        let maxSideY = signY > 0 ? (imageFrame.maxY - anchor.y) : (anchor.y - imageFrame.minY)
        let maxSide = max(min(maxSideX, maxSideY), Self.minBoxSide)

        let side = min(max(max(abs(dx), abs(dy)), Self.minBoxSide), maxSide)

        let newCorner = CGPoint(x: anchor.x + signX * side, y: anchor.y + signY * side)
        let origin = CGPoint(x: min(anchor.x, newCorner.x), y: min(anchor.y, newCorner.y))
        return CGRect(origin: origin, size: CGSize(width: side, height: side))
    }

    private func anchorPoint(_ corner: Corner, of rect: CGRect) -> CGPoint {
        switch corner {
        case .topLeft: CGPoint(x: rect.maxX, y: rect.maxY)     // jangkar: kanan-bawah
        case .topRight: CGPoint(x: rect.minX, y: rect.maxY)    // jangkar: kiri-bawah
        case .bottomLeft: CGPoint(x: rect.maxX, y: rect.minY)  // jangkar: kanan-atas
        case .bottomRight: CGPoint(x: rect.minX, y: rect.minY) // jangkar: kiri-atas
        }
    }

    private func cornerPoint(_ corner: Corner, of rect: CGRect) -> CGPoint {
        switch corner {
        case .topLeft: CGPoint(x: rect.minX, y: rect.minY)
        case .topRight: CGPoint(x: rect.maxX, y: rect.minY)
        case .bottomLeft: CGPoint(x: rect.minX, y: rect.maxY)
        case .bottomRight: CGPoint(x: rect.maxX, y: rect.maxY)
        }
    }

    private func clampedSide(_ side: CGFloat, imageFrame: CGRect, origin: CGPoint) -> CGFloat {
        let maxSide = min(imageFrame.maxX - origin.x, imageFrame.maxY - origin.y)
        return min(max(side, Self.minBoxSide), max(maxSide, Self.minBoxSide))
    }

    private func clamp(_ rect: CGRect, to bounds: CGRect) -> CGRect {
        var r = rect
        r.origin.x = min(max(r.origin.x, bounds.minX), bounds.maxX - r.width)
        r.origin.y = min(max(r.origin.y, bounds.minY), bounds.maxY - r.height)
        return r
    }

    private func fitScale(viewport: CGSize, source: CGSize) -> CGFloat {
        guard source.width > 0, source.height > 0 else { return 1 }
        return min(viewport.width / source.width, viewport.height / source.height)
    }

    private static func initialBox(in imageFrame: CGRect) -> CGRect {
        let side = min(imageFrame.width, imageFrame.height) * 0.8
        return CGRect(x: imageFrame.midX - side / 2, y: imageFrame.midY - side / 2,
                      width: side, height: side)
    }

    // MARK: - Konfirmasi

    private func confirm() {
        guard let cropped = croppedImage() else { return }
        onConfirm(cropped)
    }

    /// Memetakan kotak di layar balik ke pixel sumber lalu memotongnya --
    /// selalu dari `pixelAccurateCGImage`, bukan `NSImage.size` yang bisa
    /// menyusut karena metadata DPI, dengan alasan sama seperti pipeline
    /// analisis.
    private func croppedImage() -> NSImage? {
        guard let cg = image.pixelAccurateCGImage, imageFrame.width > 0 else { return nil }
        let fit = fitScale(viewport: imageFrame.size, source: image.pixelSize)
        let displayed = displayedBox(in: imageFrame)

        let pixelRect = CGRect(
            x: (displayed.minX - imageFrame.minX) / fit,
            y: (displayed.minY - imageFrame.minY) / fit,
            width: displayed.width / fit,
            height: displayed.height / fit
        )
        guard let croppedCG = cg.cropping(to: pixelRect) else { return nil }
        return NSImage(cgImage: croppedCG, size: NSSize(width: croppedCG.width, height: croppedCG.height))
    }
}

// NSImage.size berbasis point, bukan pixel -- foto yang bawa metadata DPI
// bisa melaporkan point jauh lebih kecil dari pixel aslinya. Baik pipeline
// analisis maupun UI crop butuh ukuran pixel asli, jadi ini dibuat sebagai
// extension bersama, bukan diduplikasi di dua tempat.
extension NSImage {
    var pixelSize: CGSize {
        representations.reduce(into: CGSize.zero) { size, rep in
            size.width = max(size.width, CGFloat(rep.pixelsWide))
            size.height = max(size.height, CGFloat(rep.pixelsHigh))
        }
    }

    var pixelAccurateCGImage: CGImage? {
        let pixels = pixelSize
        var rect = CGRect(origin: .zero,
                          size: pixels.width > 0 && pixels.height > 0 ? pixels : size)
        return cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }
}
