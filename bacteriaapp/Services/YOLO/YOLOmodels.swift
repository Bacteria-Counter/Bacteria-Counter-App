//
//  YOLOv26sInference.swift
//  bacteriaapp
//
//  Created by Regina Celine Adiwinata on 13/08/26.
//

import Foundation

nonisolated enum YOLOModelVariant: String, CaseIterable, Identifiable, Sendable {
    case v11s
    case v26s

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .v11s: return "YOLOv11s"
        case .v26s: return "YOLOv26s"
        }
    }

    /// Nama file .mlmodelc di app bundle (sesuaikan kalau nama file kamu beda).
    var resourceName: String {
        switch self {
        case .v11s: return "YOLOv11s"
        case .v26s: return "YOLOv26s"
        }
    }

    var inputSize: Int {
        switch self {
        case .v11s: return 1024
        case .v26s: return 1280
        }
    }

    /// true kalau model masih pakai pipeline NMS terpisah (input iouThreshold/
    /// confidenceThreshold, output confidence+coordinates 2 tensor).
    /// false kalau model end2end (1 output tensor [1, N, 6], sudah final).
    var usesLegacyNMSPipeline: Bool {
        self == .v11s
    }

    var confidenceThreshold: Double { 0.25 }
    var iouThreshold: Double { 0.45 }  // hanya dipakai kalau usesLegacyNMSPipeline
}
