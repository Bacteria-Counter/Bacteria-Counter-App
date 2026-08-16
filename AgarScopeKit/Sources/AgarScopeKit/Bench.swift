import CoreGraphics
import Foundation

/// The seam that lets this pipeline be compared against a foreign one.
///
/// The app never calls any of this. It exists because the amel/celine branch
/// carries a second, complete counting chain -- learned dish segmentation, a
/// circular crop, grayscale CLAHE, a fixed input size -- and the question
/// "which chain is better" cannot be answered by running each end to end. Two
/// chains differing in four ways at once produce a number that cannot be
/// attributed to any one of them.
///
/// So the two halves are exposed separately: `preprocess` hands out what this
/// pipeline feeds its models, and `analyze` accepts an image somebody else
/// prepared. Everything here routes into the SAME functions the app runs; none
/// of it is a reimplementation, which is the whole point of putting it in the
/// library rather than copying the sources into a harness.
///
/// One limit worth stating: a checkpoint is trained together with its
/// preprocessing, so swapping chains is expected to hurt both sides. The
/// measurement is of how MUCH, not of which chain is better in the abstract --
/// answering that would need a model retrained under each, which is a training
/// job and not a benchmark.
extension AgarScope {
    public enum Bench {
        /// This pipeline's preprocessing, alone: CLAHE on the L channel with a
        /// and b untouched.
        ///
        /// Deliberately does NOT crop and does NOT resize. Not cropping is a
        /// property of this chain rather than an omission -- the dish circle is
        /// used to choose the input size and to reject detections outside the
        /// rim, never to cut the frame down. Not resizing leaves the caller's
        /// own resize in place, so a foreign model still meets the input size
        /// it was exported at.
        public static func preprocess(image: CGImage) -> CGImage? {
            guard let bmp = ImageOps.from(image) else { return nil }
            return ImageOps.cgImage(Pipeline.clahe(bmp))
        }

        /// This pipeline's models and postprocessing, on an image prepared
        /// elsewhere.
        ///
        /// - Parameter skipCLAHE: pass true when the caller's chain has already
        ///   contrast-equalised. Only reaches the FastSAM path, which is the
        ///   only one of the three that applies CLAHE at all: `mac1`'s
        ///   transform is the identity and CSRNet has its own resize instead.
        /// - Parameter forcedSize: pins the model input size, disabling both
        ///   the adaptive rule and `sam_micro`'s 4480 escalation, so that
        ///   resolution can be measured apart from the checkpoint.
        public static func analyze(image: CGImage, model: Model, modelDirectory: URL,
                                   skipCLAHE: Bool = false,
                                   forcedSize: Int? = nil) throws -> Output {
            guard let bmp = ImageOps.from(image) else { throw AnalyzeError.imageUnreadable }
            let dir = modelDirectory.path
            guard FileManager.default.fileExists(atPath: dir) else {
                throw AnalyzeError.modelsNotFound(dir)
            }
            let w = Double(bmp.width), h = Double(bmp.height)

            if model == .csrnet {
                let r = try CSRNet.count(image: bmp, modelDir: dir)
                return Output(totalColonies: r.count, averageConfidence: 0,
                              modelUsed: model.rawValue, detections: [],
                              imageWidth: w, imageHeight: h,
                              countability: CFU.assess(count: r.count),
                              heatmapPNG: nil, imgsz: CSRNet.workingSize, escalated: false)
            }

            if model.isSAM {
                let r = try Pipeline.count(image: bmp, modelDir: dir,
                                           micro: model == .samMicro,
                                           skipCLAHE: skipCLAHE, forcedSize: forcedSize)
                let conf = r.colonies.isEmpty ? 0
                    : r.colonies.reduce(0) { $0 + $1.circularity } / Double(r.colonies.count) * 100
                return Output(totalColonies: r.count,
                              averageConfidence: (conf * 10).rounded() / 10,
                              modelUsed: model.rawValue, detections: [],
                              imageWidth: w, imageHeight: h,
                              countability: CFU.assess(count: r.count), heatmapPNG: nil,
                              imgsz: r.imgsz, escalated: r.escalated)
            }

            let r = try Detect.count(image: bmp, modelDir: dir, key: model.rawValue,
                                     forcedSize: forcedSize)
            return Output(totalColonies: r.count,
                          averageConfidence: (r.confidence * 10).rounded() / 10,
                          modelUsed: model.rawValue, detections: [],
                          imageWidth: w, imageHeight: h,
                          countability: CFU.assess(count: r.count), heatmapPNG: nil,
                          imgsz: r.imgsz, escalated: false)
        }
    }
}
