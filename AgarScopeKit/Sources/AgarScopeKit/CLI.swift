import Foundation

/// The command-line front end, kept inside the library rather than in the
/// executable target.
///
/// Every figure in README.md was produced by these flags, so they have to run
/// against exactly the code the app links -- not a parallel copy that could
/// drift from it. The executable target is now a single line that calls in
/// here.
public enum AgarScopeCLI {
    public static func main(_ a: [String]) -> Int32 {
    if a.count > 1 && a[1] == "--cfu-test" { return CFUTest.run() == 23 ? 0 : 1 }
    if a.count >= 5 && a[1] == "--dump-lb" {
        guard let bmp = ImageOps.load(a[2]) else { return 1 }
        let size = Int(a[3])!
        let lb = ImageOps.letterbox(Pipeline.clahe(bmp), targetW: size, targetH: size)
        var rgb = [UInt8](); rgb.reserveCapacity(size*size*3)
        for i in 0..<(size*size) { rgb.append(lb.rgba[i*4]); rgb.append(lb.rgba[i*4+1]); rgb.append(lb.rgba[i*4+2]) }
        let ref = [UInt8](try! Data(contentsOf: URL(fileURLWithPath: a[4])))
        var diff = 0, maxd = 0; var sum = 0.0
        for i in 0..<min(ref.count, rgb.count) {
            let d = abs(Int(rgb[i]) - Int(ref[i]))
            if d > 0 { diff += 1 }
            maxd = max(maxd, d); sum += Double(d)
        }
        print(String(format: "beda letterbox: %.2f%% byte, rata2 %.4f, maks %d",
                     Double(diff)/Double(ref.count)*100, sum/Double(ref.count), maxd))
        return 0
    }
    if a.count >= 4 && a[1] == "--colony-size" {
        // Median colony radius as a percentage of the dish radius -- the same
        // quantity sam_micro's escalation triggers on. Needed to ask whether a
        // benchmark can separate models by COLONY SIZE at all, which is a
        // different question from separating them by colonies PER PLATE.
        let modelDir = a[2]
        for path in a[3...] {
            guard let bmp = ImageOps.load(path) else { continue }
            let name = (path as NSString).lastPathComponent
            guard let r = try? Pipeline.count(image: bmp, modelDir: modelDir, micro: false),
                  let d = r.dish, !r.colonies.isEmpty else {
                print("\(name)|0|-"); continue
            }
            let areas = r.colonies.map { $0.area }.sorted()
            let median = areas[areas.count / 2]
            let pct = (median / Double.pi).squareRoot() / d.r * 100
            print(String(format: "%@|%d|%.3f", name, r.count, pct))
        }
        return 0
    }
    if a.count >= 6 && a[1] == "--dump-lab2rgb" {
        // LAB -> RGB alone, fed OpenCV's own LAB bytes. make_clahe is a round trip
        // (BGR->LAB, adjust L, LAB->BGR) and a whole-transform comparison cannot
        // say which half of it a difference came from.
        let w = Int(a[3])!, h = Int(a[4])!
        let lab = [UInt8](try! Data(contentsOf: URL(fileURLWithPath: a[2])))
        var rgb = [UInt8](); rgb.reserveCapacity(w * h * 3)
        for i in 0..<(w * h) {
            let (r, g, b) = ColorLAB.toRGB(L: Double(lab[i*3]), a: Double(lab[i*3+1]),
                                           b: Double(lab[i*3+2]))
            rgb.append(r); rgb.append(g); rgb.append(b)
        }
        rgb.withUnsafeBufferPointer { try! Data(buffer: $0).write(to: URL(fileURLWithPath: a[5])) }
        print("lab2rgb|\(w)x\(h)")
        return 0
    }
    if a.count >= 5 && a[1] == "--dump-prep" {
        // Preprocessed pixels as raw RGB, so OpenCV can be asked the same question.
        // "area" is CSRNet's INTER_AREA resize rather than a model preprocessor,
        // but it is the same kind of claim and deserves the same kind of check.
        guard let bmp = ImageOps.load(a[3]) else { return 1 }
        let p = a[2] == "area" ? CSRNet.resizeArea(bmp, newW: CSRNet.workingSize,
                                                   newH: CSRNet.workingSize)
                               : Detect.preprocess(a[2], bmp)
        var rgb = [UInt8](); rgb.reserveCapacity(p.width * p.height * 3)
        for i in 0..<(p.width * p.height) {
            rgb.append(p.rgba[i*4]); rgb.append(p.rgba[i*4+1]); rgb.append(p.rgba[i*4+2])
        }
        print("prep|\(p.width)x\(p.height)")
        rgb.withUnsafeBufferPointer { try! Data(buffer: $0).write(to: URL(fileURLWithPath: a[4])) }
        return 0
    }
    if a.count >= 5 && a[1] == "--dump-density" {
        // The CSRNet density map itself, not just its sum -- a sum can match while
        // the map underneath is wrong in two places that cancel.
        guard let bmp = ImageOps.load(a[3]) else { return 1 }
        let r = try! CSRNet.count(image: bmp, modelDir: a[2])
        print("density|\(r.mapW)x\(r.mapH)|\(r.density.reduce(0, +))")
        r.density.withUnsafeBufferPointer { try! Data(buffer: $0).write(to: URL(fileURLWithPath: a[4])) }
        return 0
    }
    if a.count >= 6 && a[1] == "--dump-boxes" {
        // Post-NMS boxes in original-image coordinates, so the comparison can be
        // per-box rather than only on the count -- two paths can agree on how many
        // colonies there are while disagreeing about which.
        guard let bmp = ImageOps.load(a[4]) else { return 1 }
        let r = try! Detect.count(image: bmp, modelDir: a[2], key: a[3])
        print("boxes|\(r.count)|\(r.imgsz)")
        var flat = [Float]()
        for b in r.boxes {
            flat.append(contentsOf: [Float(b.x1), Float(b.y1), Float(b.x2), Float(b.y2), Float(b.conf)])
        }
        flat.withUnsafeBufferPointer { try! Data(buffer: $0).write(to: URL(fileURLWithPath: a[5])) }
        return 0
    }
    if a.count >= 5 && a[1] == "--dump-cand" {
        // Ekspor kandidat pra-NMS sebagai biner, supaya torchvision bisa
        // menjalankan NMS pada kotak yang identik.
        guard let bmp = ImageOps.load(a[3]) else { return 1 }
        let size = 4480
        let lb = ImageOps.letterbox(Pipeline.clahe(bmp), targetW: size, targetH: size)
        let model = try! Inference.load(dir: a[2], name: "fastsam_\(size)")
        let raw = try! Inference.run(model, lb)!
        let n = raw.det.shape[2].intValue
        let ptr = raw.det.dataPointer.assumingMemoryBound(to: Float.self)
        var out = [Float]()
        for i in 0..<n {
            let c = ptr[4*n+i]
            if c <= 0.2 { continue }
            let cx = ptr[0*n+i], cy = ptr[1*n+i], w = ptr[2*n+i], h = ptr[3*n+i]
            out.append(contentsOf: [cx-w/2, cy-h/2, cx+w/2, cy+h/2, c])
        }
        print("kandidat|\(out.count/5)")
        out.withUnsafeBufferPointer { try! Data(buffer: $0).write(to: URL(fileURLWithPath: a[4])) }
        print("swift_nms|\(PostProcess.nms(raw.det, confThreshold: 0.2).count)")
        return 0
    }
    if a.count > 3 && a[1] == "--lchan" {
        guard let bmp = ImageOps.load(a[2]) else { return 1 }
        let n = bmp.width * bmp.height
        var l = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            let (L,_,_) = ColorLAB.toLAB(r: bmp.rgba[i*4], g: bmp.rgba[i*4+1], b: bmp.rgba[i*4+2])
            l[i] = UInt8(max(0,min(255,Int(L.rounded(.toNearestOrEven)))))
        }
        let mean = l.reduce(0.0) { $0 + Double($1) } / Double(n)
        print(String(format: "swift L mean %.4f", mean))
        let ref = [UInt8](try! Data(contentsOf: URL(fileURLWithPath: a[3])))
        var diff = 0, maxd = 0, sum = 0.0
        for i in 0..<min(ref.count, n) {
            let d = abs(Int(l[i]) - Int(ref[i]))
            if d > 0 { diff += 1 }
            maxd = max(maxd, d); sum += Double(d)
        }
        print(String(format: "beda vs OpenCV: %.2f%% piksel, rata2 %.4f, maks %d",
                     Double(diff)/Double(ref.count)*100, sum/Double(ref.count), maxd))
        return 0
    }
    if a.count > 4 && a[1] == "--clahe-bin" {
        let w = Int(a[3])!, h = Int(a[4])!
        let data = try! Data(contentsOf: URL(fileURLWithPath: a[2]))
        let l = [UInt8](data)
        let out = CLAHE.applyToLuminance(l, width: w, height: h)
        let mean = out.reduce(0.0) { $0 + Double($1) } / Double(out.count)
        print(String(format: "swift CLAHE mean %.4f", mean))
        if a.count > 5 {
            let ref = [UInt8](try! Data(contentsOf: URL(fileURLWithPath: a[5])))
            var diff = 0, maxd = 0, sum = 0.0
            for i in 0..<min(ref.count, out.count) {
                let d = abs(Int(out[i]) - Int(ref[i]))
                if d > 0 { diff += 1 }
                maxd = max(maxd, d); sum += Double(d)
            }
            print(String(format: "beda: %.2f%% piksel, rata2 %.4f, maks %d",
                         Double(diff)/Double(ref.count)*100, sum/Double(ref.count), maxd))
        }
        return 0
    }
    if a.count > 1 && a[1] == "--inv-test" {
        for (L,A,B) in [(216.0,141.0,105.0),(176.0,147.0,95.0),(103.0,136.0,114.0),
                        (255.0,128.0,128.0),(0.0,128.0,128.0),(137.0,128.0,128.0),
                        (200.0,120.0,140.0),(80.0,150.0,100.0)] {
            let (r,g,b) = ColorLAB.toRGB(L: L, a: A, b: B)
            print("LAB(\(Int(L)),\(Int(A)),\(Int(B))) -> RGB \(r),\(g),\(b)")
        }
        return 0
    }
    // Mode diagnosis: laporkan tiap tahap, untuk dibandingkan dengan Python.
    if a.count >= 5 && a[1] == "--trace-extL" {
        // Jalankan pipeline memakai kanal L dari OpenCV, untuk memisahkan
        // pengaruh konversi LAB dari sisa rantai.
        let modelDir = a[2], path = a[3], lpath = a[4]
        guard let bmp = ImageOps.load(path) else { return 1 }
        let n = bmp.width * bmp.height
        var ab = [(UInt8, UInt8)](repeating: (128,128), count: n)
        for i in 0..<n {
            let (_,A,B) = ColorLAB.toLAB(r: bmp.rgba[i*4], g: bmp.rgba[i*4+1], b: bmp.rgba[i*4+2])
            ab[i] = (UInt8(max(0,min(255,Int(A.rounded())))), UInt8(max(0,min(255,Int(B.rounded())))))
        }
        let lext = [UInt8](try! Data(contentsOf: URL(fileURLWithPath: lpath)))
        let eq = CLAHE.applyToLuminance(lext, width: bmp.width, height: bmp.height)
        var ci = bmp
        for i in 0..<n {
            let (r,g,b) = ColorLAB.toRGB(L: Double(eq[i]), a: Double(ab[i].0), b: Double(ab[i].1))
            ci.rgba[i*4]=r; ci.rgba[i*4+1]=g; ci.rgba[i*4+2]=b
        }
        var sm = 0.0
        for i in stride(from:0,to:ci.rgba.count,by:4) { sm += Double(ci.rgba[i])+Double(ci.rgba[i+1])+Double(ci.rgba[i+2]) }
        print(String(format: "clahe_mean|%.5f", sm/Double(n*3)/255.0))
        let dish = DishDetect.find(bmp)
        let size = 4480
        let lb = ImageOps.letterbox(ci, targetW: size, targetH: size)
        let model = try! Inference.load(dir: modelDir, name: "fastsam_\(size)")
        let raw = try! Inference.run(model, lb)!
        let dets = PostProcess.nms(raw.det, confThreshold: Pipeline.conf)
        print("nms|\(dets.count)")
        let masks = PostProcess.masks(proto: raw.proto, detections: dets,
                                      letterboxW: size, letterboxH: size,
                                      originalW: bmp.width, originalH: bmp.height)
        let areas = masks.map { m in m.reduce(0) { $0 + Int($1) } }.sorted()
        print("mask|\(masks.count)|\(areas.isEmpty ? 0 : areas[areas.count/2])")
        print("filter|\(Pipeline.filter(masks: masks, width: bmp.width, height: bmp.height, original: bmp, dish: dish).count)")
        return 0
    }
    if a.count >= 4 && a[1] == "--trace" {
        let modelDir = a[2], path = a[3]
        let size = a.count > 4 ? (Int(a[4]) ?? 4480) : 4480
        guard let bmp = ImageOps.load(path) else { print("gagal muat"); return 1 }
        print("gambar|\(bmp.width)x\(bmp.height)")
        let dish = DishDetect.find(bmp)
        if let d = dish { print(String(format: "cawan|%.1f|%.1f|%.1f", d.cx, d.cy, d.r)) }
        let ci = Pipeline.clahe(bmp)
        var s = 0.0
        for i in stride(from: 0, to: ci.rgba.count, by: 4) {
            s += Double(ci.rgba[i]) + Double(ci.rgba[i+1]) + Double(ci.rgba[i+2])
        }
        print(String(format: "clahe_mean|%.5f", s / Double(ci.width*ci.height*3) / 255.0))
        let lb = ImageOps.letterbox(ci, targetW: size, targetH: size)
        var s2 = 0.0
        for i in stride(from: 0, to: lb.rgba.count, by: 4) {
            s2 += Double(lb.rgba[i]) + Double(lb.rgba[i+1]) + Double(lb.rgba[i+2])
        }
        print(String(format: "letterbox|%dx%d|%.5f", lb.width, lb.height, s2/Double(lb.width*lb.height*3)/255.0))
        do {
            let model = try Inference.load(dir: modelDir, name: "fastsam_\(size)")
            guard let raw = try Inference.run(model, lb) else { print("inferensi gagal"); return 1 }
            // 5 kandidat teratas mentah, untuk diadu dengan Python
            let ds = raw.det.shape.map { $0.intValue }
            let n = ds[2]
            let ptr = raw.det.dataPointer.assumingMemoryBound(to: Float.self)
            print("strides|\(raw.det.strides.map { $0.intValue })")
            var top: [(Float, Int)] = []
            for i in 0..<n { top.append((ptr[4*n+i], i)) }
            top.sort { $0.0 > $1.0 }
            for (c, i) in top.prefix(5) {
                print(String(format: "cand|%.1f|%.1f|%.1f|%.1f|%.4f",
                             ptr[0*n+i], ptr[1*n+i], ptr[2*n+i], ptr[3*n+i], c))
            }
            let dets = PostProcess.nms(raw.det, confThreshold: Pipeline.conf)
            print("nms|\(dets.count)")
            let masks = PostProcess.masks(proto: raw.proto, detections: dets,
                                          letterboxW: size, letterboxH: size,
                                          originalW: bmp.width, originalH: bmp.height)
            let areas = masks.map { m in m.reduce(0) { $0 + Int($1) } }.sorted()
            print("mask|\(masks.count)|\(areas.isEmpty ? 0 : areas[areas.count/2])")
            let kept = Pipeline.filter(masks: masks, width: bmp.width, height: bmp.height,
                                       original: bmp, dish: dish)
            print("filter|\(kept.count)")
        } catch { print("error|\(error)") }
        return 0
    }

    guard a.count >= 3 else {
        print("usage: agarscope <modelDir> <image...> [--micro] [--model <key>]")
        print("  key: sam_tuned sam_micro yolo_old yolo_new mac1 mac2 dog_blend clahe lab_ab csrnet")
        return 1
    }
    let modelDir = a[1]
    // --micro predates --model and still means sam_micro. Kept working rather than
    // replaced: verify_2b.py drives the binary through it, and a verification
    // harness that has to be edited to keep passing has stopped being one.
    let micro = a.contains("--micro")
    var modelKey = micro ? "sam_micro" : "sam_tuned"
    if let i = a.firstIndex(of: "--model"), i + 1 < a.count { modelKey = a[i + 1] }

    let samKeys = ["sam_tuned", "sam_micro"]
    let yoloKeys = ["yolo_old", "yolo_new", "mac1", "mac2", "dog_blend", "clahe", "lab_ab"]
    guard samKeys.contains(modelKey) || yoloKeys.contains(modelKey) || modelKey == "csrnet" else {
        print("model tidak dikenal: \(modelKey)"); return 1
    }

    var skip = false
    for arg in a[2...] {
        if skip { skip = false; continue }
        if arg == "--model" { skip = true; continue }
        if arg.hasPrefix("--") { continue }
        let path = arg
        guard let bmp = ImageOps.load(path) else { print("\(path): gagal muat"); continue }
        let name = (path as NSString).lastPathComponent
        do {
            let t0 = Date()
            if samKeys.contains(modelKey) {
                let r = try Pipeline.count(image: bmp, modelDir: modelDir,
                                           micro: modelKey == "sam_micro")
                print(String(format: "%@|%d|%d|%@|%.1f", name, r.count, r.imgsz,
                             r.escalated ? "esk" : "-", Date().timeIntervalSince(t0)))
            } else if modelKey == "csrnet" {
                let r = try CSRNet.count(image: bmp, modelDir: modelDir)
                print(String(format: "%@|%d|%d|-|%.1f", name, r.count, CSRNet.workingSize,
                             Date().timeIntervalSince(t0)))
            } else {
                let r = try Detect.count(image: bmp, modelDir: modelDir, key: modelKey)
                print(String(format: "%@|%d|%d|-|%.1f", name, r.count, r.imgsz,
                             Date().timeIntervalSince(t0)))
            }
        } catch { print("\(path): error \(error)") }
    }

        return 0
    }
}
