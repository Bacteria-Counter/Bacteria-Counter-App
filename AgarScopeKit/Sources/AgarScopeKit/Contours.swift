import Foundation

/// Contour measurements on a binary mask, reproducing what the Python pipeline
/// reads from cv2.findContours + contourArea + arcLength.
///
/// Written out rather than using Vision's VNDetectContoursRequest because that
/// returns normalised curves after its own simplification, and the circularity
/// threshold of 0.75 was calibrated against OpenCV's polygon area and perimeter
/// specifically. A differently simplified outline changes the perimeter most,
/// and circularity divides by perimeter squared.
enum Contours {
    /// Trace the outer boundary of the largest connected component.
    ///
    /// Moore-neighbour tracing, the same border-following family OpenCV uses for
    /// RETR_EXTERNAL, walking clockwise from the first foreground pixel found.
    static func outerBoundary(_ mask: [UInt8], width: Int, height: Int) -> [(Int, Int)] {
        var best: [(Int, Int)] = []
        var visited = [Bool](repeating: false, count: width * height)

        // 8-neighbour offsets, clockwise from due east.
        let dx = [1, 1, 0, -1, -1, -1, 0, 1]
        let dy = [0, 1, 1, 1, 0, -1, -1, -1]

        func isSet(_ x: Int, _ y: Int) -> Bool {
            x >= 0 && y >= 0 && x < width && y < height && mask[y * width + x] != 0
        }

        for y in 0..<height {
            for x in 0..<width where isSet(x, y) && !visited[y * width + x] {
                // Only start on a boundary pixel of an untraced component.
                if isSet(x - 1, y) { continue }
                var contour: [(Int, Int)] = []
                var cx = x, cy = y
                var dir = 6                     // arrived from the north-west
                let startX = x, startY = y
                var startDir = -1
                var guardCount = 0
                repeat {
                    contour.append((cx, cy))
                    visited[cy * width + cx] = true
                    var found = false
                    // Resume the scan just behind where we came from.
                    for k in 0..<8 {
                        let d = (dir + 6 + k) % 8
                        let nx = cx + dx[d], ny = cy + dy[d]
                        if isSet(nx, ny) {
                            if startDir < 0 { startDir = d }
                            cx = nx; cy = ny; dir = d
                            found = true
                            break
                        }
                    }
                    if !found { break }         // isolated pixel
                    guardCount += 1
                } while !(cx == startX && cy == startY) && guardCount < width * height
                if contour.count > best.count { best = contour }
            }
        }
        return best
    }

    /// Shoelace area of the traced polygon, as cv2.contourArea returns.
    static func area(_ pts: [(Int, Int)]) -> Double {
        guard pts.count >= 3 else { return 0 }
        var sum = 0.0
        for i in 0..<pts.count {
            let a = pts[i], b = pts[(i + 1) % pts.count]
            sum += Double(a.0 * b.1 - b.0 * a.1)
        }
        return abs(sum) / 2
    }

    /// Closed-curve perimeter, as cv2.arcLength(closed: true) returns.
    static func perimeter(_ pts: [(Int, Int)]) -> Double {
        guard pts.count >= 2 else { return 0 }
        var total = 0.0
        for i in 0..<pts.count {
            let a = pts[i], b = pts[(i + 1) % pts.count]
            total += (Double(b.0 - a.0) * Double(b.0 - a.0)
                      + Double(b.1 - a.1) * Double(b.1 - a.1)).squareRoot()
        }
        return total
    }

    /// Smallest circle enclosing a point set -- cv2.minEnclosingCircle, which
    /// _sam_response() uses to turn each kept mask into an overlay circle.
    ///
    /// Welzl's algorithm with a shuffled input. The shuffle is what makes it
    /// linear on average; the fixed seed keeps the same photo drawing the same
    /// circle twice, which matters when a user is comparing two runs by eye.
    static func minEnclosingCircle(_ pts: [(Int, Int)]) -> (cx: Double, cy: Double, r: Double) {
        guard !pts.isEmpty else { return (0, 0, 0) }
        var p = pts.map { (Double($0.0), Double($0.1)) }
        var rng: UInt64 = 0x9E3779B97F4A7C15
        func next() -> UInt64 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng }
        for i in stride(from: p.count - 1, to: 0, by: -1) {
            p.swapAt(i, Int(next() % UInt64(i + 1)))
        }

        func circle2(_ a: (Double, Double), _ b: (Double, Double)) -> (Double, Double, Double) {
            let cx = (a.0 + b.0) / 2, cy = (a.1 + b.1) / 2
            return (cx, cy, max(dist((cx, cy), a), dist((cx, cy), b)))
        }
        func dist(_ a: (Double, Double), _ b: (Double, Double)) -> Double {
            ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1)).squareRoot()
        }
        func circle3(_ a: (Double, Double), _ b: (Double, Double),
                     _ c: (Double, Double)) -> (Double, Double, Double) {
            let d = 2 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1))
            if abs(d) < 1e-12 { return circle2(a, b) }
            let a2 = a.0 * a.0 + a.1 * a.1, b2 = b.0 * b.0 + b.1 * b.1, c2 = c.0 * c.0 + c.1 * c.1
            let cx = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d
            let cy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d
            return (cx, cy, dist((cx, cy), a))
        }
        func inside(_ c: (Double, Double, Double), _ q: (Double, Double)) -> Bool {
            dist((c.0, c.1), q) <= c.2 + 1e-9
        }

        var best = (p[0].0, p[0].1, 0.0)
        for i in 0..<p.count where !inside(best, p[i]) {
            best = (p[i].0, p[i].1, 0.0)
            for j in 0..<i where !inside(best, p[j]) {
                best = circle2(p[i], p[j])
                for k in 0..<j where !inside(best, p[k]) {
                    best = circle3(p[i], p[j], p[k])
                }
            }
        }
        return (best.0, best.1, best.2)
    }

    /// (circularity, area) for one mask. 4*pi*A/P^2 is 1 for a perfect disc.
    static func circularity(_ mask: [UInt8], width: Int, height: Int) -> (Double, Double) {
        let pts = outerBoundary(mask, width: width, height: height)
        let a = area(pts)
        let p = perimeter(pts)
        guard p > 0 else { return (0, a) }
        return (4 * Double.pi * a / (p * p), a)
    }
}
