import Foundation

/// APHA 2002 colony-counting rules -- turning per-plate counts into the figure
/// a lab actually reports. Ported from cfu_calculator.py.
///
/// This is the layer above detection. The models answer "how many colonies are
/// on this photo"; this answers "given plates at several dilutions, possibly in
/// duplicate, possibly ruined, what number goes on the report?".
///
/// Two things are deliberately preserved from the Python rather than tidied:
/// the half-away-from-zero rounding (the source document rounds 32500 to 33000
/// where a scientific convention would give 32000), and regulation 3 computing
/// its ratio on the already-rounded figures, which is what the document's
/// worked examples do.
public enum CFU {
    /// A 15x100 mm dish is ~56 cm^2, the document's own conversion factor.
    public static let defaultDishAreaCm2 = 56.0
    public static let countableMin = 25
    public static let countableMax = 250
    public static let densityLimitPerCm2 = 100.0
    public static let spreadLabAccidentFraction = 0.25

    public enum PlateStatus: String {
        case ok, spreading, labAccident = "lab_accident", tntc
    }

    /// One physical petri dish. `dilution` is the factor as a fraction --
    /// 0.01 for 1:100 -- so a larger value means less diluted.
    public struct Plate {
        public var dilution: Double
        public var count: Int?
        public var status: PlateStatus
        /// For a spreader, how much of the dish it covers (0-1). Past 25% the
        /// plate becomes a laboratory accident under regulation 8.
        public var spreadFraction: Double?

        public init(dilution: Double, count: Int? = nil, status: PlateStatus = .ok,
                    spreadFraction: Double? = nil) {
            self.dilution = dilution; self.count = count
            self.status = status; self.spreadFraction = spreadFraction
        }

        var effectiveStatus: PlateStatus {
            if status == .spreading, let f = spreadFraction, f > spreadLabAccidentFraction {
                return .labAccident
            }
            return status
        }
    }

    public struct Result {
        /// nil when the outcome is not reportable as a number at all.
        public var value: Double?
        public var display: String
        public var regulations: [Int]
        public var estimated: Bool
        /// "<" or ">" when the figure is a bound rather than a measurement.
        public var bounded: String?
        public var detail: String
    }

    /// Round to `digits` significant figures, half AWAY from zero.
    ///
    /// Not the usual half-to-even. Every worked example in the document that
    /// shows its arithmetic rounds this way -- most explicitly sample 1009,
    /// "32500 = 3.3 x 10^4".
    public static func roundSig(_ x: Double, digits: Int = 2) -> Double {
        if x == 0 { return 0 }
        let exponent = floor(log10(abs(x)))
        let q = pow(10.0, exponent - Double(digits) + 1)
        return (x / q).rounded(.toNearestOrAwayFromZero) * q
    }

    /// Render as the document does: "2.3 x 10^4 CFU/ml (estimated)".
    public static func format(_ value: Double, estimated: Bool = false,
                             bounded: String? = nil, unit: String = "CFU/ml") -> String {
        var base: String
        if value == 0 {
            base = "0 \(unit)"
        } else {
            let exponent = Int(floor(log10(abs(value))))
            let mantissa = value / pow(10.0, Double(exponent))
            var m = String(format: "%.1f", mantissa)
            while m.hasSuffix("0") { m.removeLast() }
            if m.hasSuffix(".") { m.removeLast() }
            base = "\(m) x 10^\(exponent) \(unit)"
        }
        if let b = bounded { base = "\(b) \(base)" }
        if estimated { base += " (estimated)" }
        return base
    }

    public struct Countability: Sendable {
        public var status: String
        public var regulation: Int
        public var reliable: Bool
        public var densityPerCm2: Double
        public var advisory: String
    }

    /// Judge a single plate against the reliability rules, so the app can say
    /// more than a bare number. This deliberately does NOT change the count --
    /// it labels how far the count can be trusted, which is otherwise something
    /// a technician has to remember by hand.
    public static func assess(count: Int, dishAreaCm2: Double = defaultDishAreaCm2) -> Countability {
        let density = dishAreaCm2 > 0 ? Double(count) / dishAreaCm2 : 0
        let rounded = (density * 10).rounded() / 10
        if count == 0 {
            return Countability(status: "no_growth", regulation: 6, reliable: false,
                                densityPerCm2: 0, advisory: "Tidak ada koloni terdeteksi.")
        }
        if density > densityLimitPerCm2 {
            return Countability(status: "tntc", regulation: 7, reliable: false,
                                densityPerCm2: rounded,
                                advisory: String(format: "TNTC - kepadatan %.0f koloni/cm^2 melebihi ", density)
                                    + "\(Int(densityLimitPerCm2))/cm^2. Angka ini di luar batas "
                                    + "estimasi; gunakan pengenceran yang lebih tinggi.")
        }
        if count < countableMin {
            return Countability(status: "below_range", regulation: 5, reliable: false,
                                densityPerCm2: rounded,
                                advisory: "Di bawah \(countableMin) koloni - hanya boleh "
                                    + "dilaporkan sebagai estimasi.")
        }
        if count > countableMax {
            return Countability(status: "above_range", regulation: 4, reliable: false,
                                densityPerCm2: rounded,
                                advisory: "Di atas \(countableMax) koloni - hanya boleh "
                                    + "dilaporkan sebagai estimasi. Gunakan pengenceran lebih "
                                    + "tinggi bila ada.")
        }
        return Countability(status: "countable", regulation: 1, reliable: true,
                            densityPerCm2: rounded,
                            advisory: "Dalam rentang layak hitung \(countableMin)-\(countableMax) "
                                + "koloni.")
    }

    /// All plates sharing one dilution, resolved to a single number.
    struct Level {
        var dilution: Double
        var count: Int?
        var status: PlateStatus
    }

    /// Regulation 2: average duplicates at one dilution, ignoring spreaders and
    /// lab accidents. If nothing usable remains the level carries the status
    /// forward instead -- a TNTC plate still says "very high", which regulation
    /// 7 can use, whereas an accident says nothing at all.
    static func resolveDuplicates(_ plates: [Plate]) -> Level {
        let dilution = plates[0].dilution
        let usable = plates.filter { $0.effectiveStatus == .ok && $0.count != nil }
        if !usable.isEmpty {
            let avg = Double(usable.reduce(0) { $0 + ($1.count ?? 0) }) / Double(usable.count)
            return Level(dilution: dilution,
                         count: Int(avg.rounded(.toNearestOrAwayFromZero)), status: .ok)
        }
        let statuses = Set(plates.map { $0.effectiveStatus })
        if statuses.contains(.tntc) { return Level(dilution: dilution, count: nil, status: .tntc) }
        if statuses.contains(.spreading) {
            return Level(dilution: dilution, count: nil, status: .spreading)
        }
        return Level(dilution: dilution, count: nil, status: .labAccident)
    }

    public enum CFUError: Error { case noPlates, unhandled }

    public static func calculate(_ plates: [Plate],
                                 dishAreaCm2: Double = defaultDishAreaCm2,
                                 method: String = "pour",
                                 unit: String = "CFU/ml") throws -> Result {
        guard !plates.isEmpty else { throw CFUError.noPlates }
        // Spread plate uses a 0.1 mL inoculum, so its result is divided by 0.1.
        let factor = method == "spread" ? 10.0 : 1.0

        var byDilution: [Double: [Plate]] = [:]
        for p in plates { byDilution[p.dilution, default: []].append(p) }
        // Least diluted first: 1e-2 before 1e-3.
        let levels = byDilution.keys.sorted(by: >).map { resolveDuplicates(byDilution[$0]!) }
        let usedDupRule = byDilution.values.contains { $0.count > 1 }

        func cfuOf(_ lv: Level) -> Double {
            roundSig(Double(lv.count!) / lv.dilution * factor)
        }

        var regs: [Int] = []
        if usedDupRule { regs.append(2) }
        if levels.contains(where: { $0.status == .spreading || $0.status == .labAccident })
            || plates.contains(where: { $0.status == .spreading || $0.status == .labAccident }) {
            regs.append(8)
        }
        func finish(_ r: Result) -> Result {
            var out = r
            out.regulations = Array(Set(out.regulations)).sorted()
            return out
        }

        let countable = levels.filter { $0.status == .ok && $0.count != nil }
        let inRange = countable.filter { $0.count! >= countableMin && $0.count! <= countableMax }

        // --- Nothing usable at all ---
        if countable.isEmpty && !levels.contains(where: { $0.status == .tntc }) {
            let statuses = Set(levels.map { $0.status })
            if statuses == [.spreading] || (statuses.contains(.spreading)
                                            && statuses.contains(.labAccident)) {
                return finish(Result(value: nil, display: "Spreading", regulations: regs + [8],
                                     estimated: false, bounded: nil,
                                     detail: "Every plate is a spreader -- no countable plate."))
            }
            return finish(Result(value: nil, display: "Lab accident", regulations: regs + [8],
                                 estimated: false, bounded: nil,
                                 detail: "Every plate is a laboratory accident -- nothing to report."))
        }

        // --- Regulation 6: no colonies anywhere ---
        if !countable.isEmpty && countable.allSatisfy({ $0.count == 0 }) {
            let lowest = countable.map { $0.dilution }.max()!   // least diluted
            let bound = 1.0 / lowest * factor
            return finish(Result(value: bound,
                                 display: format(bound, estimated: true, bounded: "<", unit: unit),
                                 regulations: regs + [6], estimated: true, bounded: "<",
                                 detail: "No colonies at any dilution -- reported as less than "
                                       + "the lowest dilution factor."))
        }

        // --- Regulation 7: density beyond estimation ---
        for lv in countable where lv.count! > countableMax
            && Double(lv.count!) / dishAreaCm2 > densityLimitPerCm2 {
            let highest = levels.map { $0.dilution }.min()!     // most diluted
            let bound = dishAreaCm2 * densityLimitPerCm2 / highest * factor
            return finish(Result(value: bound,
                                 display: format(bound, estimated: true, bounded: ">", unit: unit),
                                 regulations: regs + [7], estimated: true, bounded: ">",
                                 detail: "\(lv.count!) colonies on \(fmtG(dishAreaCm2)) cm^2 exceeds "
                                       + "\(Int(densityLimitPerCm2))/cm^2 -- beyond estimation."))
        }

        // --- Regulation 1: exactly one level in 25-250 ---
        if inRange.count == 1 {
            let lv = inRange[0]
            let value = cfuOf(lv)
            return finish(Result(value: value, display: format(value, unit: unit),
                                 regulations: regs + [1], estimated: false, bounded: nil,
                                 detail: "\(lv.count!) colonies at 1:\(fmtG(1 / lv.dilution)) is "
                                       + "within \(countableMin)-\(countableMax)."))
        }

        // --- Regulation 3: two levels in 25-250 ---
        if inRange.count >= 2 {
            let lower = inRange[0], higher = inRange[1]   // lower dilution first
            let cl = cfuOf(lower), ch = cfuOf(higher)
            let ratio = cl != 0 ? ch / cl : Double.infinity
            if ratio >= 2 {
                return finish(Result(value: cl, display: format(cl, unit: unit),
                                     regulations: regs + [3], estimated: false, bounded: nil,
                                     detail: String(format: "Ratio %.1f >= 2 -- using the lower "
                                                    + "dilution's result.", ratio)))
            }
            let averaged = roundSig((cl + ch) / 2)
            return finish(Result(value: averaged, display: format(averaged, unit: unit),
                                 regulations: regs + [3], estimated: false, bounded: nil,
                                 detail: String(format: "Ratio %.1f < 2 -- averaging both "
                                                + "dilution levels.", ratio)))
        }

        // --- Regulation 4: nothing in range, but something over 250 ---
        let over = countable.filter { $0.count! > countableMax }
        if !over.isEmpty {
            let closest = over.min(by: { abs($0.count! - countableMax) < abs($1.count! - countableMax) })!
            let value = cfuOf(closest)
            var r = regs + [4]
            if levels.contains(where: { $0.status == .tntc }) || closest.count! > countableMax {
                r.append(7)
            }
            return finish(Result(value: value,
                                 display: format(value, estimated: true, unit: unit),
                                 regulations: r, estimated: true, bounded: nil,
                                 detail: "No plate in \(countableMin)-\(countableMax); using the "
                                       + "plate closest to \(countableMax) (\(closest.count!)) as "
                                       + "an estimate."))
        }

        // --- Regulation 5: everything under 25 ---
        let under = countable.filter { $0.count! < countableMin }
        if !under.isEmpty {
            let leastDiluted = under.max(by: { $0.dilution < $1.dilution })!
            let value = cfuOf(leastDiluted)
            return finish(Result(value: value,
                                 display: format(value, estimated: true, unit: unit),
                                 regulations: regs + [5], estimated: true, bounded: nil,
                                 detail: "All plates under \(countableMin) -- using the "
                                       + "least-diluted plate (\(leastDiluted.count!) colonies) "
                                       + "as an estimate."))
        }

        throw CFUError.unhandled
    }

    /// Python's "%g": drop the trailing zeros an integral value would show.
    static func fmtG(_ v: Double) -> String {
        v == v.rounded() && abs(v) < 1e15 ? String(Int(v)) : String(format: "%g", v)
    }
}
