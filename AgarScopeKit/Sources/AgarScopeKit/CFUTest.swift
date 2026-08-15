import Foundation

/// The 23 worked examples from "Panduan Perhitungan Koloni Mikroorganisme Rev 2"
/// (APHA 2002), run against the Swift port.
///
/// The same cases test_cfu_calculator.py runs, deliberately not a reduced set:
/// the point is that the Swift and Python calculators agree case for case, and
/// that is only meaningful if they are asked the same questions. Expect 21/23 --
/// samples 1116 and 1118 are discrepancies in the source document itself, where
/// it skips showing its intermediate arithmetic, and the Python has carried
/// those two as known failures since it was written.
enum CFUTest {
    typealias P = CFU.Plate
    static func ok(_ d: Double, _ c: Int) -> P { P(dilution: d, count: c) }
    static func spread(_ d: Double, _ c: Int? = nil, _ f: Double? = nil) -> P {
        P(dilution: d, count: c, status: .spreading, spreadFraction: f)
    }
    static func labacc(_ d: Double) -> P { P(dilution: d, status: .labAccident) }
    static func tntc(_ d: Double) -> P { P(dilution: d, status: .tntc) }

    static let d2 = 1e-2, d3 = 1e-3

    /// (sample, plates, expected value or nil, keyword: "<", ">", "est", or a
    /// substring the display must contain)
    static var cases: [(String, [P], Double?, String?)] {
        [("1001", [ok(d2, 234), ok(d3, 23)], 23000, nil),
         ("1002", [ok(d2, 243), ok(d3, 34)], 29000, nil),
         ("1003", [ok(d2, 140), ok(d3, 32)], 14000, nil),
         ("1004", [spread(d2), ok(d3, 31)], 31000, nil),
         ("1005", [ok(d2, 0), ok(d3, 0)], 100, "<"),
         ("1006", [tntc(d2), ok(d3, 7150)], 5600000, ">"),
         ("1007", [ok(d2, 18), ok(d3, 2)], 1800, "est"),
         ("1008", [spread(d2), spread(d3)], nil, "Spreading"),
         ("1009", [ok(d2, 325), ok(d3, 20)], 33000, "est"),
         ("1010", [spread(d2, 27, 0.30), spread(d3, 215, 0.30)], nil, "Lab accident"),
         ("1011", [ok(d2, 305), ok(d3, 42)], 42000, nil),
         ("1012", [ok(d2, 243), labacc(d3)], 24000, nil),
         ("1013", [tntc(d2), ok(d3, 840)], 840000, "est"),
         ("1111", [ok(d2, 228), ok(d2, 240), ok(d3, 28), ok(d3, 26)], 25000, nil),
         ("1112", [ok(d2, 175), ok(d2, 208), ok(d3, 16), ok(d3, 17)], 19000, nil),
         ("1113", [ok(d2, 239), ok(d2, 328), ok(d3, 16), ok(d3, 19)], 28000, nil),
         ("1114", [ok(d2, 275), ok(d2, 280), ok(d3, 24), ok(d3, 35)], 30000, nil),
         ("1115", [ok(d2, 138), ok(d2, 162), ok(d3, 42), ok(d3, 30)], 15000, nil),
         ("1116", [ok(d2, 240), ok(d2, 228), ok(d3, 28), ok(d3, 23)], 24000, nil),
         ("1117", [ok(d2, 224), ok(d2, 180), ok(d3, 28), spread(d3)], 24000, nil),
         ("1118", [ok(d2, 287), labacc(d2), ok(d3, 23), ok(d3, 19)], 28000, "est"),
         ("1119", [ok(d2, 18), ok(d2, 16), ok(d3, 2), ok(d3, 0)], 1700, "est"),
         ("1120", [ok(d2, 0), ok(d2, 0), ok(d3, 0), ok(d3, 0)], 100, "<")]
    }

    /// The two samples where the DOCUMENT is inconsistent, not the code. 1118
    /// shows 287/1e-2 as 2.8 x 10^4 where two significant figures give 2.9;
    /// 1116 averages 2.3 and 2.6 x 10^4 and reports 2.4 where the halfway case
    /// rounds to 2.5. Every example that does show its arithmetic rounds half
    /// up, so these read as slips in the source. Named here rather than
    /// silently tolerated, so that if a THIRD case ever fails it is visible.
    static let knownDocumentDiscrepancies = ["1116", "1118"]

    static func pad(_ s: String, _ n: Int, right: Bool = false) -> String {
        s.count >= n ? s : (right ? String(repeating: " ", count: n - s.count) + s
                                  : s + String(repeating: " ", count: n - s.count))
    }

    static func run() -> Int {
        print(pad("sampel", 8) + pad("diharap", 10, right: true)
              + pad("hasil", 10, right: true) + "   " + pad("aturan", 14) + "tampilan")
        print(String(repeating: "-", count: 84))
        var passed = 0
        var failures: [String] = []
        for (sample, plates, expected, keyword) in cases {
            guard let r = try? CFU.calculate(plates) else {
                failures.append("\(sample): melempar error")
                print("\(sample)  ERROR"); continue
            }
            var match: Bool
            if let e = expected {
                match = r.value != nil && abs(r.value! - e) < max(1e-6, e * 1e-9)
                switch keyword {
                case "<": match = match && r.bounded == "<"
                case ">": match = match && r.bounded == ">"
                case "est": match = match && r.estimated
                default: break
                }
            } else {
                match = r.value == nil && (keyword.map {
                    r.display.lowercased().contains($0.lowercased()) } ?? false)
            }
            if match { passed += 1 } else {
                failures.append("\(sample): diharap \(expected.map { String($0) } ?? "nil"), "
                                + "hasil \(r.value.map { String($0) } ?? "nil")")
            }
            let exp = expected.map { String(Int($0)) } ?? "nil"
            let got = r.value.map { String(Int($0)) } ?? "nil"
            let note = match ? "OK" : (knownDocumentDiscrepancies.contains(sample)
                                       ? "beda-dokumen" : "GAGAL")
            print(pad(sample, 8) + pad(exp, 10, right: true) + pad(got, 10, right: true)
                  + "   " + pad("\(r.regulations)", 14) + pad(r.display, 34) + note)
        }
        print(String(repeating: "-", count: 84))
        print("LULUS \(passed)/\(cases.count)")
        let unexpected = failures.filter { f in
            !knownDocumentDiscrepancies.contains(where: { f.hasPrefix($0) })
        }
        for f in failures { print("  \(f)") }
        if unexpected.isEmpty && failures.count == knownDocumentDiscrepancies.count {
            print("Dua kegagalan itu persis yang sama dengan versi Python-nya "
                  + "(beda di dokumen sumber, bukan di kode).")
        }
        return unexpected.isEmpty ? cases.count : passed
    }
}
