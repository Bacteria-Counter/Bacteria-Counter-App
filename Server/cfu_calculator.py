"""APHA 2002 microbial colony counting standards -- turning raw colony
counts into a reportable CFU/mL (or CFU/g) figure.

This is the layer *above* detection: the YOLO/SAM/CSRNet models in server.py
answer "how many colonies are on this one photo", while this module answers
"given plates at several dilutions (possibly in duplicate, possibly
spreading or ruined), what number does the lab actually report?".

Implements regulations 1-8 from "Panduan Perhitungan Koloni Mikroorganisme"
(APHA 2002):

  1. A plate with 25-250 colonies is the usable one -> CFU = count / dilution
  2. Duplicate plates at the same dilution are averaged (excluding any plate
     that is spreading or a lab accident)
  3. If two dilution levels are both in 25-250: if the higher dilution's CFU
     is >= 2x the lower dilution's, report the lower dilution's result;
     otherwise average the two
  4. If no plate is in 25-250 but some exceed 250: use the plate closest to
     250 and report as an estimate
  5. If every plate is under 25: use the least-diluted plate, report as an
     estimate
  6. If every plate is zero: report as "< 1/lowest dilution", estimated
  7. Plates over 250: if colony density exceeds 100/cm^2 the plate is beyond
     estimation, report as "> dish_area x 100 x highest dilution factor";
     otherwise it is treated as a countable estimate (regulation 4)
  8. Spreading colonies: a spreader covering >25% of the dish makes the plate
     a laboratory accident; chain-formation spreaders count as one colony
     (that judgement happens at counting time, not here)

Rounding follows the source document: every CFU figure is reported to 2
significant figures, and regulation 3's ratio/average are computed on those
rounded figures (not the raw products) -- matching the worked examples.

Note (from the document): these formulas assume the Total Plate Count *pour
plate* method with a 1 mL inoculum. For the *spread plate* method (0.1 mL
inoculum) the result is divided by 0.1, i.e. multiplied by 10.

Verified against all 23 worked examples in the source document
(test_cfu_calculator.py): 21 match exactly. The two that don't are samples
1116 and 1118, and in both the document skips showing its intermediate
arithmetic:

  1118: 287 colonies / 1e-2 = 28,700 -> 2.9 x 10^4 at two significant
        figures. The document reports 2.8 x 10^4.
  1116: averaging 2.3 x 10^4 and 2.6 x 10^4 gives exactly 2.45 x 10^4, a
        halfway case. The document reports 2.4 x 10^4; rounding half up
        gives 2.5 x 10^4.

Every example where the document *does* show its arithmetic rounds half up
(most explicitly sample 1009: "32500 = 3.3 x 10^4"), so that is the rule
implemented here, and those two look like slips in the document rather than
a different rule. Flagged rather than fudged -- if your lab's convention
differs, change round_sig()'s rounding mode in one place.
"""
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from math import floor, log10
from typing import Literal, Optional

# A 15x100 mm petri dish is ~56 cm^2 -- the document's default conversion
# factor for the regulation 7 density check.
DEFAULT_DISH_AREA_CM2 = 56.0

COUNTABLE_MIN = 25
COUNTABLE_MAX = 250
DENSITY_LIMIT_PER_CM2 = 100
SPREAD_LAB_ACCIDENT_FRACTION = 0.25

PlateStatus = Literal["ok", "spreading", "lab_accident", "tntc"]


@dataclass
class PlateReading:
    """One physical petri dish.

    dilution: the dilution *factor* as a fraction, e.g. 1e-2 for a 1:100
        dilution. Larger value = less diluted.
    count: colonies counted on this plate. Required when status is "ok".
    status:
        "ok"           - counted normally
        "spreading"    - has spreader colonies (see spread_fraction)
        "lab_accident" - unusable, excluded from all calculations
        "tntc"         - too numerous to count, no usable number
    spread_fraction: for "spreading" plates, how much of the dish the
        spreader covers (0.0-1.0). Over 25% the plate becomes a lab
        accident per regulation 8.
    """
    dilution: float
    count: Optional[int] = None
    status: PlateStatus = "ok"
    spread_fraction: Optional[float] = None

    def effective_status(self) -> PlateStatus:
        if self.status == "spreading" and self.spread_fraction is not None:
            if self.spread_fraction > SPREAD_LAB_ACCIDENT_FRACTION:
                return "lab_accident"
        return self.status


@dataclass
class CFUResult:
    value: Optional[float]          # CFU/mL, None when not reportable as a number
    display: str                    # what the lab writes down
    regulations: list[int] = field(default_factory=list)
    estimated: bool = False
    bounded: Optional[str] = None   # "<", ">" or None
    detail: str = ""                # human-readable reasoning


def round_sig(x: float, digits: int = 2) -> float:
    """Round to N significant figures, half away from zero (matching the
    document's worked examples, e.g. 32500 -> 33000)."""
    if x == 0:
        return 0.0
    exponent = floor(log10(abs(x)))
    # scaleb, not Decimal(10) ** n -- quantize keys off the *exponent* of its
    # argument, and Decimal(1000) has exponent 0, which would round to units.
    quant = Decimal(1).scaleb(exponent - digits + 1)
    return float(Decimal(x).quantize(quant, rounding=ROUND_HALF_UP))


def _round_half_up(x: float) -> int:
    return int(Decimal(x).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def format_cfu(value: float, estimated: bool = False, bounded: Optional[str] = None,
               unit: str = "CFU/ml") -> str:
    """Render like the source document: '2.3 x 10^4 CFU/ml (estimated)'."""
    if value == 0:
        base = f"0 {unit}"
    else:
        exponent = floor(log10(abs(value)))
        mantissa = value / (10 ** exponent)
        mantissa_str = f"{mantissa:.1f}".rstrip("0").rstrip(".")
        base = f"{mantissa_str} x 10^{exponent} {unit}"
    if bounded:
        base = f"{bounded} {base}"
    if estimated:
        base += " (estimated)"
    return base


def assess_plate_countability(count: int, dish_area_cm2: float = DEFAULT_DISH_AREA_CM2) -> dict:
    """Judge a single plate's count against the APHA reliability rules, so
    the app can say more than a bare number.

    The document is explicit that only 25-250 colonies on a plate gives a
    figure you can report as-is (regulation 1). Below that (regulation 5) or
    above it (regulation 4) the number is an estimate, and past ~100
    colonies/cm^2 (regulation 7) it stops being estimable at all.

    This deliberately does NOT change the count -- it labels how much the
    count can be trusted, which is the part a technician would otherwise
    have to remember by hand.
    """
    density = count / dish_area_cm2 if dish_area_cm2 > 0 else 0.0

    if count == 0:
        return {"status": "no_growth", "regulation": 6, "reliable": False,
                "densityPerCm2": 0.0,
                "advisory": "Tidak ada koloni terdeteksi."}

    if density > DENSITY_LIMIT_PER_CM2:
        return {"status": "tntc", "regulation": 7, "reliable": False,
                "densityPerCm2": round(density, 1),
                "advisory": (f"TNTC - kepadatan {density:.0f} koloni/cm^2 melebihi "
                             f"{DENSITY_LIMIT_PER_CM2}/cm^2. Angka ini di luar batas "
                             f"estimasi; gunakan pengenceran yang lebih tinggi.")}

    if count < COUNTABLE_MIN:
        return {"status": "below_range", "regulation": 5, "reliable": False,
                "densityPerCm2": round(density, 1),
                "advisory": (f"Di bawah {COUNTABLE_MIN} koloni - hanya boleh dilaporkan "
                             f"sebagai estimasi.")}

    if count > COUNTABLE_MAX:
        return {"status": "above_range", "regulation": 4, "reliable": False,
                "densityPerCm2": round(density, 1),
                "advisory": (f"Di atas {COUNTABLE_MAX} koloni - hanya boleh dilaporkan "
                             f"sebagai estimasi. Gunakan pengenceran lebih tinggi bila ada.")}

    return {"status": "countable", "regulation": 1, "reliable": True,
            "densityPerCm2": round(density, 1),
            "advisory": (f"Dalam rentang layak hitung {COUNTABLE_MIN}-{COUNTABLE_MAX} "
                         f"koloni.")}


@dataclass
class _Level:
    """All plates sharing one dilution, resolved down to a single number."""
    dilution: float
    count: Optional[int]
    status: PlateStatus
    n_plates_averaged: int = 0


def _resolve_duplicates(plates: list[PlateReading]) -> _Level:
    """Regulation 2: average duplicate plates at the same dilution, ignoring
    any that are spreading or lab accidents. If nothing usable remains, the
    level carries that status forward instead of a number."""
    dilution = plates[0].dilution
    usable = [p for p in plates if p.effective_status() == "ok" and p.count is not None]

    if usable:
        avg = sum(p.count for p in usable) / len(usable)
        return _Level(dilution=dilution, count=_round_half_up(avg), status="ok",
                      n_plates_averaged=len(usable))

    statuses = {p.effective_status() for p in plates}
    # A plate that is TNTC still tells us "very high", which regulation 7
    # can use; lab accidents and spreaders tell us nothing usable.
    if "tntc" in statuses:
        return _Level(dilution=dilution, count=None, status="tntc")
    if "spreading" in statuses:
        return _Level(dilution=dilution, count=None, status="spreading")
    return _Level(dilution=dilution, count=None, status="lab_accident")


def calculate_cfu(
    plates: list[PlateReading],
    dish_area_cm2: float = DEFAULT_DISH_AREA_CM2,
    method: Literal["pour", "spread"] = "pour",
    unit: str = "CFU/ml",
) -> CFUResult:
    """Apply APHA 2002 regulations 1-8 to a set of plates from one sample."""
    if not plates:
        raise ValueError("No plates provided")

    # Spread plate uses a 0.1 mL inoculum, so its result is divided by 0.1.
    method_factor = 10.0 if method == "spread" else 1.0

    by_dilution: dict[float, list[PlateReading]] = {}
    for p in plates:
        by_dilution.setdefault(p.dilution, []).append(p)

    # Sorted least-diluted first (1e-2 before 1e-3).
    levels = [_resolve_duplicates(group)
              for _, group in sorted(by_dilution.items(), key=lambda kv: -kv[0])]

    used_dup_rule = any(len(g) > 1 for g in by_dilution.values())

    def cfu_of(level: _Level) -> float:
        return round_sig(level.count / level.dilution * method_factor)

    countable = [lv for lv in levels if lv.status == "ok" and lv.count is not None]
    in_range = [lv for lv in countable if COUNTABLE_MIN <= lv.count <= COUNTABLE_MAX]

    regs: list[int] = []
    if used_dup_rule:
        regs.append(2)
    if any(lv.status in ("spreading", "lab_accident") for lv in levels) or \
       any(p.status in ("spreading", "lab_accident") for p in plates):
        regs.append(8)

    # --- Nothing usable at all ---
    if not countable and all(lv.status != "tntc" for lv in levels):
        statuses = {lv.status for lv in levels}
        if statuses == {"spreading"} or "spreading" in statuses and "lab_accident" in statuses:
            return CFUResult(None, "Spreading", sorted(set(regs + [8])),
                             detail="Every plate is a spreader -- no countable plate.")
        return CFUResult(None, "Lab accident", sorted(set(regs + [8])),
                         detail="Every plate is a laboratory accident -- nothing to report.")

    # --- Regulation 6: no colonies anywhere ---
    if countable and all(lv.count == 0 for lv in countable):
        lowest = max(lv.dilution for lv in countable)  # least diluted
        bound = 1.0 / lowest * method_factor
        regs.append(6)
        return CFUResult(bound, format_cfu(bound, estimated=True, bounded="<", unit=unit),
                         sorted(set(regs)), estimated=True, bounded="<",
                         detail="No colonies at any dilution -- reported as less than the "
                                "lowest dilution factor.")

    # --- Regulation 7: density beyond estimation ---
    for lv in countable:
        if lv.count > COUNTABLE_MAX and (lv.count / dish_area_cm2) > DENSITY_LIMIT_PER_CM2:
            highest = min(l.dilution for l in levels)  # most diluted
            bound = dish_area_cm2 * DENSITY_LIMIT_PER_CM2 / highest * method_factor
            regs.append(7)
            return CFUResult(bound, format_cfu(bound, estimated=True, bounded=">", unit=unit),
                             sorted(set(regs)), estimated=True, bounded=">",
                             detail=f"{lv.count} colonies on {dish_area_cm2:g} cm^2 exceeds "
                                    f"{DENSITY_LIMIT_PER_CM2}/cm^2 -- beyond estimation.")

    # --- Regulation 1: exactly one level in 25-250 ---
    if len(in_range) == 1:
        lv = in_range[0]
        value = cfu_of(lv)
        regs.append(1)
        return CFUResult(value, format_cfu(value, unit=unit), sorted(set(regs)),
                         detail=f"{lv.count} colonies at 1:{1/lv.dilution:g} is within "
                                f"{COUNTABLE_MIN}-{COUNTABLE_MAX}.")

    # --- Regulation 3: two levels in 25-250 ---
    if len(in_range) >= 2:
        lower, higher = in_range[0], in_range[1]  # lower dilution first
        cfu_lower, cfu_higher = cfu_of(lower), cfu_of(higher)
        ratio = cfu_higher / cfu_lower if cfu_lower else float("inf")
        regs.append(3)
        if ratio >= 2:
            return CFUResult(cfu_lower, format_cfu(cfu_lower, unit=unit), sorted(set(regs)),
                             detail=f"Ratio {ratio:.1f} >= 2 -- using the lower dilution's "
                                    f"result.")
        averaged = round_sig((cfu_lower + cfu_higher) / 2)
        return CFUResult(averaged, format_cfu(averaged, unit=unit), sorted(set(regs)),
                         detail=f"Ratio {ratio:.1f} < 2 -- averaging both dilution levels.")

    # --- Regulation 4: nothing in range, but something over 250 ---
    over = [lv for lv in countable if lv.count > COUNTABLE_MAX]
    if over:
        closest = min(over, key=lambda lv: abs(lv.count - COUNTABLE_MAX))
        value = cfu_of(closest)
        regs.append(4)
        if any(lv.status == "tntc" for lv in levels) or closest.count > COUNTABLE_MAX:
            regs.append(7)
        return CFUResult(value, format_cfu(value, estimated=True, unit=unit),
                         sorted(set(regs)), estimated=True,
                         detail=f"No plate in {COUNTABLE_MIN}-{COUNTABLE_MAX}; using the plate "
                                f"closest to {COUNTABLE_MAX} ({closest.count}) as an estimate.")

    # --- Regulation 5: everything under 25 ---
    under = [lv for lv in countable if lv.count < COUNTABLE_MIN]
    if under:
        least_diluted = max(under, key=lambda lv: lv.dilution)
        value = cfu_of(least_diluted)
        regs.append(5)
        return CFUResult(value, format_cfu(value, estimated=True, unit=unit),
                         sorted(set(regs)), estimated=True,
                         detail=f"All plates under {COUNTABLE_MIN} -- using the least-diluted "
                                f"plate ({least_diluted.count} colonies) as an estimate.")

    raise ValueError(f"Unhandled plate combination: {levels}")
