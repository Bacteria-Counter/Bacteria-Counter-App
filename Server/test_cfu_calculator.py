"""Verify cfu_calculator.py against all 23 worked examples in
'Panduan Perhitungan Koloni Mikroorganisme Rev 2' (APHA 2002).

Run:  .venv/bin/python3 test_cfu_calculator.py

Expect 21/23. Samples 1116 and 1118 are known discrepancies in the source
document itself -- see the module docstring in cfu_calculator.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cfu_calculator import PlateReading as P, calculate_cfu

D2, D3 = 1e-2, 1e-3

def ok(d, c):   return P(dilution=d, count=c)
def spread(d, c=None, frac=None): return P(dilution=d, count=c, status="spreading", spread_fraction=frac)
def labacc(d):  return P(dilution=d, status="lab_accident")
def tntc(d):    return P(dilution=d, status="tntc")

# (sample, plates, expected numeric value or None, expected display keyword)
CASES = [
    ("1001", [ok(D2, 234), ok(D3, 23)],                       23000,   None),
    ("1002", [ok(D2, 243), ok(D3, 34)],                       29000,   None),
    ("1003", [ok(D2, 140), ok(D3, 32)],                       14000,   None),
    ("1004", [spread(D2), ok(D3, 31)],                        31000,   None),
    ("1005", [ok(D2, 0), ok(D3, 0)],                          100,     "<"),
    ("1006", [tntc(D2), ok(D3, 7150)],                        5600000, ">"),
    ("1007", [ok(D2, 18), ok(D3, 2)],                         1800,    "est"),
    ("1008", [spread(D2), spread(D3)],                        None,    "Spreading"),
    ("1009", [ok(D2, 325), ok(D3, 20)],                       33000,   "est"),
    ("1010", [spread(D2, 27, 0.30), spread(D3, 215, 0.30)],   None,    "Lab accident"),
    ("1011", [ok(D2, 305), ok(D3, 42)],                       42000,   None),
    ("1012", [ok(D2, 243), labacc(D3)],                       24000,   None),
    ("1013", [tntc(D2), ok(D3, 840)],                         840000,  "est"),
    ("1111", [ok(D2, 228), ok(D2, 240), ok(D3, 28), ok(D3, 26)], 25000, None),
    ("1112", [ok(D2, 175), ok(D2, 208), ok(D3, 16), ok(D3, 17)], 19000, None),
    ("1113", [ok(D2, 239), ok(D2, 328), ok(D3, 16), ok(D3, 19)], 28000, None),
    ("1114", [ok(D2, 275), ok(D2, 280), ok(D3, 24), ok(D3, 35)], 30000, None),
    ("1115", [ok(D2, 138), ok(D2, 162), ok(D3, 42), ok(D3, 30)], 15000, None),
    ("1116", [ok(D2, 240), ok(D2, 228), ok(D3, 28), ok(D3, 23)], 24000, None),
    ("1117", [ok(D2, 224), ok(D2, 180), ok(D3, 28), spread(D3)], 24000, None),
    ("1118", [ok(D2, 287), labacc(D2), ok(D3, 23), ok(D3, 19)], 28000, "est"),
    ("1119", [ok(D2, 18), ok(D2, 16), ok(D3, 2), ok(D3, 0)],    1700,  "est"),
    ("1120", [ok(D2, 0), ok(D2, 0), ok(D3, 0), ok(D3, 0)],      100,   "<"),
]

passed, failed = 0, []
print(f"{'sample':<8}{'expected':>12}  {'got':>12}   regs        display")
print("-" * 78)
for sample, plates, expected, keyword in CASES:
    try:
        r = calculate_cfu(plates)
    except Exception as e:
        failed.append((sample, expected, f"EXCEPTION: {e}"))
        print(f"{sample:<8}{str(expected):>12}  {'ERROR':>12}   -           {e}")
        continue

    got = r.value
    match = False
    if expected is None:
        match = r.value is None and keyword and keyword.lower() in r.display.lower()
    else:
        match = got is not None and abs(got - expected) < max(1e-6, expected * 1e-9)
        if keyword == "<":
            match = match and r.bounded == "<"
        elif keyword == ">":
            match = match and r.bounded == ">"
        elif keyword == "est":
            match = match and r.estimated

    flag = "OK " if match else "FAIL"
    if match:
        passed += 1
    else:
        failed.append((sample, expected, got))
    print(f"{sample:<8}{str(expected):>12}  {str(got):>12}   {str(r.regulations):<11} {r.display}   {flag}")

print("-" * 78)
print(f"PASSED {passed}/{len(CASES)}")
if failed:
    print("\nFAILURES:")
    for s, exp, got in failed:
        print(f"  {s}: expected {exp}, got {got}")
