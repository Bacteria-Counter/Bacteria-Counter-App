"""Step 2b verification: the Swift port against the Python Core ML path.

Both sides read the SAME pre-generated PNGs. That matters twice over: the
earlier 2a comparison was invalidated by one side seeing JPEG-compressed pixels
and the other raw ones, and CoreGraphics' JPEG decoder was separately measured
to differ from OpenCV's by 1-4 units per pixel. PNG removes both confounds, so
what remains is genuinely Swift versus Python.

Ground truth here is NOT the original benchmark's. The scratchpad holding it was
cleared by the OS and it was rebuilt from the YOLO labels in pca_light.zip,
which count roughly 3 colonies lower per plate than the mask-derived originals.
MAE figures are therefore not comparable to anything recorded before 15 August;
the two paths are compared against each other, measured the same way.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
SET = ROOT / "bench_data/verify_set"
# Overridable so the quantised directory can be measured through the exact
# same harness -- a separate copy of this script would be a second variable.
MODELS = Path(os.environ.get("AGARSCOPE_MODELS",
                             os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius") + "/coreml_models"))
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"
# verify_2a and coreml_fastsam used to live in a session scratchpad under
# /private/tmp, which macOS cleared mid-project once already. They are kept
# here now; the old location stays on the path only as a fallback.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "Bacteria-Counter-App/Server"))

import cv2
import numpy as np
import verify_2a as V
from coreml_fastsam import run_coreml


def swift_counts(files, micro=False):
    """One process for the whole batch -- model load dominates otherwise."""
    cmd = [str(SWIFT), str(MODELS)] + [str(SET / f) for f in files]
    if micro:
        cmd.append("--micro")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    counts = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3:
            counts[parts[0]] = int(parts[1])
    if not counts:
        print("  swift stderr:", out.stderr[-400:])
    return counts


def python_count(path, micro=False):
    img = cv2.imread(str(path))
    imgsz, dish = V.adaptive(img)
    ci = V.clahe(img)
    ml, key = V.coreml_for(imgsz)
    kept = V.filter_masks(run_coreml(ml, ci, key), img, dish)
    if micro and dish is not None and len(kept) >= V.ESCALATE_MIN_COUNT:
        med = float(np.median([k["area"] for k in kept]))
        if (med / np.pi) ** 0.5 / dish[2] * 100 < V.ESCALATE_MAX_PCT:
            m2, k2 = V.coreml_for(V.ESCALATE_IMGSZ)
            kept = V.filter_masks(run_coreml(m2, ci, k2), img, dish)
    return len(kept)


def report(name, rows, truths=None):
    diff = [b - a for a, b in rows]
    same = sum(1 for a, b in rows if a == b)
    print(f"\n=== {name} (n={len(rows)}) ===")
    if truths:
        mp = sum(abs(a - t) for (a, _), t in zip(rows, truths)) / len(rows)
        ms = sum(abs(b - t) for (_, b), t in zip(rows, truths)) / len(rows)
        print(f"  MAE  Python {mp:.2f} | Swift {ms:.2f} | selisih {ms-mp:+.2f}")
    print(f"  total Python {sum(a for a, _ in rows)} | Swift {sum(b for _, b in rows)}")
    print(f"  hitungan identik {same}/{len(rows)}")
    print(f"  beda: rata-rata {np.mean(diff):+.2f}, median {np.median(diff):+.1f}, "
          f"maks |{max(abs(d) for d in diff)}|")
    within1 = sum(1 for d in diff if abs(d) <= 1)
    print(f"  dalam +/-1: {within1}/{len(rows)}")
    return {"same": same, "mean": float(np.mean(diff)),
            "max": int(max(abs(d) for d in diff)), "within1": within1}


def main():
    manifest = json.load(open(ROOT / "bench_data/verify_manifest.json"))
    bench = {e["base"]: e["truth"]
             for e in json.load(open(ROOT / "bench_data/pca_benchmark.json"))}
    out = {}

    only = sys.argv[1:] or None
    for group, micro in (("holdout", False), ("empty", False), ("lab", True)):
        if only and group not in only:
            continue
        items = [(base, fn) for g, base, fn in manifest if g == group]
        print(f"\nmenjalankan {group}: {len(items)} gambar (micro={micro})", flush=True)
        sw = swift_counts([fn for _, fn in items], micro=micro)
        rows, truths, names = [], [], []
        for base, fn in items:
            if fn not in sw:
                print(f"  swift tidak melaporkan {fn}")
                continue
            py = python_count(SET / fn, micro=micro)
            rows.append((py, sw[fn]))
            names.append(base)
            if group == "holdout" and base in bench:
                truths.append(bench[base])
        if group == "lab":
            for (py, s), n in zip(rows, names):
                print(f"  {n:<8} Python {py:>4} | Swift {s:>4}  ({s-py:+d})")
        out[group] = report(group, rows, truths if len(truths) == len(rows) else None)
        worst = sorted(zip(names, rows), key=lambda x: -abs(x[1][1] - x[1][0]))[:3]
        print("  selisih terbesar: " + ", ".join(f"{n[:20]} {a}->{b}" for n, (a, b) in worst))

    json.dump(out, open(ROOT / "bench_data/verify_2b_result.json", "w"), indent=1)
    print("\nVERIFIKASI 2B SELESAI")


if __name__ == "__main__":
    main()
