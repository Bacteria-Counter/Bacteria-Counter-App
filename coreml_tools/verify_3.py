"""Step 3 verification: the Swift YOLO and CSRNet ports against Python Core ML.

Same shape as verify_2b.py, and for the same reasons. Both sides read the same
pre-generated PNGs, so neither JPEG decoding nor a rescale can creep in as a
second variable; and both sides run Core ML, so this measures the port and not
the conversion, which step 1 already measured separately.

Two populations, because they fail differently:
  holdout 42 -- accuracy, against ground truth
  empty 34   -- false positives, where a regression is most damaging

The lab photos are excluded here on purpose. They are the one population with
no measured ground truth, and the accepted rescale limitation from step 2 lands
squarely on them, so a difference there would be uninterpretable.

Usage:  python verify_3.py [model ...]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
SET = ROOT / "bench_data/verify_set"
# Overridable so the quantised directory can be measured through the exact
# same harness -- a separate copy of this script would be a second variable.
MODELS = Path(os.environ.get("AGARSCOPE_MODELS",
                             os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius") + "/coreml_models"))
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "Bacteria-Counter-App/Server"))

import coremltools as ct
import cv2
import fastsam_colony_count as FC
from coreml_yolo import nearest_size, run_csrnet_coreml, run_yolo_coreml

YOLO_MODELS = ["yolo_old", "yolo_new", "mac1", "mac2", "dog_blend", "clahe", "lab_ab"]
_cache = {}


def make_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def make_dog_blend(img, sigma1=2, sigma2=12, strength=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dog = cv2.GaussianBlur(gray, (0, 0), sigma1) - cv2.GaussianBlur(gray, (0, 0), sigma2)
    return np.clip(img.astype(np.float32) + (strength * dog)[..., None], 0, 255).astype(np.uint8)


def make_lab_ab(img):
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.merge([cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX),
                      cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX),
                      np.full_like(a, 128)])


PREPROCESS = {"dog_blend": make_dog_blend, "clahe": make_clahe, "lab_ab": make_lab_ab}


def model_for(key, size):
    name = f"{key}_{size}" if key != "csrnet" else "csrnet"
    if name not in _cache:
        _cache[name] = ct.models.MLModel(str(MODELS / f"{name}.mlpackage"))
    return _cache[name]


def adaptive(img):
    """server.py's adaptive_imgsz, measured on the ORIGINAL frame."""
    h, w = img.shape[:2]
    d = FC.find_dish_circle(img)
    if d is None:
        return 1536
    return max(1280, min(3200, int(round((581 * max(h, w) / d[2]) / 32) * 32)))


def python_count(path, key):
    img = cv2.imread(str(path))
    if key == "csrnet":
        return int(round(float(run_csrnet_coreml(model_for("csrnet", 0), img).sum())))
    size = nearest_size(adaptive(img))
    src = PREPROCESS[key](img) if key in PREPROCESS else img
    return len(run_yolo_coreml(model_for(key, size), src, size))


def swift_counts(files, key):
    """One process for the whole batch -- model load dominates otherwise."""
    cmd = [str(SWIFT), str(MODELS)] + [str(SET / f) for f in files] + ["--model", key]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    counts = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and parts[1].lstrip("-").isdigit():
            counts[parts[0]] = int(parts[1])
    if not counts:
        print("  swift stderr:", out.stderr[-400:])
    return counts


def report(name, rows, truths=None):
    diff = [b - a for a, b in rows]
    same = sum(1 for a, b in rows if a == b)
    line = f"  {name:<9} n={len(rows):<3}"
    if truths:
        mp = sum(abs(a - t) for (a, _), t in zip(rows, truths)) / len(rows)
        ms = sum(abs(b - t) for (_, b), t in zip(rows, truths)) / len(rows)
        line += f" MAE Py {mp:6.2f} | Sw {ms:6.2f}"
    else:
        line += f" total Py {sum(a for a, _ in rows):5d} | Sw {sum(b for _, b in rows):5d}"
    within1 = sum(1 for d in diff if abs(d) <= 1)
    line += (f"  identik {same:>3}/{len(rows)}  +/-1 {within1:>3}/{len(rows)}"
             f"  bias {np.mean(diff):+.2f}  maks |{max(abs(d) for d in diff)}|")
    print(line, flush=True)
    return {"n": len(rows), "same": same, "within1": within1,
            "bias": float(np.mean(diff)), "max": int(max(abs(d) for d in diff))}


def main():
    manifest = json.load(open(ROOT / "bench_data/verify_manifest.json"))
    bench = {e["base"]: e["truth"]
             for e in json.load(open(ROOT / "bench_data/pca_benchmark.json"))}
    keys = sys.argv[1:] or YOLO_MODELS + ["csrnet"]
    out_path = ROOT / "bench_data/verify_3_result.json"
    out = json.load(open(out_path)) if out_path.exists() else {}

    for key in keys:
        print(f"\n=== {key} ===", flush=True)
        out.setdefault(key, {})
        for group in ("holdout", "empty"):
            items = [(base, fn) for g, base, fn in manifest if g == group]
            sw = swift_counts([fn for _, fn in items], key)
            rows, truths = [], []
            for base, fn in items:
                if fn not in sw:
                    print(f"  swift tidak melaporkan {fn}")
                    continue
                rows.append((python_count(SET / fn, key), sw[fn]))
                if group == "holdout" and base in bench:
                    truths.append(bench[base])
            if not rows:
                continue
            out[key][group] = report(group, rows,
                                     truths if len(truths) == len(rows) else None)
            json.dump(out, open(out_path, "w"), indent=1)

    print("\nVERIFIKASI 3 SELESAI")


if __name__ == "__main__":
    main()
