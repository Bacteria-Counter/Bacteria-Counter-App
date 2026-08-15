"""Compare the YOLO variants across every bright-background plate we have.

The question this answers is which models earn their place in the app. Seven
YOLO variants is a lot of picker entries for a technician to reason about, and
several were added to be compared rather than to be kept.

Populations, because a model can win one and lose another:
  bright 126 -- the full PCA benchmark with ground truth. Accuracy.
  empty 34   -- real empty plates from the user's own lab. False positives,
                which is the failure that matters for a sterility check: a
                missed colony is an underestimate, an invented one on a
                negative control is a wrong conclusion.
  lab 3      -- the actual target domain. No ground truth exists, so counts
                are reported and not scored.

Checked before running: none of the seven checkpoints lists a PCA path in its
`train_args.data`, so all 126 images are honest test data for every one of
them. (Directories named `mac1+pca` on this machine are a separate, rejected
experiment.)

Everything runs through the SHIPPED int8 models and the same Swift binary, so
the only thing varying between rows is the checkpoint.

Usage:  python eval_yolo.py [model ...]
"""
import os
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
MODELS = ROOT / "coreml_models_block32"
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"
BRIGHT = ROOT / "bench_data/pca_images"
SET = ROOT / "bench_data/verify_set"

YOLO = ["yolo_old", "yolo_new", "mac1", "mac2", "dog_blend", "clahe", "lab_ab"]
# Not YOLO, included so a "keep only one" recommendation is made against what
# the app actually offers rather than against the YOLO shortlist alone.
REFERENCE = ["sam_tuned", "csrnet"]


def run(files, key):
    cmd = [str(SWIFT), str(MODELS)] + [str(f) for f in files] + ["--model", key]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
    counts = {}
    for line in out.stdout.splitlines():
        p = line.strip().split("|")
        if len(p) >= 3 and p[1].lstrip("-").isdigit():
            counts[p[0]] = int(p[1])
    if not counts:
        print("  swift stderr:", out.stderr[-300:])
    return counts


def main():
    bench = json.load(open(ROOT / "bench_data/pca_benchmark.json"))
    bright = [(e["base"], BRIGHT / f"{e['base']}.jpg", e["truth"]) for e in bench]
    bright = [b for b in bright if b[1].exists()]
    empty = sorted(SET.glob("e_*.png"))
    lab = sorted(SET.glob("l_*.png"))
    print(f"terang {len(bright)} | kosong {len(empty)} | lab {len(lab)}\n", flush=True)

    keys = sys.argv[1:] or YOLO + REFERENCE
    out_path = ROOT / "bench_data/eval_yolo_result.json"
    results = json.load(open(out_path)) if out_path.exists() else {}

    for key in keys:
        r = {}
        counts = run([f for _, f, _ in bright], key)
        got = [(t, counts[f.name]) for _, f, t in bright if f.name in counts]
        truth = np.array([t for t, _ in got], float)
        pred = np.array([p for _, p in got], float)
        err = pred - truth
        r["bright"] = {
            "n": len(got),
            "mae": float(np.abs(err).mean()),
            "bias": float(err.mean()),
            "median_abs": float(np.median(np.abs(err))),
            # sMAPE tracks proportional error, which MAE hides: being 10 out on
            # a plate of 12 is a different failure from 10 out on a plate of 240.
            "smape": float(np.mean(2 * np.abs(err) / np.maximum(truth + pred, 1)) * 100),
            "within_2": int((np.abs(err) <= 2).sum()),
        }
        # APHA only lets you report 25-250 directly, so accuracy inside that
        # window is worth more than accuracy on a plate you would re-plate.
        for name, mask in [("jarang_<25", truth < 25),
                           ("layak_25-250", (truth >= 25) & (truth <= 250))]:
            if mask.sum():
                r["bright"][name] = {"n": int(mask.sum()),
                                     "mae": float(np.abs(err[mask]).mean()),
                                     "bias": float(err[mask].mean())}

        ec = run(empty, key)
        vals = [ec[f.name] for f in empty if f.name in ec]
        r["empty"] = {"n": len(vals), "total_fp": int(sum(vals)),
                      "plates_with_fp": int(sum(1 for v in vals if v > 0)),
                      "worst": int(max(vals)) if vals else 0}

        lc = run(lab, key)
        r["lab"] = {f.name.replace("l_", "").replace(".png", ""): lc[f.name]
                    for f in lab if f.name in lc}

        b = r["bright"]
        print(f"{key:<11} MAE {b['mae']:6.2f}  bias {b['bias']:+6.2f}  "
              f"sMAPE {b['smape']:5.1f}%  dalam+-2 {b['within_2']:>3}/{b['n']}  |  "
              f"kosong {r['empty']['total_fp']:>3} palsu di "
              f"{r['empty']['plates_with_fp']}/{r['empty']['n']} cawan  |  "
              f"lab {r['lab']}", flush=True)
        results[key] = r
        json.dump(results, open(out_path, "w"), indent=1)

    print("\nEVALUASI SELESAI")


if __name__ == "__main__":
    main()
