"""Step 5 verification: what int8 weights actually cost, in colonies.

Deliberately NOT a Python-versus-Swift comparison. Steps 2 and 3 already
established that the port agrees with the reference; asking that question again
with quantised weights would answer it twice and the quantisation question not
at all. Here the code is identical on both sides and the ONLY variable is the
weights -- float32 models against int8 models, same binary, same images.

Reported against ground truth as well as against each other, because those can
disagree: quantisation that shifts every count by one in the same direction is
a different problem from quantisation that scatters them.

Usage:  python verify_5.py [model ...]
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
SET = ROOT / "bench_data/verify_set"
FLOAT_MODELS = ROOT / "coreml_models"
INT8_MODELS = Path(os.environ.get("AGARSCOPE_QUANT_MODELS",
                                  str(ROOT / "coreml_models_block32")))
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"

MODELS = ["sam_tuned", "yolo_old", "yolo_new", "mac1", "mac2",
          "dog_blend", "clahe", "lab_ab", "csrnet"]


def swift_counts(model_dir, files, key):
    cmd = [str(SWIFT), str(model_dir)] + [str(SET / f) for f in files] + ["--model", key]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
    counts = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3 and parts[1].lstrip("-").isdigit():
            counts[parts[0]] = int(parts[1])
    if not counts:
        print("  swift stderr:", out.stderr[-400:])
    return counts


def main():
    manifest = json.load(open(ROOT / "bench_data/verify_manifest.json"))
    bench = {e["base"]: e["truth"]
             for e in json.load(open(ROOT / "bench_data/pca_benchmark.json"))}
    keys = sys.argv[1:] or MODELS
    out = {}

    for key in keys:
        print(f"\n=== {key} ===", flush=True)
        out[key] = {}
        for group in ("holdout", "empty", "lab"):
            items = [(base, fn) for g, base, fn in manifest if g == group]
            files = [fn for _, fn in items]
            # sam_micro's escalation is what makes the lab photos interesting;
            # for every other model the lab group is just three more plates.
            k = "sam_micro" if (group == "lab" and key == "sam_tuned") else key
            f32 = swift_counts(FLOAT_MODELS, files, k)
            i8 = swift_counts(INT8_MODELS, files, k)
            rows, truths = [], []
            for base, fn in items:
                if fn not in f32 or fn not in i8:
                    continue
                rows.append((f32[fn], i8[fn]))
                if base in bench:
                    truths.append(bench[base])
            if not rows:
                continue
            diff = [b - a for a, b in rows]
            same = sum(1 for a, b in rows if a == b)
            line = f"  {group:<8} n={len(rows):<3}"
            if len(truths) == len(rows):
                mf = sum(abs(a - t) for (a, _), t in zip(rows, truths)) / len(rows)
                mi = sum(abs(b - t) for (_, b), t in zip(rows, truths)) / len(rows)
                line += f" MAE f32 {mf:6.2f} | int8 {mi:6.2f}"
            else:
                line += (f" total f32 {sum(a for a, _ in rows):5d} | "
                         f"int8 {sum(b for _, b in rows):5d}")
            line += (f"  identik {same:>3}/{len(rows)}  bias {np.mean(diff):+.2f}"
                     f"  maks |{max(abs(d) for d in diff)}|")
            print(line, flush=True)
            out[key][group] = {"n": len(rows), "same": same,
                               "bias": float(np.mean(diff)),
                               "max": int(max(abs(d) for d in diff))}
            json.dump(out, open(ROOT / "bench_data/verify_5_result.json", "w"), indent=1)

    print("\nVERIFIKASI 5 SELESAI")


if __name__ == "__main__":
    main()
