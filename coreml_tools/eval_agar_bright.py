"""The same comparison again, on AGAR's bright-background test images.

PCA and AGAR disagree about almost everything except the colour of the plate:
different lab, different camera, different agar, 3688x4000 instead of
2560x1922. If a model's ranking survives both, it is a property of the model;
if it flips, the PCA ranking was a property of one photographer.

Only the 99 BRIGHT images from the held-out 700-image test split are used --
the dark majority is a domain this app will never see.

One asymmetry to keep in mind when reading the numbers: AGAR is in-domain for
the YOLO family, which was trained on AGAR train-split images (yolo_old
entirely, the rest with 600 real AGAR images mixed into their fine-tuning),
while FastSAM is zero-shot and has never been trained on anything here. The
test split itself was deliberately never touched during training, so this is
not leakage -- but it does mean AGAR flatters YOLO in a way PCA did not.

Usage:  python eval_agar_bright.py
"""
import os
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
MODELS = ROOT / "coreml_models_block32"
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"
MANIFEST = Path(os.environ.get("AGAR_MANIFEST", "/Users/satriabaladewaharahap/Downloads/bacteriacounter/kaggle_manifest.csv"))
AGAR = Path(os.environ.get("AGAR_DATASET", "/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset"))

CANDIDATES = ["sam_micro", "csrnet", "mac1", "dog_blend", "yolo_old", "yolo_new",
              "mac2", "clahe", "lab_ab"]


def run(files, key, extra=None):
    cmd = [str(SWIFT), str(MODELS)] + [str(f) for f in files]
    cmd += extra or ["--model", key]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
    res = {}
    for line in out.stdout.splitlines():
        p = line.strip().split("|")
        if len(p) >= 3 and p[1].lstrip("-").isdigit():
            res[p[0]] = p
    if not res:
        print("  stderr:", out.stderr[-300:])
    return res


def main():
    rows = [r for r in csv.DictReader(open(MANIFEST))
            if r["split"] == "test" and r["background"] == "bright"]
    items = [(AGAR / r["image_filename"], int(r["cfu_count"])) for r in rows]
    items = [(f, c) for f, c in items if f.exists()]
    files = [f for f, _ in items]
    truth = np.array([c for _, c in items], float)
    print(f"AGAR bright test: {len(items)} gambar, koloni median {np.median(truth):.0f}\n",
          flush=True)

    # Colony size first, so the same size stratification as the PCA run can be
    # applied -- and so the two benchmarks can be compared on that axis at all.
    sizes = {}
    cmd = [str(SWIFT), "--colony-size", str(MODELS)] + [str(f) for f in files]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
    for line in out.stdout.splitlines():
        p = line.strip().split("|")
        if len(p) == 3 and p[2] != "-":
            sizes[p[0]] = float(p[2])
    pct = np.array([sizes.get(f.name, np.nan) for f in files])
    ok = ~np.isnan(pct)
    if ok.sum():
        print(f"radius koloni / radius cawan: min {np.nanmin(pct):.2f}%  median "
              f"{np.nanmedian(pct):.2f}%  maks {np.nanmax(pct):.2f}%  "
              f"(terukur di {ok.sum()}/{len(files)})\n", flush=True)
    qs = np.nanpercentile(pct, [33, 67]) if ok.sum() else [0, 0]
    buckets = [("kecil", ok & (pct <= qs[0])),
               ("sedang", ok & (pct > qs[0]) & (pct <= qs[1])),
               ("besar", ok & (pct > qs[1]))]

    header = (f"{'model':<11}{'MAE semua':>11}{'bias':>9}"
              + "".join(f"{n + '(' + str(int(m.sum())) + ')':>13}" for n, m in buckets))
    print(header)
    print("-" * len(header))
    out_json = {}
    for key in CANDIDATES:
        res = run(files, key)
        if len(res) < len(files):
            print(f"{key:<11} tidak lengkap ({len(res)}/{len(files)})"); continue
        pred = np.array([int(res[f.name][1]) for f in files], float)
        err = np.abs(pred - truth)
        row = f"{key:<11}{err.mean():>11.2f}{(pred - truth).mean():>+9.2f}"
        out_json[key] = {"mae": float(err.mean()), "bias": float((pred - truth).mean())}
        for name, mask in buckets:
            v = err[mask].mean() if mask.sum() else float("nan")
            out_json[key][name] = float(v)
            row += f"{v:>13.2f}"
        print(row, flush=True)
        json.dump(out_json, open(ROOT / "bench_data/eval_agar_bright.json", "w"), indent=1)

    print("\nterbaik:")
    for name in ["mae"] + [n for n, _ in buckets]:
        best = min(out_json, key=lambda k: out_json[k][name])
        print(f"  {name:<8} {best} ({out_json[best][name]:.2f})")


if __name__ == "__main__":
    main()
