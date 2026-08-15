"""Do any models specialise by COLONY SIZE rather than by colonies per plate?

The earlier comparison stratified by how many colonies a plate holds, which is
not the same question. A plate can hold 200 pinpoint colonies or 200 large
ones, and the failure modes are different: a detector that misses faint
sub-pixel blobs and a detector that merges touching large ones are opposite
problems, and an average over both hides each.

This re-runs every candidate -- including the six YOLO variants already cut --
per image, and buckets the plates by the median colony radius as a fraction of
the dish radius (the same measure sam_micro escalates on). If one of the cut
models is quietly the best at small colonies, that would be a reason to put it
back, and an aggregate MAE would never have shown it.

Usage:  python eval_by_colony_size.py
"""
import os
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
MODELS = ROOT / "coreml_models_block32"
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"
BRIGHT = ROOT / "bench_data/pca_images"

CANDIDATES = ["mac1", "sam_micro", "csrnet", "yolo_old", "yolo_new", "mac2",
              "dog_blend", "clahe", "lab_ab"]


def run(files, key):
    cmd = [str(SWIFT), str(MODELS)] + [str(f) for f in files] + ["--model", key]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
    res = {}
    for line in out.stdout.splitlines():
        p = line.strip().split("|")
        if len(p) >= 3 and p[1].lstrip("-").isdigit():
            res[p[0]] = int(p[1])
    return res


def main():
    sizes = {}
    for line in open(ROOT / "bench_data/colony_sizes.txt"):
        p = line.strip().split("|")
        if len(p) == 3 and p[2] != "-":
            sizes[p[0]] = float(p[2])

    bench = json.load(open(ROOT / "bench_data/pca_benchmark.json"))
    items = [(BRIGHT / f"{e['base']}.jpg", e["truth"]) for e in bench]
    items = [(f, t) for f, t in items if f.exists() and f.name in sizes]
    files = [f for f, _ in items]
    pct = np.array([sizes[f.name] for f in files])

    print(f"{len(items)} plat terang dengan ukuran koloni terukur")
    print(f"radius koloni / radius cawan: min {pct.min():.2f}%  median "
          f"{np.median(pct):.2f}%  maks {pct.max():.2f}%")
    qs = np.percentile(pct, [33, 67])
    print(f"batas tersil: {qs[0]:.2f}% dan {qs[1]:.2f}%")
    print(f"di bawah ambang eskalasi 1,5%: {(pct < 1.5).sum()} plat\n")

    buckets = [("kecil", pct <= qs[0]), ("sedang", (pct > qs[0]) & (pct <= qs[1])),
               ("besar", pct > qs[1])]
    truth = np.array([t for _, t in items], float)

    header = f"{'model':<11}" + "".join(f"{n + ' (n=' + str(int(m.sum())) + ')':>17}"
                                        for n, m in buckets)
    print(header + f"{'semua':>10}")
    print("-" * len(header + "     semua"))
    out = {}
    for key in CANDIDATES:
        counts = run(files, key)
        if len(counts) < len(files):
            print(f"{key:<11} tidak lengkap ({len(counts)}/{len(files)})")
            continue
        pred = np.array([counts[f.name] for f in files], float)
        err = np.abs(pred - truth)
        row = f"{key:<11}"
        out[key] = {}
        for name, mask in buckets:
            out[key][name] = float(err[mask].mean())
            row += f"{err[mask].mean():>17.2f}"
        row += f"{err.mean():>10.2f}"
        out[key]["semua"] = float(err.mean())
        print(row, flush=True)
        json.dump(out, open(ROOT / "bench_data/eval_by_colony_size.json", "w"), indent=1)

    print("\nterbaik per kelompok:")
    for name, _ in buckets + [("semua", None)]:
        best = min(out, key=lambda k: out[k][name])
        print(f"  {name:<8} {best} ({out[best][name]:.2f})")


if __name__ == "__main__":
    main()
