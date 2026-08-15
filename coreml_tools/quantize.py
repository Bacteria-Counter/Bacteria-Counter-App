"""Step 5: quantise the Core ML weights and measure what it costs.

Run before deciding how the models ship, not after. 627 MB is the number that
makes bundling awkward; if int8 weights bring that to a fifth of it without
moving the counts, the packaging decision stops being a trade-off at all.

Weights only, per-channel, symmetric. Activations are left alone deliberately:
quantising those needs calibration data and changes numerics everywhere,
whereas weight-only quantisation is a pure size play whose cost can be measured
against the counts we already have.

Usage:  python quantize.py [--mode int8|palette6|palette4] [--out DIR]
"""
import os
import argparse
import json
import shutil
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
SRC = ROOT / "coreml_models"


def size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def config_for(mode: str):
    if mode == "block32":
        # One scale per block of 32 input channels instead of one per output
        # channel. This is the only scheme measured here that leaves the counts
        # alone -- see the note on `retarget` below and the sweep recorded in
        # AgarScopeKit/README.md.
        return cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_block",
            block_size=32))
    if mode == "int8":
        return cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    nbits = int(mode.replace("palette", ""))
    return cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(
        mode="kmeans", nbits=nbits, granularity="per_grouped_channel", group_size=16))


def retarget(m, version: int = 9):
    """Re-stamp the model's spec version so the iOS18 compression passes apply.

    Ultralytics' CoreML exporter does not expose minimum_deployment_target, so
    every YOLO and FastSAM file here came out at spec version 6 -- and both
    per_block quantisation and per_grouped_channel palettisation refuse to run
    below version 9. Re-exporting all 34 would mean redoing step 1.

    Raising the version is safe in this direction: the ops were emitted for an
    older spec and every one of them is still valid in the newer one, which is
    additive. It does raise the minimum OS to macOS 15; this app targets
    macOS 26.5, so that costs nothing here.
    """
    spec = m.get_spec()
    if spec.specificationVersion >= version:
        return m
    spec.specificationVersion = version
    return ct.models.MLModel(spec, weights_dir=m.weights_dir, skip_model_load=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="block32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / f"coreml_models_{args.mode}"
    out_dir.mkdir(exist_ok=True)
    config = config_for(args.mode)

    packages = sorted(p for p in SRC.iterdir() if p.suffix == ".mlpackage")
    report = []
    for i, p in enumerate(packages, 1):
        dst = out_dir / p.name
        if dst.exists():
            report.append({"model": p.stem, "before_mb": round(size_mb(p), 1),
                           "after_mb": round(size_mb(dst), 1), "skipped": True})
            print(f"[{i}/{len(packages)}] {p.name} sudah ada, lewati", flush=True)
            continue
        t0 = time.time()
        m = retarget(ct.models.MLModel(str(p), skip_model_load=True))
        q = cto.palettize_weights(m, config) if args.mode.startswith("palette") \
            else cto.linear_quantize_weights(m, config)
        q.save(str(dst))
        before, after = size_mb(p), size_mb(dst)
        report.append({"model": p.stem, "before_mb": round(before, 1),
                       "after_mb": round(after, 1)})
        print(f"[{i}/{len(packages)}] {p.name}  {before:.1f} -> {after:.1f} MB "
              f"({time.time()-t0:.0f}s)", flush=True)

    # The conversion report is not a model but the pipeline reads the directory
    # as a unit; copy it so the quantised directory is a drop-in replacement.
    for extra in SRC.glob("*.json"):
        shutil.copy(extra, out_dir / extra.name)

    total_before = sum(r["before_mb"] for r in report)
    total_after = sum(r["after_mb"] for r in report)
    print(f"\nTOTAL {total_before:.0f} -> {total_after:.0f} MB "
          f"({total_after/total_before*100:.0f}%)")
    json.dump({"mode": args.mode, "total_before_mb": total_before,
               "total_after_mb": total_after, "models": report},
              open(out_dir / "quantization_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
