"""Before and after: the Python server against what ships today.

Everything measured so far compared one link at a time -- PyTorch to Core ML,
Core ML to Swift, float32 to int8. Those are the right comparisons for finding
faults, but they do not answer "what changed for the person using the app",
because assuming three measured deltas compose is exactly the sort of
assumption this project has been caught by before.

So this runs the two ENDS against each other, on the same plates:

  before -- server.py's code path, verbatim: PyTorch checkpoints through
            ultralytics, adaptive imgsz, and for FastSAM the CLAHE-to-temporary-
            JPEG round trip the server really did. That round trip is not a
            confound here; it is part of what the old system was.
  after  -- the shipped Swift binary against the int8 Core ML models.

Reported for accuracy and for speed, since both changed.

Usage:  python eval_before_after.py [model ...]
"""
import os
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
SERVER = ROOT / "Bacteria-Counter-App/Server"
SET = ROOT / "bench_data/verify_set"
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"
INT8 = ROOT / "coreml_models_block32"
sys.path.insert(0, str(SERVER))

import cv2
import torch
import torchvision.transforms as T
from csrnet_model import CSRNet
from fastsam_colony_count import count_colonies_fastsam, find_dish_circle
from ultralytics import FastSAM, YOLO

YOLO_PATHS = {"yolo_old": "models_trained/YOLO/counter/best.pt",
              "yolo_new": "colony_finetuned_best.pt",
              "mac1": "mac1_best.pt", "mac2": "mac2_best.pt"}
SAM_ARGS = dict(conf=0.2, min_circularity=0.75, dish_margin_ratio=1.0,
                min_area_frac=1e-7, max_area_frac=0.005,
                drop_square_like=False, reject_markings=True)


def adaptive_imgsz(img):
    h, w = img.shape[:2]
    d = find_dish_circle(img)
    if d is None:
        return 1536
    return max(1280, min(3200, int(round((581 * max(h, w) / d[2]) / 32) * 32)))


def before_yolo(model, path):
    img = cv2.imread(str(path))
    r = model.predict(source=str(path), imgsz=adaptive_imgsz(img), conf=0.4,
                      max_det=1000, device="cpu", verbose=False)[0]
    return 0 if r.boxes is None else len(r.boxes)


def before_sam(model, path):
    """run_sam() verbatim, JPEG round trip included."""
    img = cv2.imread(str(path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    clahe = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
        cv2.imwrite(t.name, clahe)
        tmp = t.name
    count, _, _ = count_colonies_fastsam(model, tmp, imgsz=adaptive_imgsz(img),
                                         color_ref_bgr=img, **SAM_ARGS)
    Path(tmp).unlink(missing_ok=True)
    return count


def before_csrnet(model, tf, path):
    img = cv2.imread(str(path))
    small = cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        d = model(tf(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).unsqueeze(0))
    return round(float(d.sum().item()))


def after(files, key):
    """The shipped binary. First image carries the model load; the app loads
    once and keeps it, so the median is the honest steady-state figure."""
    cmd = [str(SWIFT), str(INT8)] + [str(f) for f in files] + ["--model", key]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
    counts, secs = {}, []
    for line in out.stdout.splitlines():
        p = line.strip().split("|")
        if len(p) >= 5 and p[1].lstrip("-").isdigit():
            counts[p[0]] = int(p[1])
            secs.append(float(p[4]))
    if not counts:
        print("  swift stderr:", out.stderr[-300:])
    return counts, (float(np.median(secs[1:] or secs)) if secs else 0.0)


def main():
    files = sorted(SET.glob("h_*.png"))
    bench = {e["base"]: e["truth"]
             for e in json.load(open(ROOT / "bench_data/pca_benchmark.json"))}
    truth = [bench[f.name[2:-4]] for f in files]
    keys = sys.argv[1:] or ["yolo_new", "mac1", "sam_tuned", "csrnet"]
    print(f"{len(files)} cawan holdout\n", flush=True)

    sam_model = csr_model = csr_tf = None
    results = {}
    for key in keys:
        t0 = time.time()
        if key in YOLO_PATHS:
            m = YOLO(str(SERVER / YOLO_PATHS[key]))
            fn = lambda p, m=m: before_yolo(m, p)
        elif key == "sam_tuned":
            if sam_model is None:
                sam_model = FastSAM(str(SERVER / "FastSAM-s.pt"))
            fn = lambda p: before_sam(sam_model, p)
        elif key == "csrnet":
            if csr_model is None:
                csr_model = CSRNet(load_weights=True)
                csr_model.load_state_dict(torch.load(SERVER / "csrnet_best.pt",
                                                     map_location="cpu", weights_only=True))
                csr_model.eval()
                csr_tf = T.Compose([T.ToTensor(),
                                    T.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
            fn = lambda p: before_csrnet(csr_model, csr_tf, p)
        else:
            print(f"lewati {key}"); continue
        load_s = time.time() - t0

        before, times = [], []
        for f in files:
            t = time.time()
            before.append(fn(f))
            times.append(time.time() - t)
        after_counts, after_s = after(files, key)
        aft = [after_counts[f.name] for f in files]

        b = np.array(before, float); a = np.array(aft, float); t = np.array(truth, float)
        r = {"mae_before": float(np.abs(b - t).mean()),
             "mae_after": float(np.abs(a - t).mean()),
             "bias_before": float((b - t).mean()), "bias_after": float((a - t).mean()),
             "identical": int((b == a).sum()), "n": len(files),
             "within1": int((np.abs(b - a) <= 1).sum()),
             "sec_before": float(np.median(times)), "sec_after": after_s,
             "load_before_s": load_s}
        results[key] = r
        print(f"{key:<11} MAE {r['mae_before']:6.2f} -> {r['mae_after']:6.2f}   "
              f"identik {r['identical']:>2}/{r['n']}  dalam+-1 {r['within1']:>2}/{r['n']}   "
              f"{r['sec_before']:5.2f}s -> {r['sec_after']:5.2f}s per foto   "
              f"(muat model {load_s:.1f}s)", flush=True)
        json.dump(results, open(ROOT / "bench_data/eval_before_after.json", "w"), indent=1)

    print("\nSELESAI")


if __name__ == "__main__":
    main()
