"""SAM proposes candidates, YOLO verifies each one at native crop resolution.

Rationale (see conversation): YOLO alone misses small colonies on large AGAR
images because they shrink below its detection floor after the mandatory
letterbox resize to imgsz=1536. SAM finds candidates regardless of image
size, but is fooled by round confounders (printed text, grid patterns) that
a shape filter alone can't reject. Cropping each SAM candidate and re-running
YOLO on just that crop removes the resolution-loss problem AND lets YOLO's
learned "is this really a colony" judgement reject SAM's false positives —
AS LONG AS the domain matches what YOLO was trained on (AGAR only, not the
user's own lab photos).
"""
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM, YOLO

sys.path.insert(0, ".")
from fastsam_colony_count import count_colonies_fastsam

RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
SAMPLE_FILE = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/agar_clahe_sample.json"
OUT_CSV = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/ensemble_vs_solo_agar.csv"

YOLO_MODEL_PATH = "models_trained/YOLO/counter/best.pt"
VERIFY_CONF = 0.25
VERIFY_IMGSZ = 640
CROP_MARGIN_FACTOR = 2.5  # crop side = 2 * radius * this factor


def make_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def verify_with_yolo(yolo_model, img_bgr, cx, cy, radius):
    h, w = img_bgr.shape[:2]
    half = max(20, radius * CROP_MARGIN_FACTOR)
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return False
    crop = img_bgr[y1:y2, x1:x2]
    res = yolo_model.predict(source=crop, imgsz=VERIFY_IMGSZ, conf=VERIFY_CONF,
                              max_det=50, device="cpu", verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return False
    # accept if ANY verified box's center falls reasonably near the crop
    # center (i.e., near the original SAM candidate), not just anywhere in
    # the (generously margined) crop
    ch, cw = crop.shape[:2]
    ccx, ccy = cw / 2, ch / 2
    xyxy = boxes.xyxy.cpu().numpy()
    for (bx1, by1, bx2, by2) in xyxy:
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if abs(bcx - ccx) < half * 0.9 and abs(bcy - ccy) < half * 0.9:
            return True
    return False


def main():
    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    print("Loading models...")
    sam_model = FastSAM("FastSAM-s.pt")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    results = []
    t0 = time.time()
    for i, (sid, bg, gt) in enumerate(sample):
        img_path = RAW_DIR / f"{sid}.jpg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # SAM candidates (reuse the already-tuned CLAHE + shape-filter pipeline)
        clahe_bgr = make_clahe(img_bgr)
        clahe_path = f"/tmp/ens_clahe_{sid}.jpg"
        cv2.imwrite(clahe_path, clahe_bgr)
        sam_count, kept, _dish = count_colonies_fastsam(
            sam_model, clahe_path, imgsz=3840, min_area_frac=0.000005, max_area_frac=0.02,
        )

        # YOLO solo (single-shot, the already-validated setting)
        yres = yolo_model.predict(source=str(img_path), imgsz=1536, conf=0.4,
                                   max_det=1000, device="cpu", verbose=False)[0]
        yolo_solo_count = 0 if yres.boxes is None else len(yres.boxes)

        # Ensemble: verify each SAM candidate with YOLO on its native-res crop
        ensemble_count = 0
        for k in kept:
            cx, cy = k["centroid"]
            # approximate radius from mask area (kept has 'area', not radius directly)
            radius = float(np.sqrt(k["area"] / np.pi))
            if verify_with_yolo(yolo_model, img_bgr, cx, cy, radius):
                ensemble_count += 1

        results.append({
            "sample_id": sid, "background": bg, "gt": gt,
            "sam_solo": sam_count, "yolo_solo": yolo_solo_count, "ensemble": ensemble_count,
        })
        print(f"[{i+1}/{len(sample)}] {sid} (bg={bg}, gt={gt}): "
              f"sam={sam_count} yolo={yolo_solo_count} ensemble={ensemble_count} "
              f"elapsed={time.time()-t0:.0f}s")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("Saved to", OUT_CSV)
    print("Total time:", time.time() - t0)


if __name__ == "__main__":
    main()
