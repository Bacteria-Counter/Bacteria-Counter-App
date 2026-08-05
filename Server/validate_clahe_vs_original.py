import csv
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import FastSAM

sys.path.insert(0, ".")
from fastsam_colony_count import count_colonies_fastsam

RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
SAMPLE_FILE = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/agar_clahe_sample.json"
OUT_CSV = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/clahe_vs_original_agar.csv"


def make_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def main():
    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    model = FastSAM("FastSAM-s.pt")
    results = []
    t0 = time.time()
    for i, (sid, bg, gt) in enumerate(sample):
        img_path = RAW_DIR / f"{sid}.jpg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"SKIP {sid}: could not read image")
            continue

        clahe_bgr = make_clahe(img_bgr)
        clahe_path = f"/tmp/clahe_agar_{sid}.jpg"
        cv2.imwrite(clahe_path, clahe_bgr)

        count_orig, _, _ = count_colonies_fastsam(model, str(img_path), imgsz=3840,
                                                    min_area_frac=0.000005, max_area_frac=0.02)
        count_clahe, _, _ = count_colonies_fastsam(model, clahe_path, imgsz=3840,
                                                     min_area_frac=0.000005, max_area_frac=0.02)

        results.append({"sample_id": sid, "background": bg, "gt": gt,
                         "pred_original": count_orig, "pred_clahe": count_clahe})
        print(f"[{i+1}/{len(sample)}] {sid} (bg={bg}, gt={gt}): "
              f"original={count_orig}  clahe={count_clahe}  elapsed={time.time()-t0:.0f}s")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("Saved to", OUT_CSV)
    print("Total time:", time.time() - t0)


if __name__ == "__main__":
    main()
