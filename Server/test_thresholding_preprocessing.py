"""Tests whether heavy classical preprocessing (Otsu global thresholding,
Sauvola local/adaptive thresholding) helps or hurts the already-validated
fine-tuned YOLO colony counter, on the same 18-image stratified AGAR
ground-truth sample used throughout this project.

Rationale: the user found a reference (tang111111.github.io/ColonyCountingML)
whose pipeline binarizes each plate with Otsu before counting, and asked to
also explore Sauvola (a local-window adaptive threshold, better than Otsu
under uneven illumination). But that reference's own CNN component failed to
generalize from synthetic to real photos for lack of real training data --
the same domain-gap problem this project already solved differently (SAM
Copy-Paste fine-tuning). Also, a full classical-CV/Otsu-style watershed
reimplementation was already tried in this project and failed badly
(MAE ~120-708, confounded by printed text) -- so thresholding is not a new
idea here, just not yet tested specifically as a YOLO *preprocessing* step
rather than as a standalone counter.

This script tests exactly that: does binarizing the image before feeding it
to the fine-tuned YOLO model help, on the same ground truth used for every
other benchmark in this project?
"""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from ultralytics import YOLO

RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
SAMPLE_FILE = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/agar_clahe_sample.json"
YOLO_MODEL_PATH = "colony_finetuned_best.pt"
YOLO_CONF = 0.4
YOLO_IMGSZ = 1536


def otsu_binary_3ch(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def sauvola_binary_3ch(img_bgr, window_size):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = threshold_sauvola(gray, window_size=window_size)
    binary = (gray > thresh).astype(np.uint8) * 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def clahe_enhance(img_bgr):
    # same recipe already validated for the SAM pipeline (make_clahe in
    # server.py / fastsam_colony_count.py) -- gentler than binarization,
    # keeps color/texture, only boosts local contrast on the L channel.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def run_yolo_on(model, img_bgr):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        path = tmp.name
    res = model.predict(source=path, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
                         max_det=1000, device="cpu", verbose=False)[0]
    return 0 if res.boxes is None else len(res.boxes)


def main():
    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    print("Loading fine-tuned YOLO...")
    model = YOLO(YOLO_MODEL_PATH)

    variants = ["baseline", "otsu", "sauvola_w25", "sauvola_w51", "clahe"]
    results = {v: [] for v in variants}
    gts = []

    for i, (sid, bg, gt) in enumerate(sample):
        img_path = RAW_DIR / f"{sid}.jpg"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        gts.append(gt)

        counts = {
            "baseline": run_yolo_on(model, img_bgr),
            "otsu": run_yolo_on(model, otsu_binary_3ch(img_bgr)),
            "sauvola_w25": run_yolo_on(model, sauvola_binary_3ch(img_bgr, 25)),
            "sauvola_w51": run_yolo_on(model, sauvola_binary_3ch(img_bgr, 51)),
            "clahe": run_yolo_on(model, clahe_enhance(img_bgr)),
        }
        for v in variants:
            results[v].append(counts[v])

        print(f"[{i+1}/{len(sample)}] {sid} (bg={bg}, gt={gt}): "
              + " ".join(f"{v}={counts[v]}" for v in variants))

    print("\n=== MAE vs ground truth (n={}) ===".format(len(gts)))
    for v in variants:
        mae = sum(abs(c - g) for c, g in zip(results[v], gts)) / len(gts)
        print(f"{v:14s} MAE = {mae:.2f}")


if __name__ == "__main__":
    main()
