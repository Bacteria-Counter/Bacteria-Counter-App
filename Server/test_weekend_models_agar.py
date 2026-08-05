"""Validates the 3 weekend-trained variants (DoG-blend, CLAHE, LAB a/b baked
into training) against the same 18-image stratified AGAR ground-truth sample
used for every other benchmark in this project. Crucially, applies the SAME
preprocessing each model was actually trained on before inference -- these
models expect that transformed input distribution, not raw images (that's
the whole point of baking it into training instead of bolting it onto
inference like the earlier CLAHE-only test).
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from skimage.filters import threshold_sauvola  # unused here but keep parity with other scripts

RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
SAMPLE_FILE = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/agar_clahe_sample.json"

MODELS = {
    "dog_blend": "/Users/satriabaladewaharahap/bacteriaserius/DOG BLEND MODEL/best-2.pt",
    "clahe": "/Users/satriabaladewaharahap/bacteriaserius/CLAHE MODEL/best-2.pt",
    "lab_ab": "/Users/satriabaladewaharahap/bacteriaserius/LAB AB MODEL/best-2.pt",
}


def dog_blend(img_bgr, sigma1=2, sigma2=12, strength=1.0):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    dog = g1 - g2
    out = img_bgr.astype(np.float32) + (strength * dog)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_lab(img_bgr, clip=3.0, tile=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def lab_ab_isolate(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_stretch = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_stretch = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([a_stretch, b_stretch, np.full_like(a, 128)])


PREPROC = {
    "dog_blend": dog_blend,
    "clahe": clahe_lab,
    "lab_ab": lab_ab_isolate,
}


def main():
    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    results = {name: [] for name in MODELS}

    for name, ckpt in MODELS.items():
        print(f"\n=== {name} ===")
        model = YOLO(ckpt)
        fn = PREPROC[name]
        for sid, bg, gt in sample:
            img_path = RAW_DIR / f"{sid}.jpg"
            img = cv2.imread(str(img_path))
            proc = fn(img)
            res = model.predict(source=proc, imgsz=1536, conf=0.4, max_det=1000, verbose=False)[0]
            pred = 0 if res.boxes is None else len(res.boxes)
            results[name].append({"sample_id": sid, "background": bg, "gt": gt, "pred": pred})
            print(f"  {sid} (bg={bg}, gt={gt}): pred={pred}")
        del model

    out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/weekend_models_agar_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    for name, rows in results.items():
        mae = sum(abs(r["pred"] - r["gt"]) for r in rows) / len(rows)
        print(f"{name}: MAE = {mae:.2f}")
        for bg in ["bright", "dark", "vague"]:
            sub = [r for r in rows if r["background"] == bg]
            bg_mae = sum(abs(r["pred"] - r["gt"]) for r in sub) / len(sub)
            print(f"   {bg}: MAE = {bg_mae:.2f}")

    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
