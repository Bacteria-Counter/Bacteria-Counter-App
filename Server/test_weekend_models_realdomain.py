"""Real-domain sanity check for the 3 weekend-trained variants: run each on
the 36 real empty-background photos from the user's lab (no bacteria present
at all), applying the SAME preprocessing each model was trained on. Expect
~0 detections everywhere -- any nonzero count is a false positive on a
scene with nothing in it, the same gate every other model in this project
has had to pass before being trusted.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BG_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/background_petridish_jpg")

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


PREPROC = {"dog_blend": dog_blend, "clahe": clahe_lab, "lab_ab": lab_ab_isolate}


def main():
    files = sorted(BG_DIR.glob("*.jpg"))
    all_results = {}
    for name, ckpt in MODELS.items():
        print(f"\n=== {name} ({len(files)} empty backgrounds) ===")
        model = YOLO(ckpt)
        fn = PREPROC[name]
        flagged = []
        for f in files:
            img = cv2.imread(str(f))
            proc = fn(img)
            res = model.predict(source=proc, imgsz=1536, conf=0.4, max_det=1000, verbose=False)[0]
            count = 0 if res.boxes is None else len(res.boxes)
            if count > 0:
                flagged.append((f.name, count))
        total_false = sum(c for _, c in flagged)
        print(f"  {len(flagged)}/{len(files)} images flagged, {total_false} total false detections")
        for name2, c in flagged:
            print(f"    {name2}: {c}")
        all_results[name] = {"flagged_images": len(flagged), "total_images": len(files),
                              "total_false_detections": total_false, "details": flagged}
        del model

    out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/weekend_models_realdomain_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    main()
