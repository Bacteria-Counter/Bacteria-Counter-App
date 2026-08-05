"""Tests the LAB a/b weekend model (48-epoch checkpoint, before Kaggle
completes the remaining 22) against real internet photos + the user's
'bacteria test' reference folder -- to see whether its AGAR ground-truth
improvement (MAE 5.44, huge gain on 'vague') actually generalizes to novel
real photos, not just the AGAR domain it was validated on so far.

Applies the LAB a/b preprocessing to each photo before inference, exactly as
during training, and compares counts against the production model
(colony_finetuned_best.pt) run on the same raw photos for reference.
"""
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

LAB_AB_CKPT = "/Users/satriabaladewaharahap/bacteriaserius/LAB AB MODEL/best-2.pt"
PRODUCTION_CKPT = "colony_finetuned_best.pt"

INTERNET_DIR = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/internet_test_images"
BACTERIA_TEST_DIR = "/Users/satriabaladewaharahap/bacteriaserius/bacteria test"


def lab_ab_isolate(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_stretch = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_stretch = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([a_stretch, b_stretch, np.full_like(a, 128)])


def count(model, img, conf=0.4):
    res = model.predict(source=img, imgsz=1536, conf=conf, max_det=1000, verbose=False)[0]
    return 0 if res.boxes is None else len(res.boxes)


def main():
    lab_ab_model = YOLO(LAB_AB_CKPT)
    prod_model = YOLO(PRODUCTION_CKPT)

    files = sorted(glob.glob(os.path.join(INTERNET_DIR, "img*.jpg"))) + \
        sorted(f for f in glob.glob(os.path.join(BACTERIA_TEST_DIR, "*")) if os.path.isfile(f))

    print(f"{'file':45s} {'production':>10s} {'lab_ab':>8s}")
    prod_total = lab_total = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"  SKIP (failed to load): {f}")
            continue
        prod_n = count(prod_model, img)
        lab_n = count(lab_ab_model, lab_ab_isolate(img))
        prod_total += prod_n
        lab_total += lab_n
        print(f"{os.path.basename(f):45s} {prod_n:10d} {lab_n:8d}")

    print(f"\nTOTAL: production={prod_total}, lab_ab={lab_total}")


if __name__ == "__main__":
    main()
