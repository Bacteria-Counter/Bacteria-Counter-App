"""Explores preprocessing techniques aimed purely at HUMAN visual clarity --
making faint/blurry colonies on "vague"-background AGAR plates pop out
clearly to the eye -- as a separate question from whether a given
preprocessing helps the already-trained YOLO model (already tested
separately: Otsu/Sauvola/CLAHE all hurt YOLO's accuracy when applied at
inference time). This is step 1: find what makes colonies *humanly* visible.
Whether that same preprocessing helps a model (via retraining on
preprocessed data, not just inference-time application) is a follow-up
question, not answered here.

Uses the 3 worst "vague"-category samples plus one "bright" sample from the
18-image ground-truth sample (largest gaps between fine-tuned YOLO's
prediction and the real count), cropped to the dish area and a zoomed
sub-region so individual colonies are visible at real pixel size instead of
shrunk in a full 4000px photo -- PLUS two crops from the user's own real lab
photos (grid-mat, backlit, real condensation/bubbles/defects), since "vague
and blurry" isn't only an AGAR "vague"-background thing.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

sys.path.insert(0, ".")
from fastsam_colony_count import find_dish_circle

RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
OUT_DIR = Path("/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/enhance_test")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# worst "vague" samples: (sample_id, gt, fine-tuned-yolo-pred)
SAMPLES = [
    ("12497", 63, 10),
    ("12193", 62, 18),
    ("11971", 36, 18),
    ("515", 116, 112),  # bright category too -- "vague/blurry" isn't only an AGAR-background label
]

# Real lab photos (grid-mat, backlit, actual condensation/bubbles/defects) --
# fixed crops since find_dish_circle is tuned for AGAR's plain background,
# not this domain. (x1, y1, x2, y2) in the original 8064x4536 photo.
REAL_LAB_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/lab_photos")
REAL_PHOTOS = [
    ("IMG_1034_3", (1700, 2600, 3300, 4200)),
    ("IMG_1021_3", (1600, 2800, 3200, 4400)),
]

CROP_SIZE = 900  # zoomed sub-region side length, centered slightly off-center to catch texture, not just dish center


def crop_dish_region(img_bgr):
    circle = find_dish_circle(img_bgr)
    h, w = img_bgr.shape[:2]
    if circle is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy, r = circle
        cx, cy = int(cx), int(cy)
    # offset from dead-center so the crop catches an edge region (often where
    # faint colonies are hardest to see against gradient lighting)
    cx = cx - int(CROP_SIZE * 0.4)
    cy = cy - int(CROP_SIZE * 0.1)
    x1 = max(0, cx - CROP_SIZE // 2)
    y1 = max(0, cy - CROP_SIZE // 2)
    x2 = min(w, x1 + CROP_SIZE)
    y2 = min(h, y1 + CROP_SIZE)
    return img_bgr[y1:y2, x1:x2]


def clahe_lab(img_bgr, clip=3.0, tile=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def gamma_correct(img_bgr, gamma=0.5):
    inv = 1.0 / gamma
    table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def unsharp_mask(img_bgr, sigma=5, amount=1.5):
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    return cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)


def difference_of_gaussians(img_bgr, sigma1=2, sigma2=12):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    dog = g1 - g2
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(dog, cv2.COLOR_GRAY2BGR)


def detail_enhance(img_bgr):
    return cv2.detailEnhance(img_bgr, sigma_s=12, sigma_r=0.4)


def clahe_plus_unsharp(img_bgr):
    return unsharp_mask(clahe_lab(img_bgr), sigma=4, amount=1.0)


def otsu_binary(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def sauvola_binary(img_bgr, window_size=35):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = threshold_sauvola(gray, window_size=window_size)
    binary = (gray > thresh).astype(np.uint8) * 255
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def lab_a_channel_boost(img_bgr):
    # colonies sometimes carry a subtle colour cast (yellow/brown vs the
    # agar's own tint) invisible in normal RGB but separable in LAB's a/b
    # channels; stretch + recombine to see if that channel alone reveals
    # colonies the human eye merges into the background in full colour.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_stretch = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_stretch = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([a_stretch, b_stretch, np.full_like(a, 128)])


VARIANTS = [
    ("raw", lambda im: im),
    ("clahe", clahe_lab),
    ("gamma_0.5", lambda im: gamma_correct(im, 0.5)),
    ("unsharp_mask", unsharp_mask),
    ("clahe_plus_unsharp", clahe_plus_unsharp),
    ("difference_of_gaussians", difference_of_gaussians),
    ("detail_enhance", detail_enhance),
    ("lab_ab_channels", lab_a_channel_boost),
    ("otsu", otsu_binary),
    ("sauvola", sauvola_binary),
]


def main():
    for sid, gt, pred in SAMPLES:
        img_path = RAW_DIR / f"{sid}.jpg"
        img = cv2.imread(str(img_path))
        crop = crop_dish_region(img)

        for name, fn in VARIANTS:
            out = fn(crop.copy())
            out_path = OUT_DIR / f"{sid}_gt{gt}_yolo{pred}_{name}.jpg"
            cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"sample {sid} (gt={gt}, yolo_pred={pred}): {len(VARIANTS)} variants written")

    for name, (x1, y1, x2, y2) in REAL_PHOTOS:
        img_path = REAL_LAB_DIR / f"{name}.jpg"
        img = cv2.imread(str(img_path))
        crop = img[y1:y2, x1:x2]

        for vname, fn in VARIANTS:
            out = fn(crop.copy())
            out_path = OUT_DIR / f"real_{name}_{vname}.jpg"
            cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"real photo {name}: {len(VARIANTS)} variants written")

    print("Done. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
