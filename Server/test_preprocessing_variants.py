import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import FastSAM

sys.path.insert(0, ".")
from fastsam_colony_count import count_colonies_fastsam

FILES = ["IMG_1021_3.jpg", "IMG_1031_2.jpg", "IMG_1034_3.jpg"]
DIR = "/Users/satriabaladewaharahap/bacteriaserius/lab_photos"
TMP = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad"

model = FastSAM("FastSAM-s.pt")


def make_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


fig, axes = plt.subplots(len(FILES), 2, figsize=(16, 7 * len(FILES)))
for i, fname in enumerate(FILES):
    path = f"{DIR}/{fname}"
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    clahe_bgr = make_clahe(img_bgr)
    clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)
    clahe_path = f"{TMP}/clahe_{fname}"
    cv2.imwrite(clahe_path, clahe_bgr)

    count_orig, kept_orig, dish_orig = count_colonies_fastsam(model, path, imgsz=3840, min_area_frac=0.000005, max_area_frac=0.02)
    count_clahe, kept_clahe, dish_clahe = count_colonies_fastsam(model, clahe_path, imgsz=3840, min_area_frac=0.000005, max_area_frac=0.02)

    vis_orig = img_rgb.copy()
    if dish_orig is not None:
        dx, dy, dr = dish_orig
        cv2.circle(vis_orig, (int(dx), int(dy)), int(dr), (255, 0, 255), 6)
    for k in kept_orig:
        contours, _ = cv2.findContours(k["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_orig, contours, -1, (0, 255, 60), 6)

    vis_clahe = img_rgb.copy()
    if dish_clahe is not None:
        dx, dy, dr = dish_clahe
        cv2.circle(vis_clahe, (int(dx), int(dy)), int(dr), (255, 0, 255), 6)
    for k in kept_clahe:
        contours, _ = cv2.findContours(k["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_clahe, contours, -1, (0, 255, 60), 6)

    axes[i, 0].imshow(vis_orig)
    axes[i, 0].set_title(f"{fname}\nORIGINAL: count={count_orig}", fontsize=11)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(vis_clahe)
    axes[i, 1].set_title(f"{fname}\nCLAHE-enhanced: count={count_clahe}", fontsize=11, fontweight="bold")
    axes[i, 1].axis("off")

    print(f"{fname}: original={count_orig}  clahe={count_clahe}")

plt.tight_layout()
out_path = f"{TMP}/preprocessing_comparison.png"
plt.savefig(out_path, dpi=90, bbox_inches="tight")
print("Saved to", out_path)
