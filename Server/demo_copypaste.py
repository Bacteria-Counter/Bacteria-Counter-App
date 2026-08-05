import json
import sys
import cv2
import numpy as np
from ultralytics import FastSAM

sys.path.insert(0, ".")

SAMPLE_ID = "7740"
AGAR_DIR = "/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset"
BG_PATH = "/Users/satriabaladewaharahap/bacteriaserius/lab_photos/IMG_1674.jpg"
OUT_DIR = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad"

model = FastSAM("FastSAM-s.pt")

img = cv2.imread(f"{AGAR_DIR}/{SAMPLE_ID}.jpg")
labels = json.load(open(f"{AGAR_DIR}/{SAMPLE_ID}.json"))["labels"]


def extract_colony(img, label, margin=25):
    x, y, w, h = label["x"], label["y"], label["width"], label["height"]
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    # prompt SAM with the known colony center (from ground truth), instead of
    # guessing among automatic "segment everything" masks
    point_x, point_y = cw / 2, ch / 2
    res = model.predict(source=crop, points=[[int(point_x), int(point_y)]], labels=[1],
                         device="cpu", retina_masks=True, imgsz=640, conf=0.2, verbose=False)[0]
    def ellipse_fallback():
        m = np.zeros((ch, cw), dtype=np.uint8)
        cv2.ellipse(m, (int(point_x), int(point_y)), (int(w / 2), int(h / 2)), 0, 0, 360, 1, -1)
        return m, True

    if res.masks is None or len(res.masks.data) == 0:
        return crop, *ellipse_fallback()
    masks = res.masks.data.cpu().numpy()
    # pick the smallest-area mask among candidates (the "whole crop" background
    # mask is always much larger than a single colony)
    areas = [m.mean() for m in masks]
    best_idx = int(np.argmin(areas))
    best = cv2.resize(masks[best_idx].astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
    if best.mean() > 0.85 or best.mean() < 0.02:
        return crop, *ellipse_fallback()
    return crop, best, False


def make_pale_translucent(colony_bgr, mask, alpha_strength=0.55):
    """Desaturate, brighten, add a slight cool tint, and reduce opacity to
    approximate a pale/translucent colony appearance."""
    hsv = cv2.cvtColor(colony_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.25  # desaturate
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.35 + 40, 0, 255)  # brighten
    pale_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    pale_bgr[:, :, 0] = np.clip(pale_bgr[:, :, 0] * 1.05 + 8, 0, 255)  # slight blue tint
    alpha = (mask.astype(np.float32) * alpha_strength)
    return pale_bgr.astype(np.uint8), alpha


def composite(bg_bgr, colony_bgr, colony_alpha, cx, cy):
    ch, cw = colony_bgr.shape[:2]
    x1, y1 = int(cx - cw / 2), int(cy - ch / 2)
    x2, y2 = x1 + cw, y1 + ch
    if x1 < 0 or y1 < 0 or x2 > bg_bgr.shape[1] or y2 > bg_bgr.shape[0]:
        return bg_bgr
    roi = bg_bgr[y1:y2, x1:x2].astype(np.float32)
    a = colony_alpha[:, :, None]
    blended = roi * (1 - a) + colony_bgr.astype(np.float32) * a
    bg_bgr[y1:y2, x1:x2] = blended.astype(np.uint8)
    return bg_bgr


bg = cv2.imread(BG_PATH)
bg_h, bg_w = bg.shape[:2]
composite_img = bg.copy()

results_panels = []
positions = [(bg_w * 0.3, bg_h * 0.3), (bg_w * 0.6, bg_h * 0.4),
             (bg_w * 0.4, bg_h * 0.65), (bg_w * 0.7, bg_h * 0.7)]

for i, label in enumerate(labels):
    crop, mask, is_fallback = extract_colony(img, label)
    if mask is None:
        print(f"colony {label['id']}: extraction failed")
        continue
    pale_bgr, alpha = make_pale_translucent(crop, mask)
    composite_img = composite(composite_img, pale_bgr, alpha, *positions[i % len(positions)])
    results_panels.append((crop, mask, pale_bgr))
    method = "ellipse fallback" if is_fallback else "SAM mask"
    print(f"colony {label['id']} ({label['class']}, {label['width']}x{label['height']}): extracted OK via {method}")

cv2.imwrite(f"{OUT_DIR}/composite_demo.jpg", composite_img)

# build a panel figure: original crop | extracted mask | pale version, per colony
import matplotlib.pyplot as plt
n = len(results_panels)
fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
if n == 1:
    axes = axes.reshape(1, 3)
for i, (crop, mask, pale) in enumerate(results_panels):
    axes[i, 0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title("1. Original AGAR crop" if i == 0 else "")
    axes[i, 0].axis("off")
    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("2. SAM mask" if i == 0 else "")
    axes[i, 1].axis("off")
    axes[i, 2].imshow(cv2.cvtColor(pale, cv2.COLOR_BGR2RGB))
    axes[i, 2].set_title("3. Pale/translucent version" if i == 0 else "")
    axes[i, 2].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/extraction_steps.png", dpi=110, bbox_inches="tight")
print("Saved extraction_steps.png and composite_demo.jpg")
