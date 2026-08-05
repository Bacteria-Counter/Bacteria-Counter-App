"""Composites extracted, recolored real AGAR colonies onto the user's real
empty-dish background photos to build an auto-labeled synthetic training set
for fine-tuning the YOLOv8n colony counter to the user's actual domain.

Because we choose exactly where each colony goes, YOLO-format labels are
generated for free — no manual annotation needed.
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")
from fastsam_colony_count import find_dish_circle

BACKGROUND_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/background_petridish_jpg")
COLONY_MANIFEST = Path("/Users/satriabaladewaharahap/bacteriaserius/extracted_colonies/manifest.json")
OUT_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/synthetic_dataset")

WORKING_MAX_DIM = 1536  # matches the YOLO model's training imgsz
DISH_MARGIN_RATIO = 0.92  # keep colonies off the dish rim itself
N_TRAIN = 2000
N_VAL = 400
MIN_COLONIES = 1
MAX_COLONIES = 120
MIN_COLONY_DIAM_FRAC = 0.015  # of dish diameter
MAX_COLONY_DIAM_FRAC = 0.12
MAX_PLACEMENT_ATTEMPTS = 8

random.seed(42)
np.random.seed(42)


def load_backgrounds():
    backgrounds = []
    for path in sorted(BACKGROUND_DIR.glob("*.jpg")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = WORKING_MAX_DIM / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        dish = find_dish_circle(img)
        if dish is None:
            print(f"  skipping {path.name}: no dish detected")
            continue
        backgrounds.append({"image": img, "dish": dish, "name": path.name})
    return backgrounds


def jitter_background(img_bgr):
    brightness = random.uniform(-20, 20)
    contrast = random.uniform(0.9, 1.1)
    out = img_bgr.astype(np.float32) * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def tight_alpha_bbox(bgra):
    alpha = bgra[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def random_point_in_circle(cx, cy, r):
    angle = random.uniform(0, 2 * np.pi)
    dist = r * np.sqrt(random.uniform(0, 1))
    return cx + dist * np.cos(angle), cy + dist * np.sin(angle)


def composite_one(background_entry, colonies):
    img = jitter_background(background_entry["image"].copy())
    dx, dy, dr = background_entry["dish"]
    place_r = dr * DISH_MARGIN_RATIO

    n_colonies = random.randint(MIN_COLONIES, MAX_COLONIES)
    placed = []  # (cx, cy, radius)
    labels = []

    for _ in range(n_colonies):
        colony_meta = random.choice(colonies)
        colony_bgra = cv2.imread(colony_meta["path"], cv2.IMREAD_UNCHANGED)
        if colony_bgra is None or colony_bgra.shape[2] != 4:
            continue

        diam_frac = random.uniform(MIN_COLONY_DIAM_FRAC, MAX_COLONY_DIAM_FRAC)
        target_diam = max(6, int(2 * dr * diam_frac))
        ch, cw = colony_bgra.shape[:2]
        scale = target_diam / max(ch, cw)
        new_w, new_h = max(4, int(cw * scale)), max(4, int(ch * scale))
        resized = cv2.resize(colony_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

        radius_est = max(new_w, new_h) / 2
        effective_place_r = max(1.0, place_r - radius_est)

        placed_ok = False
        for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
            px, py = random_point_in_circle(dx, dy, effective_place_r)
            too_crowded = any(
                np.hypot(px - ex, py - ey) < 0.35 * (radius_est + er)
                for ex, ey, er in placed
            )
            if too_crowded:
                continue
            placed_ok = True
            break
        if not placed_ok:
            continue

        x1 = int(px - new_w / 2)
        y1 = int(py - new_h / 2)
        x2, y2 = x1 + new_w, y1 + new_h
        ih, iw = img.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > iw or y2 > ih:
            continue

        alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
        roi = img[y1:y2, x1:x2].astype(np.float32)
        blended = roi * (1 - alpha) + resized[:, :, :3].astype(np.float32) * alpha
        img[y1:y2, x1:x2] = blended.astype(np.uint8)

        placed.append((px, py, max(new_w, new_h) / 2))

        bbox = tight_alpha_bbox(resized)
        if bbox is None:
            continue
        bx1, by1, bx2, by2 = bbox
        label_x1, label_y1 = x1 + bx1, y1 + by1
        label_x2, label_y2 = x1 + bx2, y1 + by2
        cx_n = ((label_x1 + label_x2) / 2) / iw
        cy_n = ((label_y1 + label_y2) / 2) / ih
        w_n = (label_x2 - label_x1) / iw
        h_n = (label_y2 - label_y1) / ih
        labels.append((cx_n, cy_n, w_n, h_n))

    return img, labels


def generate_split(split_name, count, backgrounds, colonies, images_dir, labels_dir):
    for i in range(count):
        bg = random.choice(backgrounds)
        img, labels = composite_one(bg, colonies)
        stem = f"{split_name}_{i:05d}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        with open(labels_dir / f"{stem}.txt", "w") as f:
            for cx, cy, w, h in labels:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if (i + 1) % 200 == 0:
            print(f"  [{split_name}] {i+1}/{count}")


def main():
    print("Loading backgrounds...")
    backgrounds = load_backgrounds()
    print(f"  {len(backgrounds)} usable backgrounds")

    with open(COLONY_MANIFEST) as f:
        colonies = json.load(f)
    print(f"Loaded {len(colonies)} extracted colonies")

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Generating train split...")
    generate_split("train", N_TRAIN, backgrounds, colonies,
                    OUT_DIR / "images" / "train", OUT_DIR / "labels" / "train")
    print("Generating val split...")
    generate_split("val", N_VAL, backgrounds, colonies,
                    OUT_DIR / "images" / "val", OUT_DIR / "labels" / "val")

    data_yaml = f"""path: {OUT_DIR}
train: images/train
val: images/val
names:
  0: colony
"""
    (OUT_DIR / "data.yaml").write_text(data_yaml)
    print("Wrote data.yaml")
    print("Done.")


if __name__ == "__main__":
    main()
