"""Extracts many real colonies from labeled AGAR images, recolors them toward
a pale/translucent appearance (matching the user's real bacteria, which look
nothing like AGAR's tan/brown colonies), and saves each as a standalone RGBA
PNG + JSON metadata for later compositing onto the user's real backgrounds.

This is extract_colony()/make_pale_translucent() from demo_copypaste.py,
scaled up across ~350 source images instead of one.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import FastSAM

sys.path.insert(0, ".")

AGAR_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
SAMPLE_FILE = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/extraction_sample_ids.json"
OUT_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/extracted_colonies")

model = FastSAM("FastSAM-s.pt")


def extract_colony(img, label, margin=25):
    x, y, w, h = label["x"], label["y"], label["width"], label["height"]
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img.shape[1], x + w + margin), min(img.shape[0], y + h + margin)
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return crop, None, False

    point_x, point_y = cw / 2, ch / 2

    def ellipse_fallback():
        m = np.zeros((ch, cw), dtype=np.uint8)
        cv2.ellipse(m, (int(point_x), int(point_y)), (max(1, int(w / 2)), max(1, int(h / 2))), 0, 0, 360, 1, -1)
        return m, True

    try:
        res = model.predict(source=crop, points=[[int(point_x), int(point_y)]], labels=[1],
                             device="cpu", retina_masks=True, imgsz=640, conf=0.2, verbose=False)[0]
    except Exception:
        m, is_fb = ellipse_fallback()
        return crop, m, is_fb

    if res.masks is None or len(res.masks.data) == 0:
        m, is_fb = ellipse_fallback()
        return crop, m, is_fb

    masks = res.masks.data.cpu().numpy()
    areas = [m.mean() for m in masks]
    best_idx = int(np.argmin(areas))
    best = cv2.resize(masks[best_idx].astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
    if best.mean() > 0.85 or best.mean() < 0.02:
        m, is_fb = ellipse_fallback()
        return crop, m, is_fb
    return crop, best, False


def make_pale_translucent(colony_bgr, mask, alpha_strength=0.65):
    hsv = cv2.cvtColor(colony_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.25
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.35 + 40, 0, 255)
    pale_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    pale_bgr[:, :, 0] = np.clip(pale_bgr[:, :, 0] * 1.05 + 8, 0, 255)
    alpha = mask.astype(np.float32) * alpha_strength
    return pale_bgr.astype(np.uint8), alpha


def main():
    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    t0 = time.time()
    total_images = sum(len(ids) for ids in sample.values())
    processed_images = 0

    for species, sample_ids in sample.items():
        species_dir = OUT_DIR / species
        species_dir.mkdir(parents=True, exist_ok=True)

        for sid in sample_ids:
            processed_images += 1
            img_path = AGAR_DIR / f"{sid}.jpg"
            json_path = AGAR_DIR / f"{sid}.json"
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            data = json.load(open(json_path))
            labels = data.get("labels", [])

            for label in labels:
                crop, mask, is_fallback = extract_colony(img, label)
                if mask is None:
                    continue
                pale_bgr, alpha = make_pale_translucent(crop, mask)
                bgra = cv2.cvtColor(pale_bgr, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)

                colony_id = f"{sid}_{label['id']}"
                out_path = species_dir / f"{colony_id}.png"
                cv2.imwrite(str(out_path), bgra)

                manifest.append({
                    "colony_id": colony_id,
                    "species": species,
                    "source_sample_id": sid,
                    "width": bgra.shape[1],
                    "height": bgra.shape[0],
                    "orig_width": label["width"],
                    "orig_height": label["height"],
                    "used_sam_mask": not is_fallback,
                    "path": str(out_path),
                })

            if processed_images % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {processed_images}/{total_images} source images, "
                      f"{len(manifest)} colonies extracted so far, {elapsed:.0f}s elapsed")

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. {len(manifest)} colonies extracted from {processed_images} images.")
    print(f"SAM mask used: {sum(1 for m in manifest if m['used_sam_mask'])}, "
          f"ellipse fallback: {sum(1 for m in manifest if not m['used_sam_mask'])}")
    print("Total time:", time.time() - t0)


if __name__ == "__main__":
    main()
