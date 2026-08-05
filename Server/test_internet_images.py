"""Runs the old (AGAR-only) and new (fine-tuned) YOLO checkpoints on 10 real
bacteria-colony photos sourced from Wikimedia Commons, and produces a
side-by-side visual comparison per image so the user can judge for
themselves whether fine-tuning helped or hurt on genuinely new photos
neither model was validated against before.
"""
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

IMG_DIR = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/internet_test_images"
OUT_DIR = "/Users/satriabaladewaharahap/bacteriaserius/internet_test_results"
os.makedirs(OUT_DIR, exist_ok=True)

OLD_CKPT = "models_trained/YOLO/counter/best.pt"
NEW_CKPT = "colony_finetuned_best.pt"

IMGSZ = 1536
CONF = 0.4

old_model = YOLO(OLD_CKPT)
new_model = YOLO(NEW_CKPT)

SOURCE_NAMES = {
    "img01.jpg": "Colonies of E. coli on agar plate",
    "img02.jpg": "Green colonies of E. coli on agar plate",
    "img03.jpg": "Agar plate with colonies",
    "img04.jpg": "VCU agar plate colonies",
    "img05.jpg": "Agar plate with colonies (square)",
    "img06.jpg": "Pseudomonas aeruginosa on blood agar",
    "img07.jpg": "Salmonella growth on MacConkey medium",
    "img08.jpg": "Citrobacter freundii",
    "img09.jpg": "Klebsiella pneumoniae (MLF) on MacConkey agar",
    "img10.jpg": "Fecal flora on XLD agar",
}


def draw_boxes(img, boxes, color, label):
    out = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, f"{label}: {len(boxes)} colonies", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
    return out


def run(model, path):
    res = model.predict(source=path, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
    return boxes


rows = []
files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
for path in files:
    fname = os.path.basename(path)
    img = cv2.imread(path)
    h, w = img.shape[:2]

    old_boxes = run(old_model, path)
    new_boxes = run(new_model, path)

    old_vis = draw_boxes(img, old_boxes, (0, 0, 255), "OLD (AGAR-only)")
    new_vis = draw_boxes(img, new_boxes, (0, 200, 0), "NEW (fine-tuned)")

    # stack side by side, resize to common height for readability
    target_h = 700
    scale = target_h / h
    old_vis_r = cv2.resize(old_vis, (int(w * scale), target_h))
    new_vis_r = cv2.resize(new_vis, (int(w * scale), target_h))
    combined = np.hstack([old_vis_r, new_vis_r])

    title = SOURCE_NAMES.get(fname, fname)
    banner = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    combined = np.vstack([banner, combined])

    out_path = os.path.join(OUT_DIR, f"compare_{fname}")
    cv2.imwrite(out_path, combined)

    rows.append({
        "file": fname,
        "title": title,
        "old_count": len(old_boxes),
        "new_count": len(new_boxes),
        "out": out_path,
    })
    print(f"{fname:12s} | {title:45s} | old={len(old_boxes):4d} | new={len(new_boxes):4d}")

# also build one big grid of all 10 comparisons stacked vertically, downscaled
panels = [cv2.imread(r["out"]) for r in rows]
max_w = max(p.shape[1] for p in panels)
panels_r = []
for p in panels:
    if p.shape[1] != max_w:
        s = max_w / p.shape[1]
        p = cv2.resize(p, (max_w, int(p.shape[0] * s)))
    panels_r.append(p)
grid = np.vstack(panels_r)
grid_path = os.path.join(OUT_DIR, "_ALL_comparisons_grid.jpg")
cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("\nWrote grid:", grid_path)

import json
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(rows, f, indent=2)
