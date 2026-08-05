import csv
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MANIFEST = Path("/Users/satriabaladewaharahap/Downloads/bacteriacounter/kaggle_manifest.csv")
RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")

random.seed(7)
rows = list(csv.DictReader(open(MANIFEST)))
test_rows = [r for r in rows if r["split"] == "test"]
sample = random.sample(test_rows, 60)

model = YOLO("models_trained/YOLO/counter/best.pt")

rim_crops = []
for row in sample:
    sample_id = row["sample_id"]
    img_path = RAW_DIR / f"{sample_id}.jpg"
    img_bgr = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, nonblack = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(nonblack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(c)

    res = model.predict(source=str(img_path), imgsz=1536, conf=0.4, max_det=1000, device="cpu", verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))

    for (x1, y1, x2, y2) in boxes:
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        dist_from_center = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        if radius * 0.92 < dist_from_center <= radius:
            pad = 60
            cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
            cx2, cy2 = min(img_bgr.shape[1], int(x2 + pad)), min(img_bgr.shape[0], int(y2 + pad))
            crop = img_bgr[cy1:cy2, cx1:cx2].copy()
            cv2.rectangle(crop, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 255, 0), 2)
            rim_crops.append((f"{sample_id}", crop))

print(f"Found {len(rim_crops)} rim-zone detections, saving crops for visual inspection")

import matplotlib.pyplot as plt
n = len(rim_crops)
cols = 5
rows_n = (n + cols - 1) // cols
fig, axes = plt.subplots(rows_n, cols, figsize=(3 * cols, 3 * rows_n))
axes = axes.flatten() if n > 1 else [axes]
for i, (sid, crop) in enumerate(rim_crops):
    axes[i].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[i].set_title(sid, fontsize=9)
    axes[i].axis("off")
for j in range(len(rim_crops), len(axes)):
    axes[j].axis("off")
plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/rim_detections_check.png"
plt.savefig(out_path, dpi=100, bbox_inches="tight")
print("Saved to", out_path)
