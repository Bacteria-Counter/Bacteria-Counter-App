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

total_boxes = 0
outside_dish_boxes = 0
on_rim_boxes = 0
per_image_outside = []

for row in sample:
    sample_id = row["sample_id"]
    img_path = RAW_DIR / f"{sample_id}.jpg"
    img_bgr = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # dish silhouette: non-black region (background outside the dish is near-pure black)
    _, nonblack = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(nonblack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(c)

    res = model.predict(source=str(img_path), imgsz=1536, conf=0.4, max_det=1000, device="cpu", verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))

    img_outside = 0
    for (x1, y1, x2, y2) in boxes:
        bx, by = (x1 + x2) / 2, (y1 + y2) / 2
        dist_from_center = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        total_boxes += 1
        if dist_from_center > radius:
            outside_dish_boxes += 1
            img_outside += 1
        elif dist_from_center > radius * 0.92:  # rim/bezel zone (outer ~8% of radius)
            on_rim_boxes += 1
    per_image_outside.append((sample_id, len(boxes), img_outside))

print(f"Sampled images: {len(sample)}")
print(f"Total final detections (conf>=0.4): {total_boxes}")
print(f"Detections with box-center OUTSIDE the dish silhouette (pure background): {outside_dish_boxes}")
print(f"Detections with box-center ON the rim/bezel zone (outer ring): {on_rim_boxes}")
print()
print("Per-image breakdown (sample_id, total_boxes, boxes_outside_dish):")
for sid, tot, out in per_image_outside:
    if out > 0:
        print(f"  {sid}: total={tot} outside={out}")
if outside_dish_boxes == 0 and on_rim_boxes == 0:
    print("  (none flagged in this sample)")
