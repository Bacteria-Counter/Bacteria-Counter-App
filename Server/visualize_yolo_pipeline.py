import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, ".")

SAMPLE_ID = "2999"
RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
IMG_PATH = RAW_DIR / f"{SAMPLE_ID}.jpg"
JSON_PATH = RAW_DIR / f"{SAMPLE_ID}.json"
CONF_THRESH = 0.4
IMGSZ = 1536

gt = json.loads(JSON_PATH.read_text())
gt_count = int(gt["colonies_number"])
gt_boxes = gt["labels"]

img_bgr = cv2.imread(str(IMG_PATH))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_bgr.shape[:2]

# --- Panel 2: preprocessing letterbox (what actually feeds the YOLO model) ---
scale = IMGSZ / max(H, W)
new_w, new_h = int(round(W * scale)), int(round(H * scale))
resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)  # ultralytics letterbox pad color
pad_x = (IMGSZ - new_w) // 2
pad_y = (IMGSZ - new_h) // 2
canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
letterboxed_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

# --- Panel 3: raw detections at very low confidence (pre-filter noise) ---
model = YOLO("models_trained/YOLO/counter/best.pt")
res = model.predict(source=str(IMG_PATH), imgsz=IMGSZ, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
boxes_xyxy = res.boxes.xyxy.cpu().numpy()
confs = res.boxes.conf.cpu().numpy()

raw_vis = img_rgb.copy()
for (x1, y1, x2, y2), c in zip(boxes_xyxy, confs):
    color = (255, 80, 80) if c < CONF_THRESH else (255, 200, 0)
    cv2.rectangle(raw_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
cv2.putText(raw_vis, f"All candidate boxes: {len(confs)} (conf>=0.001)", (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 10)
cv2.putText(raw_vis, f"All candidate boxes: {len(confs)} (conf>=0.001)", (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (200, 0, 0), 5)

# --- Panel 4/5: filtered final detections + count ---
keep = confs >= CONF_THRESH
final_boxes = boxes_xyxy[keep]
final_count = int(keep.sum())

final_vis = img_rgb.copy()
for (x1, y1, x2, y2) in final_boxes:
    cv2.rectangle(final_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 60), 5)
label = f"Predicted count: {final_count}  |  Ground truth: {gt_count}"
cv2.putText(final_vis, label, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 10)
cv2.putText(final_vis, label, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 140, 0), 5)

# --- Ground-truth overlay for reference ---
# NOTE: AGAR's x,y is the TOP-LEFT corner of the box, not the centroid.
# Verified visually in scripts/verify_asymmetric_box.py (bacteriacounter project):
# the top-left interpretation wraps real colonies precisely; centroid does not.
gt_vis = img_rgb.copy()
for lbl in gt_boxes:
    x, y, w, h = lbl["x"], lbl["y"], lbl["width"], lbl["height"]
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    cv2.rectangle(gt_vis, (x1, y1), (x2, y2), (0, 120, 255), 5)
cv2.putText(gt_vis, f"Ground truth annotations: {gt_count}", (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 10)
cv2.putText(gt_vis, f"Ground truth annotations: {gt_count}", (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 90, 200), 5)

# --- Build montage ---
fig, axes = plt.subplots(2, 3, figsize=(21, 13))

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title(f"1. Original Dish ({W}x{H} px)\nSample ID: {SAMPLE_ID} | background={gt['background']}",
                      fontsize=13, fontweight="bold")
axes[0, 0].axis("off")

axes[0, 1].imshow(letterboxed_rgb)
axes[0, 1].set_title(f"2. Preprocessing (Letterbox)\nResized+padded to {IMGSZ}x{IMGSZ} (YOLO input)",
                      fontsize=13, fontweight="bold")
axes[0, 1].axis("off")

axes[0, 2].imshow(gt_vis)
axes[0, 2].set_title("3. Ground Truth (AGAR annotations)\nManual expert-labeled boxes", fontsize=13, fontweight="bold")
axes[0, 2].axis("off")

axes[1, 0].imshow(raw_vis)
axes[1, 0].set_title("4. Diagnostic view only (conf>=0.001)\nNOT the model's real answer - shows every raw candidate\nOrange=kept later, Red=will be dropped",
                      fontsize=13, fontweight="bold")
axes[1, 0].axis("off")

axes[1, 1].imshow(final_vis)
axes[1, 1].set_title(f"5. Filtered Detections (conf>={CONF_THRESH})\nFinal boxes after confidence threshold",
                      fontsize=13, fontweight="bold")
axes[1, 1].axis("off")

# Panel 6: side-by-side error summary
axes[1, 2].axis("off")
summary_text = (
    f"PIPELINE SUMMARY\n\n"
    f"Sample: {SAMPLE_ID}.jpg\n"
    f"Ground truth colonies: {gt_count}\n"
    f"Raw candidate boxes: {len(confs)}\n"
    f"Confidence threshold: {CONF_THRESH}\n"
    f"Final predicted count: {final_count}\n"
    f"Absolute error: {abs(final_count - gt_count)}\n\n"
    f"Model: models_trained/YOLO/counter/best.pt\n"
    f"Inference size: {IMGSZ}x{IMGSZ}\n"
    f"Device: CPU (MacBook, Apple Silicon)"
)
axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                fontsize=15, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#eaffea", edgecolor="green"))

plt.suptitle("Husseinchr YOLO Colony-Counting Pipeline — End to End", fontsize=17, fontweight="bold")
plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/yolo_pipeline_walkthrough.png"
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("Saved to", out_path)
print(f"GT={gt_count}, raw_candidates={len(confs)}, final_pred={final_count}")
