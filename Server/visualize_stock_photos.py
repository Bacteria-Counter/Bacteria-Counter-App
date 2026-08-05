import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.insert(0, ".")
from src.classical.detect_count import count_colonies

IMAGE_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius")
IMAGE_FILES = [
    "46676809-colonies-of-bacteria-from-sea-water-on-a-petri-dish-agar-plate-isolated-on-black-background-differen.jpg",
    "500_F_102366721_s7we3Wpx4TpiI9tGGdBo48oEiNC7yN3h.jpg",
    "BMEN-news-light-based-therapy-4oct2022.jpg",
    "bacteria-colonies.jpg",
    "colonies-of-pathogenic-bacteria-in-a-petri-dish-microbiological-studies-photo.jpeg",
]
CONF_THRESH = 0.4
IMGSZ = 1536

model = YOLO("models_trained/YOLO/counter/best.pt")

fig, axes = plt.subplots(len(IMAGE_FILES), 2, figsize=(14, 6 * len(IMAGE_FILES)))

for i, fname in enumerate(IMAGE_FILES):
    path = IMAGE_DIR / fname
    img_bgr = cv2.imread(str(path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Classical CV (for reference)
    classical_pred, _, _ = count_colonies(img_bgr)

    # YOLO
    res = model.predict(source=str(path), imgsz=IMGSZ, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
    boxes_xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    keep = confs >= CONF_THRESH
    final_boxes = boxes_xyxy[keep]
    yolo_count = int(keep.sum())

    yolo_vis = img_rgb.copy()
    for (x1, y1, x2, y2), c in zip(final_boxes, confs[keep]):
        cv2.rectangle(yolo_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 60), 3)
        cv2.putText(yolo_vis, f"{c:.2f}", (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    label = f"YOLO count: {yolo_count}"
    cv2.putText(yolo_vis, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 5)
    cv2.putText(yolo_vis, label, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 140, 0), 2)

    axes[i, 0].imshow(img_rgb)
    axes[i, 0].set_title(f"{fname}\n({img_bgr.shape[1]}x{img_bgr.shape[0]} px)", fontsize=10)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(yolo_vis)
    axes[i, 1].set_title(f"YOLO pred: {yolo_count}  |  Classical CV pred: {classical_pred}\n"
                          f"(raw candidates before filter: {len(confs)})", fontsize=10, fontweight="bold")
    axes[i, 1].axis("off")

    print(f"{fname}: YOLO={yolo_count} (raw={len(confs)}), Classical={classical_pred}")

plt.suptitle("Husseinchr YOLO Counter on Out-of-Distribution Stock Photos", fontsize=15, fontweight="bold")
plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/stock_photo_results.png"
plt.savefig(out_path, dpi=100, bbox_inches="tight")
print("Saved to", out_path)
