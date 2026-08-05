import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import FastSAM

sys.path.insert(0, ".")
from fastsam_colony_count import count_colonies_fastsam

FILES = ["IMG_1021_3.jpg", "IMG_1031_2.jpg", "IMG_1034_3.jpg"]
DIR = "/Users/satriabaladewaharahap/bacteriaserius/lab_photos"

model = FastSAM("FastSAM-s.pt")

fig, axes = plt.subplots(1, 3, figsize=(21, 14))
for i, fname in enumerate(FILES):
    path = f"{DIR}/{fname}"
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    count, kept, dish = count_colonies_fastsam(model, path, imgsz=3840, min_area_frac=0.000005, max_area_frac=0.02)

    vis = img_rgb.copy()
    if dish is not None:
        dx, dy, dr = dish
        cv2.circle(vis, (int(dx), int(dy)), int(dr), (255, 0, 255), 6)
    for k in kept:
        contours, _ = cv2.findContours(k["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sub_count = k.get("sub_count", 1)
        color = (0, 255, 60) if sub_count == 1 else (255, 160, 0)
        cv2.drawContours(vis, contours, -1, color, 6)
        if sub_count > 1:
            cx, cy = k["centroid"]
            cv2.putText(vis, f"x{sub_count}", (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 140, 0), 5)

    axes[i].imshow(vis)
    axes[i].set_title(f"{fname}\nFastSAM+filter count: {count}  (magenta=detected dish boundary)", fontsize=11)
    axes[i].axis("off")

plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/fastsam_filtered_result.png"
plt.savefig(out_path, dpi=90, bbox_inches="tight")
print("Saved to", out_path)
