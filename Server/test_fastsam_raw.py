import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import FastSAM

FILES = ["IMG_1021_3.jpg", "IMG_1031_2.jpg", "IMG_1034_3.jpg"]
DIR = "/Users/satriabaladewaharahap/bacteriaserius/lab_photos"

model = FastSAM("FastSAM-s.pt")

fig, axes = plt.subplots(1, 3, figsize=(21, 12))
for i, fname in enumerate(FILES):
    path = f"{DIR}/{fname}"
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    res = model(path, device="cpu", retina_masks=True, imgsz=1536, conf=0.2, iou=0.7, verbose=False)[0]
    overlay = img_rgb.copy()
    n_masks = 0
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()  # (N, H, W) at model's internal resolution
        n_masks = len(masks)
        rng = np.random.default_rng(0)
        color_layer = np.zeros_like(overlay, dtype=np.float32)
        alpha = np.zeros((H, W), dtype=np.float32)
        for m in masks:
            mask_resized = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            color = rng.integers(60, 255, size=3)
            color_layer[mask_resized] = color
            alpha[mask_resized] = 0.5
        alpha3 = alpha[:, :, None]
        overlay = (overlay * (1 - alpha3) + color_layer * alpha3).astype(np.uint8)

    axes[i].imshow(overlay)
    axes[i].set_title(f"{fname}\nFastSAM raw masks: {n_masks}", fontsize=11)
    axes[i].axis("off")

plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/fastsam_raw_masks.png"
plt.savefig(out_path, dpi=90, bbox_inches="tight")
print("Saved to", out_path)
