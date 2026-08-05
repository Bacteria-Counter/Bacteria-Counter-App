import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

IMG_PATH = "/Users/satriabaladewaharahap/bacteriaserius/lab_photos/IMG_1034_3.jpg"
CONF = 0.4
IMGSZ = 1536

model = YOLO("models_trained/YOLO/counter/best.pt")

img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img_bgr.shape[:2]

res = model.predict(source=IMG_PATH, imgsz=IMGSZ, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
boxes = res.boxes.xyxy.cpu().numpy()
confs = res.boxes.conf.cpu().numpy()
keep = confs >= CONF
final_boxes = boxes[keep]
final_count = int(keep.sum())

vis = img_rgb.copy()
for (x1, y1, x2, y2) in final_boxes:
    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 60), 6)

print(f"Image resolution: {W}x{H}")
print(f"Raw candidates (conf>=0.001): {len(confs)}")
print(f"Final count (conf>=0.4): {final_count}")
print(f"Downscale factor to reach {IMGSZ}px: {max(W,H)/IMGSZ:.2f}x")

# estimate typical colony size in the ORIGINAL image (from box sizes we did detect)
if len(final_boxes) > 0:
    ws = final_boxes[:, 2] - final_boxes[:, 0]
    hs = final_boxes[:, 3] - final_boxes[:, 1]
    print(f"Detected colony box size range (px, original res): "
          f"width {ws.min():.0f}-{ws.max():.0f}, height {hs.min():.0f}-{hs.max():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 12))
axes[0].imshow(img_rgb)
axes[0].set_title(f"Original ({W}x{H})", fontsize=12)
axes[0].axis("off")
axes[1].imshow(vis)
axes[1].set_title(f"YOLO detections (conf>=0.4): {final_count}", fontsize=12, fontweight="bold")
axes[1].axis("off")
plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/real_lab_photo_result.png"
plt.savefig(out_path, dpi=110, bbox_inches="tight")
print("Saved to", out_path)
