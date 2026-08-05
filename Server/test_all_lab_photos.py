import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

FILES = ["IMG_1021_3.jpg", "IMG_1031_2.jpg", "IMG_1034_3.jpg"]
DIR = "/Users/satriabaladewaharahap/bacteriaserius/lab_photos"
model = YOLO("models_trained/YOLO/counter/best.pt")

fig, axes = plt.subplots(1, 3, figsize=(21, 12))
for i, fname in enumerate(FILES):
    path = f"{DIR}/{fname}"
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(source=path, imgsz=1536, conf=0.001, max_det=1000, device="cpu", verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    keep = confs >= 0.4
    vis = img_rgb.copy()
    for (x1, y1, x2, y2), c in zip(boxes[keep], confs[keep]):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 60), 10)
        cv2.putText(vis, f"{c:.2f}", (int(x1), max(30, int(y1) - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 60), 4)
    axes[i].imshow(vis)
    axes[i].set_title(f"{fname}\nfinal count (conf>=0.4): {int(keep.sum())} / raw candidates: {len(confs)}", fontsize=11)
    axes[i].axis("off")

plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/all_lab_photos_result.png"
plt.savefig(out_path, dpi=100, bbox_inches="tight")
print("Saved to", out_path)
