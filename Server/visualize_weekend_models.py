"""Builds visual comparison panels for the 3 weekend-trained models against
ground truth, on representative samples per background category. Each panel
shows: ground truth boxes (red, from AGAR's own annotations) next to each
model's prediction (on the preprocessed input it was actually trained on).
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
OUT_DIR = Path("/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/weekend_viz")
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODELS = {
    "dog_blend": "/Users/satriabaladewaharahap/bacteriaserius/DOG BLEND MODEL/best-2.pt",
    "clahe": "/Users/satriabaladewaharahap/bacteriaserius/CLAHE MODEL/best-2.pt",
    "lab_ab": "/Users/satriabaladewaharahap/bacteriaserius/LAB AB MODEL/best-2.pt",
}

SAMPLES = ["515", "2724", "12497", "11971"]  # bright, dark, vague, vague (the two worst vague cases)


def dog_blend(img_bgr, sigma1=2, sigma2=12, strength=1.0):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    dog = g1 - g2
    out = img_bgr.astype(np.float32) + (strength * dog)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_lab(img_bgr, clip=3.0, tile=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def lab_ab_isolate(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_stretch = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_stretch = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([a_stretch, b_stretch, np.full_like(a, 128)])


PREPROC = {"dog_blend": dog_blend, "clahe": clahe_lab, "lab_ab": lab_ab_isolate}


def draw_gt(img_bgr, meta):
    out = img_bgr.copy()
    for lbl in meta["labels"]:
        x, y, w, h = lbl["x"], lbl["y"], lbl["width"], lbl["height"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return out


def draw_pred(img_bgr, boxes):
    out = img_bgr.copy()
    if boxes is not None:
        for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy():
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 3)
    return out


def main():
    models = {name: YOLO(ckpt) for name, ckpt in MODELS.items()}

    for sid in SAMPLES:
        img_path = RAW_DIR / f"{sid}.jpg"
        meta_path = RAW_DIR / f"{sid}.json"
        img = cv2.imread(str(img_path))
        meta = json.load(open(meta_path))
        gt_count = meta["colonies_number"]
        bg = meta["background"]

        gt_vis = draw_gt(img, meta)
        panels = [("ground truth", gt_vis, gt_count)]

        for name, model in models.items():
            fn = PREPROC[name]
            proc = fn(img)
            res = model.predict(source=proc, imgsz=1536, conf=0.4, max_det=1000, verbose=False)[0]
            pred_count = 0 if res.boxes is None else len(res.boxes)
            pred_vis = draw_pred(proc, res.boxes)
            panels.append((name, pred_vis, pred_count))

        h, w = img.shape[:2]
        scale = 480 / h
        resized = []
        for label, panel_img, count in panels:
            p = cv2.resize(panel_img, (int(w * scale), 480))
            cv2.putText(p, f"{label}: {count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(p, f"{label}: {count}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 1, cv2.LINE_AA)
            resized.append(p)
        combined = np.hstack(resized)
        banner = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, f"Sample #{sid}  (background={bg}, ground truth={gt_count})",
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        combined = np.vstack([banner, combined])

        out_path = OUT_DIR / f"compare_{sid}.jpg"
        cv2.imwrite(str(out_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"Sample {sid} (bg={bg}, gt={gt_count}): "
              + ", ".join(f"{label}={count}" for label, _, count in panels))

    print("\nDone. Output:", OUT_DIR)


if __name__ == "__main__":
    main()
