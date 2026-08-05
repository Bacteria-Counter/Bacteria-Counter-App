import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from src.classical.detect_count import count_colonies, read_image
from src.unet.torch_backend import UNetSegmenter, image_to_tensor
from ultralytics import YOLO

MANIFEST = Path("/Users/satriabaladewaharahap/Downloads/bacteriacounter/kaggle_manifest.csv")
RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")

YOLO_THRESHOLDS = [0.1, 0.25, 0.4, 0.5, 0.6]


def load_test_rows():
    rows = list(csv.DictReader(open(MANIFEST)))
    test_rows = [r for r in rows if r["split"] == "test"]
    return test_rows


def unet_count(model, ckpt, img_bgr, threshold=0.5, min_area=24):
    image_size = int(ckpt.get("image_size", 256))
    tensor = image_to_tensor(img_bgr, image_size).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).numpy()[0, 0]
    mask = (probs >= threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area)


def main():
    test_rows = load_test_rows()
    print(f"Loaded {len(test_rows)} test rows from manifest")

    yolo_model = YOLO("models_trained/YOLO/counter/best.pt")
    unet_ckpt = torch.load("models_trained/U-Net/counter/model.pt", map_location="cpu")
    unet_model = UNetSegmenter(base_channels=int(unet_ckpt.get("base_channels", 32)))
    unet_model.load_state_dict(unet_ckpt["state_dict"])
    unet_model.eval()

    results = []
    t0 = time.time()
    for i, row in enumerate(test_rows):
        sample_id = row["sample_id"]
        gt = int(row["cfu_count"])
        img_path = RAW_DIR / f"{sample_id}.jpg"
        json_path = RAW_DIR / f"{sample_id}.json"

        gt_json = json.loads(json_path.read_text())
        gt_json_count = int(gt_json["colonies_number"])

        img_bgr = read_image(img_path)

        # Classical
        classical_pred, _, _ = count_colonies(img_bgr)

        # YOLO - raw detections at low conf, threshold sweep applied later
        yolo_res = yolo_model.predict(
            source=str(img_path), imgsz=1536, conf=0.001, max_det=1000,
            device="cpu", verbose=False,
        )
        boxes = yolo_res[0].boxes
        yolo_confs = boxes.conf.cpu().numpy().tolist() if boxes is not None and len(boxes) > 0 else []

        # U-Net
        unet_pred = unet_count(unet_model, unet_ckpt, img_bgr)

        row_result = {
            "sample_id": sample_id,
            "gt": gt,
            "gt_json_count": gt_json_count,
            "background": row["background"],
            "classical_pred": classical_pred,
            "unet_pred": unet_pred,
            "yolo_confs": json.dumps(yolo_confs),
        }
        results.append(row_result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(test_rows)} done, {elapsed:.1f}s elapsed, "
                  f"~{elapsed/(i+1)*len(test_rows):.0f}s total est.")

    out_path = Path("/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/husseinchr_eval_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print("Saved to", out_path)
    print("Total time:", time.time() - t0)


if __name__ == "__main__":
    main()
