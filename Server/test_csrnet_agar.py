"""Validates the (interrupted) CSRNet training's best.pt against the same
18-image stratified AGAR ground-truth sample used for every other benchmark
in this project, plus the real-empty-background safety check -- independent
of CSRNet's own internal (noisy, possibly overfit-prone) val_MAE metric.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from csrnet_model import CSRNet

CKPT_PATH = "/Users/satriabaladewaharahap/bacteriaserius/csrnet_runs/best.pt"
RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
SAMPLE_FILE = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/agar_clahe_sample.json"
BG_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/background_petridish_jpg")

WORKING_SIZE = 768
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def predict_count(model, device, img_bgr):
    img_resized = cv2.resize(img_bgr, (WORKING_SIZE, WORKING_SIZE), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_t = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        density = model(img_t)
    return float(density.sum().item())


def main():
    device = torch.device("cpu")
    model = CSRNet(load_weights=True).to(device)
    state_dict = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded checkpoint OK")

    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    results = []
    for sid, bg, gt in sample:
        img = cv2.imread(str(RAW_DIR / f"{sid}.jpg"))
        pred = predict_count(model, device, img)
        results.append({"sample_id": sid, "background": bg, "gt": gt, "pred": pred})
        print(f"{sid} (bg={bg}, gt={gt}): pred={pred:.1f}")

    mae = sum(abs(r["pred"] - r["gt"]) for r in results) / len(results)
    print(f"\nOverall MAE = {mae:.2f}")
    for bgcat in ["bright", "dark", "vague"]:
        sub = [r for r in results if r["background"] == bgcat]
        bgmae = sum(abs(r["pred"] - r["gt"]) for r in sub) / len(sub)
        print(f"{bgcat}: MAE = {bgmae:.2f}")

    with open("/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/csrnet_agar_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # real-domain safety check
    print("\n=== Real empty-background check (36 photos, expect ~0) ===")
    files = sorted(BG_DIR.glob("*.jpg"))
    flagged = []
    for fp in files:
        img = cv2.imread(str(fp))
        pred = predict_count(model, device, img)
        if pred > 1.0:  # density-map sums are continuous, not integer counts; use a small tolerance
            flagged.append((fp.name, pred))
    print(f"{len(flagged)}/{len(files)} flagged (pred > 1.0)")
    for name, pred in flagged:
        print(f"  {name}: {pred:.2f}")


if __name__ == "__main__":
    main()
