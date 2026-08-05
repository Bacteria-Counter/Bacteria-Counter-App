"""Validates Colony Grounded SAM2 (Korporaal et al. 2026, zero-shot
Grounding-DINO + SAM2 pipeline) against the same 18-image stratified AGAR
ground-truth sample used for every other benchmark in this project.

Runs on CPU/MPS (this Mac has no CUDA) via Grounding DINO's pure-Python
deformable-attention fallback -- slower than the paper's GPU setup but
functionally equivalent for a correctness/accuracy check.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# The published checkpoints (SAM2 + fine-tuned Grounding DINO, both from
# trusted sources: Meta's official release and the paper authors' HF repo)
# were pickled with argparse.Namespace objects, which PyTorch >=2.6 no
# longer allowlists by default under weights_only=True. We trust the source
# here, so allow full unpickling.
torch.serialization.add_safe_globals([__import__("argparse").Namespace])
_orig_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_load

GSAM2_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/repos/Grounded-SAM-2")
sys.path.insert(0, str(GSAM2_DIR))
sys.path.insert(0, "/Users/satriabaladewaharahap/bacteriaserius/repos/ColonyGroundedSam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
from utils import load_image2

RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
SAMPLE_FILE = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/agar_clahe_sample.json"

DEVICE = "cpu"  # pure-python deformable-attn fallback; mps not verified stable for this op yet

SAM2_CHECKPOINT = str(GSAM2_DIR / "checkpoints" / "sam2_hiera_large.pt")
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
GROUNDING_DINO_CONFIG = str(GSAM2_DIR / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = "/Users/satriabaladewaharahap/bacteriaserius/repos/ColonyGroundedSam2/checkpoints/colony_gd.pth"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25


def main():
    with open(SAMPLE_FILE) as f:
        sample = json.load(f)

    print("Loading fine-tuned Grounding DINO...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )

    results = []
    t0 = time.time()
    for i, (sid, bg, gt) in enumerate(sample):
        img_path = RAW_DIR / f"{sid}.jpg"
        if not img_path.exists():
            continue

        image_shape = cv2.imread(str(img_path)).shape
        x1, y1, x2, y2 = 0, 0, image_shape[1], image_shape[0]
        image_source, image = load_image2(str(img_path), x1, x2, y1, y2)

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption="microbial colony.",
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )
        count = len(boxes)
        results.append({"sample_id": sid, "background": bg, "gt": gt, "pred": count})
        print(f"[{i+1}/{len(sample)}] {sid} (bg={bg}, gt={gt}): pred={count} "
              f"elapsed={time.time()-t0:.0f}s")

    mae = sum(abs(r["pred"] - r["gt"]) for r in results) / len(results)
    print(f"\n=== Colony Grounded SAM2 (detection only) MAE = {mae:.2f} (n={len(results)}) ===")

    out_path = "/private/tmp/claude-501/-Users-satriabaladewaharahap-bacteriaserius/777ce2d6-6708-4daa-83ab-e80c466c6e66/scratchpad/colony_grounded_sam2_agar_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
