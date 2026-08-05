"""Standalone CLI: Grounding DINO colony detection only (no SAM2 -- we only
need the count + boxes, not segmentation masks). Prints a single JSON line
to stdout so a caller in a different Python environment (server.py's own
venv, which doesn't have groundingdino installed) can invoke this via
subprocess and parse the result.

Usage: .venv/bin/python infer_count_only.py <image_path>
"""
import json
import os
import sys
from pathlib import Path

import cv2
import torch

torch.serialization.add_safe_globals([__import__("argparse").Namespace])
_orig_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_load

# Defaults to this machine's checkout; teammates should set GSAM2_REPO_DIR
# instead of editing this file.
GSAM2_DIR = Path(os.environ.get(
    "GSAM2_REPO_DIR",
    "/Users/satriabaladewaharahap/bacteriaserius/repos/Grounded-SAM-2",
))
sys.path.insert(0, str(GSAM2_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
from utils import load_image2

DEVICE = "cpu"
GROUNDING_DINO_CONFIG = str(GSAM2_DIR / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = str(Path(__file__).resolve().parent / "checkpoints" / "colony_gd.pth")
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25


def main():
    image_path = sys.argv[1]
    image_shape = cv2.imread(image_path).shape
    h, w = image_shape[0], image_shape[1]

    model = load_model(model_config_path=GROUNDING_DINO_CONFIG,
                        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT, device=DEVICE)

    image_source, image = load_image2(image_path, 0, w, 0, h)
    boxes, confidences, labels = predict(
        model=model, image=image, caption="microbial colony.",
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device=DEVICE,
    )
    boxes_scaled = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    conf_list = confidences.tolist() if len(confidences) else []

    result = {
        "count": len(xyxy),
        "imageWidth": w,
        "imageHeight": h,
        "boxes": xyxy.tolist(),
        "confidences": conf_list,
    }
    # Everything above may print warnings to stderr; only this line on
    # stdout is meant to be parsed by the caller.
    print("GSAM2_RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
