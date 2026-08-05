import sys
import time
from pathlib import Path

import cv2
import torch

torch.serialization.add_safe_globals([__import__("argparse").Namespace])
_orig_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_load

GSAM2_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/repos/Grounded-SAM-2")
sys.path.insert(0, str(GSAM2_DIR))
sys.path.insert(0, "/Users/satriabaladewaharahap/bacteriaserius/repos/ColonyGroundedSam2")

from groundingdino.util.inference import load_model, predict
from utils import load_image2

BG_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/background_petridish_jpg")
DEVICE = "cpu"
GROUNDING_DINO_CONFIG = str(GSAM2_DIR / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = "/Users/satriabaladewaharahap/bacteriaserius/repos/ColonyGroundedSam2/checkpoints/colony_gd.pth"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25


def main():
    print("Loading fine-tuned Grounding DINO...")
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )

    files = sorted(BG_DIR.glob("*.jpg"))
    print(f"Checking {len(files)} real empty-background photos (expect ~0 detections each)")
    flagged = []
    t0 = time.time()
    for f in files:
        image_shape = cv2.imread(str(f)).shape
        x1, y1, x2, y2 = 0, 0, image_shape[1], image_shape[0]
        image_source, image = load_image2(str(f), x1, x2, y1, y2)
        boxes, confidences, labels = predict(
            model=grounding_model, image=image, caption="microbial colony.",
            box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, device=DEVICE,
        )
        count = len(boxes)
        marker = "  <-- FLAGGED" if count > 0 else ""
        print(f"{f.name}: {count} false detections{marker} (t={time.time()-t0:.0f}s)")
        if count > 0:
            flagged.append((f.name, count))

    print()
    if flagged:
        print(f"WARNING: {len(flagged)}/{len(files)} empty backgrounds produced false detections.")
        for name, c in flagged:
            print(f"  {name}: {c}")
    else:
        print("All clear -- zero false detections on every real empty background.")


if __name__ == "__main__":
    main()
