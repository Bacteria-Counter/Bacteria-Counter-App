"""Local M3 Max training script for one weekend dataset variant
(dog_blend / clahe / lab_ab). Run this from inside the unzipped package
directory on each device -- same 2-stage fine-tune recipe used for the
already-validated colony_finetuned_best.pt (freeze backbone 10 epochs, then
unfreeze everything for 70 epochs), just targeting MPS instead of Kaggle's
CUDA T4x2.

Usage (from inside the package folder, e.g. dog_blend/):
    /path/to/.venv/bin/python train_weekend_variant.py

Expects, in the current directory:
    data.yaml, best.pt (starting checkpoint), images/, labels/

Writes stage 1 + stage 2 runs under ./runs/, and copies the final checkpoint
to ./finetuned_best.pt when done.
"""
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA_YAML = HERE / "data.yaml"
CHECKPOINT = HERE / "best.pt"

assert DATA_YAML.exists(), f"Missing {DATA_YAML} -- run this from inside the package folder"
assert CHECKPOINT.exists(), f"Missing {CHECKPOINT} -- run this from inside the package folder"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

t0 = time.time()

print("=== Stage 1: freeze backbone, 10 epochs ===")
model = YOLO(str(CHECKPOINT))
model.train(
    data=str(DATA_YAML),
    epochs=10,
    imgsz=1536,
    batch=8,  # lower to 4 if you hit a memory error
    device=DEVICE,
    freeze=10,
    patience=10,
    optimizer="auto",
    lr0=0.003,
    project=str(HERE / "runs"),
    name="stage1",
    exist_ok=True,
    seed=42,
)
stage1_checkpoint = HERE / "runs" / "stage1" / "weights" / "last.pt"
print(f"Stage 1 done in {time.time()-t0:.0f}s")

print("=== Stage 2: unfreeze everything, 70 epochs ===")
t1 = time.time()
model2 = YOLO(str(stage1_checkpoint))
model2.train(
    data=str(DATA_YAML),
    epochs=70,
    imgsz=1536,
    batch=8,
    device=DEVICE,
    patience=15,
    optimizer="auto",
    lr0=0.001,
    project=str(HERE / "runs"),
    name="stage2",
    exist_ok=True,
    seed=42,
)
final_checkpoint = HERE / "runs" / "stage2" / "weights" / "best.pt"
print(f"Stage 2 done in {time.time()-t1:.0f}s")
print(f"Total training time: {time.time()-t0:.0f}s")

shutil.copy(final_checkpoint, HERE / "finetuned_best.pt")
print(f"Copied final checkpoint to {HERE / 'finetuned_best.pt'}")
