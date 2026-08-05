"""Mixes real AGAR images (with their real ground-truth boxes) into the
synthetic training set, so fine-tuning doesn't overfit to Copy-Paste
compositing artifacts (blend edges, unnatural lighting) and forgets what a
real colony generally looks like. Only pulls from the AGAR `train` split —
never the 700-image `test` split used for all prior benchmarking in this
project, so that set stays clean for future re-validation.
"""
import csv
import json
import random
import shutil
from pathlib import Path

random.seed(11)

MANIFEST = Path("/Users/satriabaladewaharahap/Downloads/bacteriacounter/kaggle_manifest.csv")
AGAR_RAW_DIR = Path("/Users/satriabaladewaharahap/Downloads/AGAR_dataset/dataset")
SYNTHETIC_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/synthetic_dataset")
N_REAL_TRAIN = 600
N_REAL_VAL = 100


def convert_and_copy(rows, images_dir, labels_dir, prefix):
    written = 0
    for row in rows:
        sid = row["sample_id"]
        src_img = AGAR_RAW_DIR / f"{sid}.jpg"
        if not src_img.exists():
            continue
        labels = json.loads(row["labels_json"])
        if not labels:
            continue
        orig_w, orig_h = int(row["orig_width"]), int(row["orig_height"])

        yolo_lines = []
        for lbl in labels:
            x, y, w, h = lbl["x"], lbl["y"], lbl["width"], lbl["height"]
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            wn = w / orig_w
            hn = h / orig_h
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

        stem = f"{prefix}_{sid}"
        shutil.copy(src_img, images_dir / f"{stem}.jpg")
        (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n")
        written += 1
    return written


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    train_rows = [r for r in rows if r["split"] == "train" and int(r["cfu_count"]) > 0]
    random.shuffle(train_rows)

    real_train = train_rows[:N_REAL_TRAIN]
    real_val = train_rows[N_REAL_TRAIN:N_REAL_TRAIN + N_REAL_VAL]

    n1 = convert_and_copy(real_train, SYNTHETIC_DIR / "images" / "train",
                          SYNTHETIC_DIR / "labels" / "train", "real")
    n2 = convert_and_copy(real_val, SYNTHETIC_DIR / "images" / "val",
                          SYNTHETIC_DIR / "labels" / "val", "real")
    print(f"Added {n1} real AGAR images to train, {n2} to val")


if __name__ == "__main__":
    main()
