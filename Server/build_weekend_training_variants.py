"""Builds 3 preprocessed dataset variants for a controlled weekend experiment,
one per M3 Max device, to settle whether baking a preprocessing technique
into TRAINING data (not just bolting it onto inference) actually improves
accuracy -- since inference-only CLAHE/Otsu/Sauvola showed this project that
mismatched train/inference distributions hurt, regardless of how good a
technique looks visually or on internet photos.

All 3 variants share the exact same underlying data (2000 synthetic + 600
real AGAR train, 400 synthetic + 100 real AGAR val -- the same mix used for
the already-validated colony_finetuned_best.pt) and the exact same starting
checkpoint (the ORIGINAL AGAR-only best.pt, not the already-fine-tuned one --
matching how the first fine-tune was done, so results are comparable).
Labels are untouched (preprocessing doesn't change image dimensions or
colony positions) -- only the image pixels differ per variant.

Variants:
  A. dog_blend  -- Difference-of-Gaussians edge signal blended into RGB
  B. clahe      -- CLAHE contrast enhancement (this time baked into training,
                   not just applied at inference on an already-trained model)
  C. lab_ab     -- LAB a/b chrominance-channel isolation (the technique that
                   was visually judged as mixed/risky but never actually
                   validated with real numbers -- this settles it properly)
"""
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/synthetic_dataset")
CHECKPOINT_SRC = Path("/Users/satriabaladewaharahap/bacteriaserius/repos/bacterial-colony-detection/models_trained/YOLO/counter/best.pt")
OUT_ROOT = Path("/Users/satriabaladewaharahap/bacteriaserius/weekend_training_packages")


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


VARIANTS = {
    "dog_blend": dog_blend,
    "clahe": clahe_lab,
    "lab_ab": lab_ab_isolate,
}

DATA_YAML = """path: .
train: images/train
val: images/val
names:
  0: colony
"""


def main():
    for name in VARIANTS:
        pkg_dir = OUT_ROOT / name
        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (pkg_dir / sub).mkdir(parents=True, exist_ok=True)
        (pkg_dir / "data.yaml").write_text(DATA_YAML)
        shutil.copy(CHECKPOINT_SRC, pkg_dir / "best.pt")

    splits = ["train", "val"]
    total_written = {name: 0 for name in VARIANTS}
    for split in splits:
        img_files = sorted((SRC_DIR / "images" / split).glob("*.jpg"))
        print(f"--- split={split}, {len(img_files)} images ---")
        for i, img_path in enumerate(img_files):
            stem = img_path.stem
            label_path = SRC_DIR / "labels" / split / f"{stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                print("WARNING: failed to read", img_path)
                continue

            for name, fn in VARIANTS.items():
                out_img_path = OUT_ROOT / name / "images" / split / f"{stem}.jpg"
                out_lbl_path = OUT_ROOT / name / "labels" / split / f"{stem}.txt"
                processed = fn(img)
                cv2.imwrite(str(out_img_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if label_path.exists():
                    shutil.copy(label_path, out_lbl_path)
                else:
                    out_lbl_path.write_text("")
                total_written[name] += 1

            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i+1}/{len(img_files)} processed")

    print("\nDone. Images written per variant:")
    for name, count in total_written.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
