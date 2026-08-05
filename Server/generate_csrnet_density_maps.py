"""Converts our existing YOLO-format box labels (2600 train + 500 val images,
same AGAR + SAM-Copy-Paste-synthetic mix used for every YOLO fine-tune in
this project) into CSRNet-style density maps -- no new annotation needed,
since we already know each colony's center from its box.

For each image: resize to a fixed 768x768 working resolution, place a
Gaussian at each colony's (rescaled) center on a full-res canvas, then
downsample by 8x (matching CSRNet's frontend stride) and rescale by 64x to
preserve the total count after downsampling -- the same convention used by
the original CSRNet repo's own data prep.
"""
import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter

SRC_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/synthetic_dataset")
OUT_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/csrnet_dataset")
WORKING_SIZE = 768
SIGMA = 4.0


def make_density_map(points, size):
    """points: list of (x, y) in the WORKING_SIZE canvas. Returns a
    (size, size) density map whose sum equals len(points)."""
    canvas = np.zeros((size, size), dtype=np.float32)
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            canvas[yi, xi] += 1.0
    if canvas.sum() == 0:
        return canvas
    density = gaussian_filter(canvas, sigma=SIGMA, mode="constant")
    return density


def process_split(split):
    img_dir = SRC_DIR / "images" / split
    lbl_dir = SRC_DIR / "labels" / split
    out_img_dir = OUT_DIR / "images" / split
    out_den_dir = OUT_DIR / "density" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_den_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    written = 0
    total_colonies = 0
    for i, img_path in enumerate(img_files):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_resized = cv2.resize(img, (WORKING_SIZE, WORKING_SIZE), interpolation=cv2.INTER_AREA)

        points = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, w, h = map(float, line.split())
                points.append((cx * WORKING_SIZE, cy * WORKING_SIZE))

        full_res_density = make_density_map(points, WORKING_SIZE)
        target_size = WORKING_SIZE // 8
        density_small = cv2.resize(full_res_density, (target_size, target_size),
                                    interpolation=cv2.INTER_CUBIC) * 64

        # cv2.resize + gaussian blur can introduce tiny negative ripples; clip
        density_small = np.clip(density_small, 0, None)

        cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(out_den_dir / f"{stem}.npy", density_small.astype(np.float32))

        written += 1
        total_colonies += len(points)
        if (i + 1) % 300 == 0:
            print(f"  [{split}] {i+1}/{len(img_files)} done "
                  f"(density sum check: target={len(points)}, "
                  f"actual={density_small.sum():.1f})")

    print(f"{split}: wrote {written} images, {total_colonies} total colonies")


if __name__ == "__main__":
    process_split("train")
    process_split("val")
    print("Done. Output:", OUT_DIR)
