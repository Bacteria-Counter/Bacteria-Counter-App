from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.agar import load_agar_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", default=24, type=int)
    parser.add_argument("--category", default="", choices=("", "countable", "empty", "uncountable"))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-side", default=1800, type=int)
    return parser.parse_args()


def resize_for_review(image, max_side: int):
    height, width = image.shape[:2]
    scale = min(max_side / max(height, width), 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def draw_annotation(image, annotation: dict, category: str):
    output = image.copy()
    for label in annotation.get("labels", []):
        x1 = int(label["x"])
        y1 = int(label["y"])
        x2 = x1 + int(label["width"])
        y2 = y1 + int(label["height"])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
    text = (
        f"{category} | count={annotation.get('colonies_number')} | "
        f"labels={len(annotation.get('labels', []))} | classes={','.join(annotation.get('classes', []))}"
    )
    cv2.putText(output, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6)
    cv2.putText(output, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    return output


def main() -> None:
    args = parse_args()
    split_df = pd.read_csv(args.split_csv)
    if args.category:
        split_df = split_df[split_df["category"] == args.category]

    image_paths = split_df["image_path"].tolist()
    rng = random.Random(args.seed)
    rng.shuffle(image_paths)
    samples = image_paths[: min(args.num_samples, len(image_paths))]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_relative in samples:
        image_path = args.dataset_dir / image_relative
        json_path = image_path.with_suffix(".json")
        annotation = load_agar_annotation(json_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        category = image_path.parent.name
        rendered = draw_annotation(image, annotation, category)
        rendered = resize_for_review(rendered, args.max_side)
        output_path = args.output_dir / f"{category}_{image_path.stem}.jpg"
        ok = cv2.imwrite(str(output_path), rendered)
        if not ok:
            raise IOError(f"Could not save image: {output_path}")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
