from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo.export import export_agar_to_yolo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", choices=["count", "species"], required=True)
    parser.add_argument("--image-copy-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--max-train-images", default=0, type=int)
    parser.add_argument("--max-val-images", default=0, type=int)
    parser.add_argument("--max-test-images", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_agar_to_yolo(
        dataset_dir=args.dataset_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        task=args.task,
        image_copy_mode=args.image_copy_mode,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        max_test_images=args.max_test_images,
    )
    print("AGAR YOLO export summary")
    print(f"task: {summary['task']}")
    print(f"class_names: {', '.join(summary['class_names'])}")
    for split_name, split_summary in summary["splits"].items():
        print(
            f"{split_name}: images={split_summary['image_count']} "
            f"objects={split_summary['object_count']} "
            f"empty_labels={split_summary['negative_image_count']} "
            f"invalid_boxes={split_summary['invalid_box_count']}"
        )
    print(f"Saved YOLO dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
