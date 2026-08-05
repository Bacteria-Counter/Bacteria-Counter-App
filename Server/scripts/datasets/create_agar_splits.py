from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import build_inventory, split_inventory, write_split_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", default=0.70, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--test-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--skip-image-read", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory, summary = build_inventory(args.dataset_dir, read_images=not args.skip_image_read)
    if inventory.empty:
        raise ValueError(
            f"No AGAR image records found under {args.dataset_dir}. "
            "Check that the path contains data/countable, data/empty, and data/uncountable."
        )
    split_df = split_inventory(
        inventory,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    write_split_files(split_df, args.output_dir)
    split_summary = {
        "dataset": summary,
        "splits": split_df.groupby("split").size().astype(int).to_dict(),
        "categories_by_split": nested_count_dict(split_df, "split", "category"),
        "classes_by_split": nested_count_dict(split_df, "split", "primary_class"),
        "task_splits": {
            "detect_count": split_counts(split_df[split_df["detection_eligible"]]),
            "species_image": split_counts(split_df[split_df["species_image_eligible"]]),
            "countable_only": split_counts(split_df[split_df["category"] == "countable"]),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    print("AGAR-primary split summary")
    print(split_df.groupby("split").size().to_string())
    print("")
    print(split_df.groupby(["split", "category"]).size().to_string())
    print(f"Saved splits to: {args.output_dir}")


def nested_count_dict(frame, outer_column: str, inner_column: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    counts = frame.groupby([outer_column, inner_column]).size()
    for (outer, inner), value in counts.items():
        result.setdefault(str(outer), {})[str(inner)] = int(value)
    return result


def split_counts(frame) -> dict[str, int]:
    return {str(split): int(count) for split, count in frame.groupby("split").size().items()}


if __name__ == "__main__":
    main()
