from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import build_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-image-read", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory, summary = build_inventory(args.dataset_dir, read_images=not args.skip_image_read)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output_dir / "inventory.csv", index=False)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("AGAR-primary dataset summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
