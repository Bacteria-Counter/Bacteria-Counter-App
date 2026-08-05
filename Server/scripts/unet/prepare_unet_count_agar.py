from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unet.targets import PseudoMaskConfig, write_unet_count_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ellipse-shrink-ratio", default=0.9, type=float)
    parser.add_argument("--min-radius", default=2, type=int)
    parser.add_argument("--max-images", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_unet_count_targets(
        dataset_dir=args.dataset_dir,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        config=PseudoMaskConfig(
            ellipse_shrink_ratio=args.ellipse_shrink_ratio,
            min_radius=args.min_radius,
        ),
        max_images=args.max_images,
    )
    print("UNet AGAR counting target preparation summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved UNet counting manifest to: {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
