from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.agar_evaluation import load_count_config, run_agar_count_evaluation
from src.classical.detect_count import (
    ClassicalCountConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-json", default=None, type=Path)
    parser.add_argument("--threshold-method", default="otsu", choices=("otsu", "adaptive"))
    parser.add_argument("--threshold-polarity", default="dark", choices=("dark", "bright"))
    parser.add_argument("--gaussian-kernel", default=5, type=int)
    parser.add_argument("--clahe-clip-limit", default=2.0, type=float)
    parser.add_argument("--clahe-tile-grid-size", default=8, type=int)
    parser.add_argument("--background-kernel", default=61, type=int)
    parser.add_argument("--adaptive-block-size", default=51, type=int)
    parser.add_argument("--adaptive-c", default=2.0, type=float)
    parser.add_argument("--morph-open-kernel", default=3, type=int)
    parser.add_argument("--morph-close-kernel", default=3, type=int)
    parser.add_argument("--min-area", default=12.0, type=float)
    parser.add_argument("--max-area", default=10000.0, type=float)
    parser.add_argument("--min-circularity", default=0.05, type=float)
    parser.add_argument("--use-plate-mask", action="store_true")
    parser.add_argument("--plate-margin-ratio", default=0.04, type=float)
    parser.add_argument("--use-peak-count-estimation", action="store_true")
    parser.add_argument("--large-component-min-area", default=1200.0, type=float)
    parser.add_argument("--peak-blur-kernel", default=5, type=int)
    parser.add_argument("--peak-local-max-kernel", default=9, type=int)
    parser.add_argument("--peak-relative-threshold", default=0.45, type=float)
    parser.add_argument("--peak-min-distance", default=2.5, type=float)
    parser.add_argument("--disable-area-count-estimation", action="store_true")
    parser.add_argument("--area-count-scale", default=1.6, type=float)
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-limit", default=40, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_count_config(args.config_json) if args.config_json else config_from_args(args)
    evaluation = run_agar_count_evaluation(
        dataset_dir=args.dataset_dir,
        split_csv=args.split_csv,
        config=config,
        output_dir=args.output_dir,
        max_images=args.max_images,
        save_overlays=args.save_overlays,
        overlay_limit=args.overlay_limit,
    )

    print("Classical AGAR detection/counting evaluation summary")
    for key, value in evaluation.metrics.items():
        if key != "config":
            print(f"{key}: {value}")
    print(f"Saved predictions and metrics to: {args.output_dir}")


def config_from_args(args: argparse.Namespace) -> ClassicalCountConfig:
    return ClassicalCountConfig(
        gaussian_kernel=args.gaussian_kernel,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
        background_kernel=args.background_kernel,
        threshold_method=args.threshold_method,
        threshold_polarity=args.threshold_polarity,
        adaptive_block_size=args.adaptive_block_size,
        adaptive_c=args.adaptive_c,
        morph_open_kernel=args.morph_open_kernel,
        morph_close_kernel=args.morph_close_kernel,
        min_area=args.min_area,
        max_area=args.max_area,
        min_circularity=args.min_circularity,
        use_plate_mask=args.use_plate_mask,
        plate_margin_ratio=args.plate_margin_ratio,
        use_peak_count_estimation=args.use_peak_count_estimation,
        large_component_min_area=args.large_component_min_area,
        peak_blur_kernel=args.peak_blur_kernel,
        peak_local_max_kernel=args.peak_local_max_kernel,
        peak_relative_threshold=args.peak_relative_threshold,
        peak_min_distance=args.peak_min_distance,
        use_area_count_estimation=not args.disable_area_count_estimation,
        area_count_scale=args.area_count_scale,
    )


if __name__ == "__main__":
    main()
