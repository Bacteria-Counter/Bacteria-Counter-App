from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo.count_eval import (
    build_threshold_sweep,
    load_detection_split,
    run_yolo_detection_inference,
    save_threshold_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--model-path", required=False, type=Path)
    parser.add_argument("--detections-csv", required=False, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold-start", default=0.05, type=float)
    parser.add_argument("--threshold-stop", default=0.95, type=float)
    parser.add_argument("--threshold-step", default=0.05, type=float)
    parser.add_argument("--image-size", default=1536, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--min-confidence", default=0.001, type=float)
    parser.add_argument("--max-det", default=1000, type=int)
    parser.add_argument("--max-images", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_path is None and args.detections_csv is None:
        raise ValueError("Provide either --model-path or --detections-csv.")
    split_df = load_detection_split(args.split_csv, max_images=args.max_images)
    detections_df = (
        pd.read_csv(args.detections_csv)
        if args.detections_csv is not None
        else run_yolo_detection_inference(
            dataset_dir=args.dataset_dir,
            split_csv=args.split_csv,
            model_path=args.model_path,
            image_size=args.image_size,
            device=args.device,
            min_confidence=args.min_confidence,
            max_det=args.max_det,
            max_images=args.max_images,
        )
    )
    thresholds = build_threshold_values(args.threshold_start, args.threshold_stop, args.threshold_step)
    sweep_df, best_metrics, best_predictions = build_threshold_sweep(split_df, detections_df, thresholds)
    best_metrics["image_size"] = int(args.image_size)
    best_metrics["max_det"] = int(args.max_det)
    best_metrics["min_confidence"] = float(args.min_confidence)
    save_threshold_sweep(args.output_dir, detections_df, sweep_df, best_metrics, best_predictions)
    with (args.output_dir / "sweep_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print("AGAR YOLO counting threshold sweep summary")
    print(f"best_threshold: {best_metrics['confidence_threshold']}")
    print(f"mae: {best_metrics['mae']}")
    print(f"rmse: {best_metrics['rmse']}")
    print(f"bias: {best_metrics['bias']}")
    print(f"within_5_accuracy: {best_metrics['within_5_accuracy']}")
    print(f"Saved sweep outputs to: {args.output_dir}")


def build_threshold_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold-step must be positive.")
    if stop < start:
        raise ValueError("threshold-stop must be greater than or equal to threshold-start.")
    values = []
    current = float(start)
    while current <= float(stop) + 1e-9:
        values.append(round(current, 6))
        current += float(step)
    return values


if __name__ == "__main__":
    main()
