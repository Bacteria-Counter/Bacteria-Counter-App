from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.detect_count import evaluate_count_predictions
from src.yolo.count_eval import (
    build_count_predictions,
    load_detection_split,
    run_yolo_detection_inference,
    save_count_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--model-path", required=False, type=Path)
    parser.add_argument("--detections-csv", required=False, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confidence-threshold", default=0.25, type=float)
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
    predictions = build_count_predictions(split_df, detections_df, confidence_threshold=args.confidence_threshold)
    evaluation = evaluate_count_predictions(predictions)
    metrics = dict(evaluation.metrics)
    metrics["confidence_threshold"] = float(args.confidence_threshold)
    metrics["image_size"] = int(args.image_size)
    metrics["max_det"] = int(args.max_det)
    metrics["min_confidence"] = float(args.min_confidence)
    save_count_evaluation(args.output_dir, detections_df, predictions, metrics)
    with (args.output_dir / "evaluation_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print("AGAR YOLO counting evaluation summary")
    print(f"image_count: {metrics['image_count']}")
    print(f"mae: {metrics['mae']}")
    print(f"rmse: {metrics['rmse']}")
    print(f"bias: {metrics['bias']}")
    print(f"exact_match_accuracy: {metrics['exact_match_accuracy']}")
    print(f"within_5_accuracy: {metrics['within_5_accuracy']}")
    print(f"Saved predictions and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
