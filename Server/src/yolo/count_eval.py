from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.classical.detect_count import CountPrediction, evaluate_count_predictions


DETECTION_COLUMNS = [
    "image_path",
    "category",
    "primary_class",
    "true_count",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


def load_detection_split(split_csv: Path, max_images: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(split_csv)
    if "detection_eligible" in frame.columns:
        frame = frame[frame["detection_eligible"].map(is_truthy)].copy()
    if max_images > 0:
        frame = frame.head(max_images).copy()
    if frame.empty:
        raise ValueError(f"No detection/counting rows found in {split_csv}")
    return frame.reset_index(drop=True)


def run_yolo_detection_inference(
    dataset_dir: Path,
    split_csv: Path,
    model_path: Path,
    image_size: int = 1536,
    device: str = "",
    min_confidence: float = 0.001,
    max_det: int = 1000,
    max_images: int = 0,
) -> pd.DataFrame:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is required for YOLO inference. Install ultralytics in Colab before running this script.") from exc

    split_df = load_detection_split(split_csv, max_images=max_images)
    model = YOLO(str(model_path))
    rows = []
    predict_kwargs = {
        "imgsz": int(image_size),
        "conf": float(min_confidence),
        "max_det": int(max_det),
        "verbose": False,
        "save": False,
    }
    if str(device).strip():
        predict_kwargs["device"] = str(device).strip()
    for row in split_df.to_dict("records"):
        image_relative = str(row["image_path"])
        image_path = dataset_dir / image_relative
        results = model.predict(source=str(image_path), **predict_kwargs)
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        for index in range(len(confidences)):
            box = xyxy[index]
            rows.append(
                {
                    "image_path": image_relative,
                    "category": str(row.get("category", "")),
                    "primary_class": str(row.get("primary_class", "")),
                    "true_count": int(row["colonies_number"]),
                    "class_id": int(class_ids[index]),
                    "confidence": float(confidences[index]),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                }
            )
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def build_count_predictions(
    split_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    confidence_threshold: float,
) -> list[CountPrediction]:
    if detections_df.empty:
        counts = {}
    else:
        filtered = detections_df[detections_df["confidence"] >= float(confidence_threshold)].copy()
        counts = filtered.groupby("image_path").size().astype(int).to_dict()
    predictions = []
    for row in split_df.to_dict("records"):
        image_relative = str(row["image_path"])
        true_count = int(row["colonies_number"])
        predicted_count = int(counts.get(image_relative, 0))
        predictions.append(
            CountPrediction(
                image_path=image_relative,
                true_count=true_count,
                predicted_count=predicted_count,
                category=str(row.get("category", "")),
                primary_class=str(row.get("primary_class", "")),
                error=int(predicted_count - true_count),
                abs_error=int(abs(predicted_count - true_count)),
                squared_error=int((predicted_count - true_count) ** 2),
            )
        )
    return predictions


def build_threshold_sweep(
    split_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    thresholds: list[float],
) -> tuple[pd.DataFrame, dict, list[CountPrediction]]:
    records = []
    best_metrics = None
    best_predictions: list[CountPrediction] = []
    for threshold in thresholds:
        predictions = build_count_predictions(split_df, detections_df, confidence_threshold=threshold)
        evaluation = evaluate_count_predictions(predictions)
        metrics = dict(evaluation.metrics)
        metrics["confidence_threshold"] = float(threshold)
        records.append(metrics)
        if best_metrics is None or rank_metrics(metrics) < rank_metrics(best_metrics):
            best_metrics = metrics
            best_predictions = predictions
    if best_metrics is None:
        raise ValueError("No thresholds were provided for the YOLO count sweep.")
    sweep_df = pd.DataFrame(records).sort_values(
        ["mae", "rmse", "bias", "confidence_threshold"],
        ascending=[True, True, True, True],
    )
    return sweep_df.reset_index(drop=True), best_metrics, best_predictions


def rank_metrics(metrics: dict) -> tuple[float, float, float, float]:
    return (
        float(metrics.get("mae", float("inf"))),
        float(metrics.get("rmse", float("inf"))),
        -float(metrics.get("within_5_accuracy", 0.0)),
        abs(float(metrics.get("bias", float("inf")))),
    )


def save_count_evaluation(
    output_dir: Path,
    detections_df: pd.DataFrame,
    predictions: list[CountPrediction],
    metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_df.to_csv(output_dir / "raw_detections.csv", index=False)
    pd.DataFrame([asdict(prediction) for prediction in predictions]).to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_threshold_sweep(
    output_dir: Path,
    detections_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    best_metrics: dict,
    best_predictions: list[CountPrediction],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_df.to_csv(output_dir / "raw_detections.csv", index=False)
    sweep_df.to_csv(output_dir / "sweep_results.csv", index=False)
    pd.DataFrame([asdict(prediction) for prediction in best_predictions]).to_csv(output_dir / "best_predictions.csv", index=False)
    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}
