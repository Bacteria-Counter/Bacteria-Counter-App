from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd

from src.classical.detect_count import (
    ClassicalCountConfig,
    CountEvaluation,
    CountPrediction,
    count_colonies,
    evaluate_count_predictions,
    read_image,
    render_count_overlay,
)


def run_agar_count_evaluation(
    dataset_dir: Path,
    split_csv: Path,
    config: ClassicalCountConfig,
    output_dir: Path | None = None,
    max_images: int = 0,
    save_overlays: bool = False,
    overlay_limit: int = 40,
) -> CountEvaluation:
    split_df = load_detection_split(split_csv, max_images=max_images)
    if split_df.empty:
        raise ValueError(f"No detection/counting rows found in {split_csv}")

    overlay_dir = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_overlays:
            overlay_dir = output_dir / "overlays"
            overlay_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    for index, row in split_df.reset_index(drop=True).iterrows():
        image_relative = str(row["image_path"])
        image_path = dataset_dir / image_relative
        image = read_image(image_path)
        predicted_count, mask, contours = count_colonies(image, config)
        true_count = int(row["colonies_number"])
        prediction = CountPrediction(
            image_path=image_relative,
            true_count=true_count,
            predicted_count=int(predicted_count),
            category=str(row.get("category", "")),
            primary_class=str(row.get("primary_class", "")),
            error=int(predicted_count - true_count),
            abs_error=int(abs(predicted_count - true_count)),
            squared_error=int((predicted_count - true_count) ** 2),
        )
        predictions.append(prediction)
        if overlay_dir is not None and index < overlay_limit:
            overlay = render_count_overlay(image, predicted_count, contours)
            cv2.imwrite(str(overlay_dir / safe_output_name(image_relative, ".jpg")), overlay)
            cv2.imwrite(str(overlay_dir / safe_output_name(image_relative, "_mask.png")), mask)

    evaluation = evaluate_count_predictions(predictions, config)
    if output_dir is not None:
        save_evaluation(evaluation, output_dir)
    return evaluation


def load_detection_split(split_csv: Path, max_images: int = 0) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv)
    if "detection_eligible" in split_df.columns:
        split_df = split_df[split_df["detection_eligible"].map(is_truthy)].copy()
    if max_images > 0:
        split_df = split_df.head(max_images).copy()
    return split_df


def save_evaluation(evaluation: CountEvaluation, output_dir: Path) -> None:
    prediction_rows = [asdict(prediction) for prediction in evaluation.predictions]
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation.metrics, f, indent=2)


def load_count_config(path: Path) -> ClassicalCountConfig:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "config" in data and isinstance(data["config"], dict):
        data = data["config"]
    return ClassicalCountConfig(**data)


def save_count_config(config: ClassicalCountConfig, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_output_name(path: str, suffix: str) -> str:
    source = Path(path)
    stem = "__".join(source.with_suffix("").parts)
    return f"{stem}{suffix}"
