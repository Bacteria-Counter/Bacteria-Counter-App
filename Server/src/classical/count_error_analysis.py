from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


COUNT_BINS = [-1, 0, 5, 20, 50, 100, 200, float("inf")]
COUNT_LABELS = ["zero", "1-5", "6-20", "21-50", "51-100", "101-200", "201+"]


def analyze_count_predictions(predictions_csv: Path, worst_n: int = 30) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions = pd.read_csv(predictions_csv)
    required_columns = {"true_count", "predicted_count", "category", "primary_class", "image_path"}
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    frame = normalize_prediction_frame(predictions)
    summary = {
        "overall": metric_summary(frame),
        "by_category": grouped_summary(frame, "category"),
        "by_primary_class": grouped_summary(frame[frame["primary_class"] != ""], "primary_class"),
        "by_true_count_bin": grouped_summary(frame, "true_count_bin"),
        "worst_n": int(worst_n),
    }
    worst = frame.sort_values(["abs_error", "true_count"], ascending=[False, False]).head(worst_n)
    return summary, worst


def normalize_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["true_count"] = frame["true_count"].astype(int)
    frame["predicted_count"] = frame["predicted_count"].astype(int)
    frame["error"] = frame["predicted_count"] - frame["true_count"]
    frame["abs_error"] = frame["error"].abs()
    frame["squared_error"] = frame["error"] * frame["error"]
    frame["primary_class"] = frame["primary_class"].fillna("").astype(str)
    frame["true_count_bin"] = pd.cut(frame["true_count"], bins=COUNT_BINS, labels=COUNT_LABELS)
    return frame


def metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"image_count": 0}
    zero = frame["true_count"] == 0
    nonzero = frame["true_count"] > 0
    summary = {
        "image_count": int(len(frame)),
        "true_total": int(frame["true_count"].sum()),
        "predicted_total": int(frame["predicted_count"].sum()),
        "mae": float(frame["abs_error"].mean()),
        "rmse": float((frame["squared_error"].mean()) ** 0.5),
        "bias": float(frame["error"].mean()),
        "median_abs_error": float(frame["abs_error"].median()),
        "within_5_accuracy": float((frame["abs_error"] <= 5).mean()),
        "within_10_accuracy": float((frame["abs_error"] <= 10).mean()),
        "exact_match_accuracy": float((frame["abs_error"] == 0).mean()),
        "zero_image_count": int(zero.sum()),
        "nonzero_image_count": int(nonzero.sum()),
    }
    if zero.any():
        summary["zero_exact_accuracy"] = float((frame.loc[zero, "predicted_count"] == 0).mean())
        summary["empty_false_positive_mean"] = float(frame.loc[zero, "predicted_count"].mean())
    if nonzero.any():
        summary["nonzero_mape_percent"] = float(
            (frame.loc[nonzero, "abs_error"] / frame.loc[nonzero, "true_count"]).mean() * 100.0
        )
        summary["nonzero_mean_true_count"] = float(frame.loc[nonzero, "true_count"].mean())
        summary["nonzero_mean_predicted_count"] = float(frame.loc[nonzero, "predicted_count"].mean())
    return summary


def grouped_summary(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    result = {}
    for value, group in frame.groupby(column, observed=True):
        result[str(value)] = metric_summary(group)
    return result


def save_count_error_analysis(summary: dict[str, Any], worst: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    worst.to_csv(output_dir / "worst_predictions.csv", index=False)
