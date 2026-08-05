from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.classical.count_error_analysis import analyze_count_predictions, save_count_error_analysis


def test_analyze_count_predictions_groups_errors(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        [
            {
                "image_path": "a.jpg",
                "true_count": 0,
                "predicted_count": 3,
                "category": "empty",
                "primary_class": "",
            },
            {
                "image_path": "b.jpg",
                "true_count": 10,
                "predicted_count": 8,
                "category": "countable",
                "primary_class": "A",
            },
            {
                "image_path": "c.jpg",
                "true_count": 60,
                "predicted_count": 30,
                "category": "countable",
                "primary_class": "A",
            },
        ]
    )
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)

    summary, worst = analyze_count_predictions(path, worst_n=2)

    assert summary["overall"]["image_count"] == 3
    assert summary["overall"]["true_total"] == 70
    assert summary["overall"]["predicted_total"] == 41
    assert summary["by_category"]["empty"]["image_count"] == 1
    assert summary["by_primary_class"]["A"]["image_count"] == 2
    assert worst.iloc[0]["image_path"] == "c.jpg"


def test_save_count_error_analysis_writes_outputs(tmp_path: Path) -> None:
    summary = {"overall": {"image_count": 1}}
    worst = pd.DataFrame([{"image_path": "a.jpg", "abs_error": 4}])

    save_count_error_analysis(summary, worst, tmp_path)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "worst_predictions.csv").exists()
