from __future__ import annotations

import numpy as np

from scripts.unet.sweep_unet_count_postprocess_agar import (
    build_grid,
    evaluate_cached_rows,
    parse_float_list,
    parse_int_list,
    ranking_key,
)


class Args:
    thresholds = "0.3,0.5"
    min_component_areas = "4,12"


def test_parse_grid_lists() -> None:
    assert parse_float_list("0.25,0.5") == [0.25, 0.5]
    assert parse_int_list("4,8,16") == [4, 8, 16]


def test_build_grid_produces_threshold_area_pairs() -> None:
    assert build_grid(Args()) == [(0.3, 4), (0.3, 12), (0.5, 4), (0.5, 12)]


def test_evaluate_cached_rows_returns_count_and_segmentation_metrics() -> None:
    cached_rows = [
        {
            "image_path": "data/countable/a.png",
            "true_count": 1,
            "category": "countable",
            "primary_class": "A",
            "prob_mask": np.zeros((8, 8), dtype=float),
            "target_mask": np.zeros((8, 8), dtype="uint8"),
        }
    ]
    cached_rows[0]["prob_mask"][2:6, 2:6] = 0.9
    cached_rows[0]["target_mask"][2:6, 2:6] = 1

    result = evaluate_cached_rows(cached_rows, threshold=0.5, min_component_area=4)

    assert result["metrics"]["image_count"] == 1
    assert result["metrics"]["mae"] == 0.0
    assert result["metrics"]["segmentation_dice_mean"] > 0.99
    assert result["predictions"][0].predicted_count == 1


def test_ranking_key_prefers_lower_mae_then_rmse_then_abs_bias() -> None:
    better = {"mae": 10.0, "rmse": 12.0, "bias": -5.0}
    worse = {"mae": 11.0, "rmse": 9.0, "bias": 1.0}

    assert ranking_key(better) < ranking_key(worse)
