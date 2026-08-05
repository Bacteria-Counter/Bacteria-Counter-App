from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.classical.species_classification import (
    SpeciesFeatureConfig,
    evaluate_species_classifier,
    extract_species_features,
    load_species_model,
    save_species_model,
    select_species_classifier,
    train_species_classifier,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 64, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_species_features_are_stable() -> None:
    image = np.full((64, 64, 3), (20, 80, 180), dtype=np.uint8)

    features = extract_species_features(image, SpeciesFeatureConfig(image_size=32))

    assert "feature_global_b_mean" in features
    assert "feature_global_hue_hist_00" in features
    assert "feature_center_edge_density" in features
    assert "feature_radial_gray_mean_0" in features
    assert all(np.isfinite(value) for value in features.values())


def test_nearest_centroid_species_classifier_learns_color_classes(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = []
    colors = {
        "A": (20, 20, 220),
        "B": (220, 20, 20),
    }
    for label, color in colors.items():
        for index in range(3):
            relative = f"data/countable/{label}_{index}.png"
            write_image(dataset_dir / relative, color)
            rows.append(
                {
                    "image_path": relative,
                    "primary_class": label,
                    "category": "countable",
                    "species_image_eligible": True,
                }
            )
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    pd.DataFrame(rows).to_csv(val_csv, index=False)

    model, train_features = train_species_classifier(
        dataset_dir=dataset_dir,
        train_csv=train_csv,
        config=SpeciesFeatureConfig(image_size=32),
    )
    evaluation = evaluate_species_classifier(dataset_dir, val_csv, model)

    assert set(model.classes) == {"A", "B"}
    assert not train_features.empty
    assert evaluation.metrics["accuracy"] == 1.0
    assert evaluation.metrics["macro_f1"] == 1.0


def test_species_classifier_selection_returns_best_candidate(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = []
    for label, color in {"A": (10, 20, 220), "B": (220, 20, 10)}.items():
        for index in range(4):
            relative = f"data/countable/{label}_{index}.png"
            write_image(dataset_dir / relative, color)
            rows.append(
                {
                    "image_path": relative,
                    "primary_class": label,
                    "category": "countable",
                    "species_image_eligible": True,
                }
            )
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    pd.DataFrame(rows).to_csv(val_csv, index=False)

    model, train_features, val_features, selection = select_species_classifier(
        dataset_dir,
        train_csv,
        val_csv,
        SpeciesFeatureConfig(image_size=32),
        classifier_types=("centroid_l2", "knn_3"),
    )

    assert model.classifier_type in {"centroid_l2", "knn_3"}
    assert not train_features.empty
    assert not val_features.empty
    assert list(selection.columns) == ["classifier_type", "accuracy", "macro_f1", "macro_precision", "macro_recall"]


def test_species_model_roundtrip(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    rows = []
    for label, color in {"A": (10, 10, 230), "B": (230, 10, 10)}.items():
        relative = f"data/countable/{label}.png"
        write_image(dataset_dir / relative, color)
        rows.append(
            {
                "image_path": relative,
                "primary_class": label,
                "category": "countable",
                "species_image_eligible": True,
            }
        )
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    model, _ = train_species_classifier(dataset_dir, train_csv, SpeciesFeatureConfig(image_size=32))
    model_path = tmp_path / "model.json"

    save_species_model(model, model_path)
    loaded = load_species_model(model_path)

    assert loaded.classes == model.classes
    assert loaded.feature_names == model.feature_names
    assert loaded.classifier_type == model.classifier_type
