from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from scripts.yolo.train_yolo_agar import resolve_resume_checkpoint
from src.yolo.count_eval import build_count_predictions, build_threshold_sweep
from src.yolo.export import export_agar_to_yolo, load_row_labels, run_with_io_retry


def write_sample(
    root: Path,
    category: str,
    sample_id: int,
    class_name: str,
    colonies_number: int,
    labels: list[dict],
) -> tuple[Path, Path]:
    category_dir = root / "data" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((40, 50, 3), 160, dtype=np.uint8)
    image_path = category_dir / f"{sample_id}.png"
    json_path = category_dir / f"{sample_id}.json"
    cv2.imwrite(str(image_path), image)
    payload = {
        "background": "bright",
        "classes": [class_name],
        "colonies_number": colonies_number,
        "labels": labels,
        "sample_id": sample_id,
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return image_path, json_path


def write_split(root: Path, rows: list[dict], name: str) -> Path:
    path = root / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_export_agar_to_yolo_count_writes_labels_and_empty_negatives(tmp_path: Path) -> None:
    count_image, count_json = write_sample(
        tmp_path,
        "countable",
        1,
        "A",
        1,
        [{"class": "A", "x": 10, "y": 12, "width": 8, "height": 10}],
    )
    empty_image, empty_json = write_sample(tmp_path, "empty", 2, "A", 0, [])
    split_rows = [
        {
            "image_path": str(count_image.relative_to(tmp_path)),
            "json_path": str(count_json.relative_to(tmp_path)),
            "category": "countable",
            "primary_class": "A",
            "colonies_number": 1,
            "label_count": 1,
            "detection_eligible": True,
            "image_width": 50,
            "image_height": 40,
        },
        {
            "image_path": str(empty_image.relative_to(tmp_path)),
            "json_path": str(empty_json.relative_to(tmp_path)),
            "category": "empty",
            "primary_class": "A",
            "colonies_number": 0,
            "label_count": 0,
            "detection_eligible": True,
            "image_width": 50,
            "image_height": 40,
        },
    ]
    train_csv = write_split(tmp_path, split_rows, "train")
    val_csv = write_split(tmp_path, split_rows[:1], "val")
    test_csv = write_split(tmp_path, split_rows[1:], "test")

    summary = export_agar_to_yolo(
        dataset_dir=tmp_path,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_dir=tmp_path / "yolo_count",
        task="count",
    )

    assert summary["class_names"] == ["colony"]
    train_label = tmp_path / "yolo_count" / "labels" / "train" / "data__countable__1.txt"
    empty_label = tmp_path / "yolo_count" / "labels" / "train" / "data__empty__2.txt"
    assert train_label.read_text(encoding="utf-8").startswith("0 ")
    assert empty_label.read_text(encoding="utf-8") == ""
    assert (tmp_path / "yolo_count" / "data.yaml").exists()


def test_export_agar_to_yolo_species_uses_multiclass_labels(tmp_path: Path) -> None:
    image_a, json_a = write_sample(
        tmp_path,
        "countable",
        1,
        "B",
        2,
        [{"class": "B", "x": 5, "y": 8, "width": 10, "height": 12}],
    )
    image_b, json_b = write_sample(
        tmp_path,
        "countable",
        2,
        "A",
        1,
        [{"class": "A", "x": 11, "y": 9, "width": 8, "height": 8}],
    )
    rows = [
        {
            "image_path": str(image_a.relative_to(tmp_path)),
            "json_path": str(json_a.relative_to(tmp_path)),
            "category": "countable",
            "primary_class": "B",
            "colonies_number": 2,
            "label_count": 1,
            "image_width": 50,
            "image_height": 40,
        },
        {
            "image_path": str(image_b.relative_to(tmp_path)),
            "json_path": str(json_b.relative_to(tmp_path)),
            "category": "countable",
            "primary_class": "A",
            "colonies_number": 1,
            "label_count": 1,
            "image_width": 50,
            "image_height": 40,
        },
    ]
    train_csv = write_split(tmp_path, rows, "train")
    val_csv = write_split(tmp_path, rows[:1], "val")
    test_csv = write_split(tmp_path, rows[1:], "test")

    summary = export_agar_to_yolo(
        dataset_dir=tmp_path,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        output_dir=tmp_path / "yolo_species",
        task="species",
    )

    assert summary["class_names"] == ["A", "B"]
    label_lines = (tmp_path / "yolo_species" / "labels" / "train" / "data__countable__1.txt").read_text(encoding="utf-8")
    assert label_lines.startswith("1 ")
    assert summary["class_names"] == ["A", "B"]


def test_export_agar_to_yolo_species_skips_non_target_box_labels(tmp_path: Path) -> None:
    image_path, json_path = write_sample(
        tmp_path,
        "countable",
        1,
        "A",
        3,
        [
            {"class": "A", "x": 5, "y": 8, "width": 10, "height": 12},
            {"class": "Contamination", "x": 20, "y": 8, "width": 8, "height": 8},
            {"class": "Defect", "x": 32, "y": 10, "width": 7, "height": 7},
        ],
    )
    rows = [
        {
            "image_path": str(image_path.relative_to(tmp_path)),
            "json_path": str(json_path.relative_to(tmp_path)),
            "category": "countable",
            "primary_class": "A",
            "colonies_number": 3,
            "label_count": 3,
            "image_width": 50,
            "image_height": 40,
        }
    ]
    split_csv = write_split(tmp_path, rows, "train")

    summary = export_agar_to_yolo(
        dataset_dir=tmp_path,
        train_csv=split_csv,
        val_csv=split_csv,
        test_csv=split_csv,
        output_dir=tmp_path / "yolo_species_filtered",
        task="species",
    )

    label_lines = (tmp_path / "yolo_species_filtered" / "labels" / "train" / "data__countable__1.txt").read_text(encoding="utf-8").strip().splitlines()
    assert summary["class_names"] == ["A"]
    assert summary["splits"]["train"]["object_count"] == 1
    assert summary["splits"]["train"]["skipped_non_target_count"] == 2
    assert len(label_lines) == 1
    assert label_lines[0].startswith("0 ")


def test_export_agar_to_yolo_skips_invalid_boxes_and_records_them(tmp_path: Path) -> None:
    image_path, json_path = write_sample(
        tmp_path,
        "countable",
        1,
        "A",
        1,
        [{"class": "A", "x": -2, "y": 10, "width": 8, "height": 9}],
    )
    rows = [
        {
            "image_path": str(image_path.relative_to(tmp_path)),
            "json_path": str(json_path.relative_to(tmp_path)),
            "category": "countable",
            "primary_class": "A",
            "colonies_number": 1,
            "label_count": 1,
            "detection_eligible": True,
            "image_width": 50,
            "image_height": 40,
        }
    ]
    split_csv = write_split(tmp_path, rows, "train")

    summary = export_agar_to_yolo(
        dataset_dir=tmp_path,
        train_csv=split_csv,
        val_csv=split_csv,
        test_csv=split_csv,
        output_dir=tmp_path / "yolo_invalid",
        task="count",
    )

    assert summary["splits"]["train"]["invalid_box_count"] == 1
    assert (tmp_path / "yolo_invalid" / "labels" / "train" / "data__countable__1.txt").read_text(encoding="utf-8") == ""


def test_build_count_predictions_and_threshold_sweep_follow_confidence_threshold() -> None:
    split_df = pd.DataFrame(
        [
            {"image_path": "a.png", "colonies_number": 2, "category": "countable", "primary_class": "A"},
            {"image_path": "b.png", "colonies_number": 0, "category": "empty", "primary_class": ""},
        ]
    )
    detections_df = pd.DataFrame(
        [
            {"image_path": "a.png", "confidence": 0.9},
            {"image_path": "a.png", "confidence": 0.6},
            {"image_path": "a.png", "confidence": 0.2},
            {"image_path": "b.png", "confidence": 0.4},
        ]
    )

    predictions = build_count_predictions(split_df, detections_df, confidence_threshold=0.5)
    sweep_df, best_metrics, best_predictions = build_threshold_sweep(split_df, detections_df, [0.2, 0.5, 0.8])

    assert predictions[0].predicted_count == 2
    assert predictions[1].predicted_count == 0
    assert float(best_metrics["confidence_threshold"]) == 0.5
    assert best_predictions[0].predicted_count == 2
    assert float(sweep_df.iloc[0]["confidence_threshold"]) == 0.5
    assert set(float(value) for value in sweep_df["confidence_threshold"]) == {0.2, 0.5, 0.8}


def test_resolve_resume_checkpoint_uses_output_dir_last_pt_when_requested(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")

    resolved = resolve_resume_checkpoint(tmp_path / "run", resume=True, resume_from="")

    assert resolved == checkpoint


def test_resolve_resume_checkpoint_uses_explicit_path_when_provided(tmp_path: Path) -> None:
    checkpoint = tmp_path / "manual_last.pt"
    checkpoint.write_bytes(b"checkpoint")

    resolved = resolve_resume_checkpoint(tmp_path / "run", resume=False, resume_from=str(checkpoint))

    assert resolved == checkpoint


def test_run_with_io_retry_recovers_from_transient_abort(tmp_path: Path) -> None:
    state = {"calls": 0}

    def flaky():
        state["calls"] += 1
        if state["calls"] == 1:
            raise ConnectionAbortedError("temporary drive failure")
        return "ok"

    result = run_with_io_retry(flaky, tmp_path / "x.json", attempts=2, delay_seconds=0.0)

    assert result == "ok"
    assert state["calls"] == 2


def test_load_row_labels_retries_transient_exists_failure(tmp_path: Path, monkeypatch) -> None:
    image_path, json_path = write_sample(
        tmp_path,
        "countable",
        1000,
        "A",
        1,
        [{"class": "A", "x": 5, "y": 6, "width": 8, "height": 9}],
    )
    row = {
        "image_path": str(image_path.relative_to(tmp_path)),
        "json_path": str(json_path.relative_to(tmp_path)),
    }
    state = {"calls": 0}
    original_exists = Path.exists

    def flaky_exists(self):
        if self == json_path:
            state["calls"] += 1
            if state["calls"] == 1:
                raise ConnectionAbortedError("temporary drive failure")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", flaky_exists)
    labels = load_row_labels(tmp_path, row)

    assert len(labels) == 1
    assert labels[0]["class"] == "A"
