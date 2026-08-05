from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd
import yaml

from src.classical.detect_count import read_image
from src.datasets.agar import load_agar_annotation


YOLO_TASKS = ("count", "species")
IMAGE_COPY_MODES = ("copy", "symlink")
RETRYABLE_IO_EXCEPTIONS = (ConnectionAbortedError, OSError)


def export_agar_to_yolo(
    dataset_dir: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    output_dir: Path,
    task: str,
    image_copy_mode: str = "copy",
    max_train_images: int = 0,
    max_val_images: int = 0,
    max_test_images: int = 0,
) -> dict:
    active_task = normalize_task(task)
    active_copy_mode = normalize_copy_mode(image_copy_mode)
    split_paths = {"train": train_csv, "val": val_csv, "test": test_csv}
    split_limits = {"train": max_train_images, "val": max_val_images, "test": max_test_images}
    class_names = resolve_class_names(dataset_dir, split_paths, active_task, split_limits)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": active_task,
        "image_copy_mode": active_copy_mode,
        "class_names": class_names,
        "allowed_label_classes": class_names if active_task == "species" else [],
        "splits": {},
    }
    for split_name, split_csv in split_paths.items():
        split_df = load_export_split(split_csv, active_task, max_images=split_limits[split_name])
        split_summary = export_split(
            dataset_dir=dataset_dir,
            split_df=split_df,
            output_dir=output_dir,
            split_name=split_name,
            task=active_task,
            class_to_index=class_to_index,
            image_copy_mode=active_copy_mode,
        )
        summary["splits"][split_name] = split_summary
    write_dataset_metadata(output_dir, class_names)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def normalize_task(task: str) -> str:
    active_task = str(task).strip().lower()
    if active_task not in YOLO_TASKS:
        raise ValueError(f"Unsupported YOLO task: {task}")
    return active_task


def normalize_copy_mode(image_copy_mode: str) -> str:
    active_copy_mode = str(image_copy_mode).strip().lower()
    if active_copy_mode not in IMAGE_COPY_MODES:
        raise ValueError(f"Unsupported image copy mode: {image_copy_mode}")
    return active_copy_mode


def resolve_class_names(
    dataset_dir: Path,
    split_paths: dict[str, Path],
    task: str,
    split_limits: dict[str, int],
) -> list[str]:
    if task == "count":
        return ["colony"]
    class_names = set()
    for split_name, split_csv in split_paths.items():
        split_df = load_export_split(split_csv, task, max_images=split_limits[split_name])
        for row in split_df.to_dict("records"):
            primary_class = str(row.get("primary_class", "")).strip()
            if primary_class:
                class_names.add(primary_class)
    if not class_names:
        raise ValueError("No species classes were found for YOLO export.")
    return sorted(class_names)


def load_export_split(split_csv: Path, task: str, max_images: int = 0) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv)
    if task == "count":
        if "detection_eligible" in split_df.columns:
            split_df = split_df[split_df["detection_eligible"].map(is_truthy)].copy()
    else:
        split_df = split_df[split_df["category"].astype(str) == "countable"].copy()
        if "label_count" in split_df.columns:
            split_df = split_df[split_df["label_count"].fillna(0).astype(int) > 0].copy()
    if max_images > 0:
        split_df = split_df.head(max_images).copy()
    if split_df.empty:
        raise ValueError(f"No rows available for YOLO {task} export in {split_csv}")
    return split_df.reset_index(drop=True)


def export_split(
    dataset_dir: Path,
    split_df: pd.DataFrame,
    output_dir: Path,
    split_name: str,
    task: str,
    class_to_index: dict[str, int],
    image_copy_mode: str,
) -> dict:
    image_dir = output_dir / "images" / split_name
    label_dir = output_dir / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    object_count = 0
    negative_image_count = 0
    invalid_box_count = 0
    missing_annotation_count = 0
    skipped_non_target_count = 0
    for row in split_df.to_dict("records"):
        image_relative = str(row["image_path"])
        image_source = dataset_dir / image_relative
        image_target = image_dir / make_export_image_name(image_relative)
        transfer_image(image_source, image_target, image_copy_mode)
        label_target = label_dir / f"{image_target.stem}.txt"
        image_width, image_height = resolve_image_size(row, image_source)
        label_lines, line_summary = build_label_lines(
            dataset_dir=dataset_dir,
            row=row,
            task=task,
            class_to_index=class_to_index,
            image_width=image_width,
            image_height=image_height,
        )
        label_target.write_text("\n".join(label_lines), encoding="utf-8")
        image_count += 1
        object_count += line_summary["object_count"]
        invalid_box_count += line_summary["invalid_box_count"]
        missing_annotation_count += line_summary["missing_annotation_count"]
        skipped_non_target_count += line_summary["skipped_non_target_count"]
        if not label_lines:
            negative_image_count += 1
    return {
        "image_count": image_count,
        "object_count": object_count,
        "negative_image_count": negative_image_count,
        "invalid_box_count": invalid_box_count,
        "missing_annotation_count": missing_annotation_count,
        "skipped_non_target_count": skipped_non_target_count,
    }


def resolve_image_size(row: dict, image_path: Path) -> tuple[int, int]:
    width = int(row.get("image_width") or 0)
    height = int(row.get("image_height") or 0)
    if width > 0 and height > 0:
        return width, height
    image = run_with_io_retry(lambda: read_image(image_path), image_path)
    return int(image.shape[1]), int(image.shape[0])


def build_label_lines(
    dataset_dir: Path,
    row: dict,
    task: str,
    class_to_index: dict[str, int],
    image_width: int,
    image_height: int,
) -> tuple[list[str], dict[str, int]]:
    summary = {"object_count": 0, "invalid_box_count": 0, "missing_annotation_count": 0, "skipped_non_target_count": 0}
    labels = load_row_labels(dataset_dir, row)
    if not labels and str(row.get("json_path", "")).strip():
        summary["missing_annotation_count"] = 0
    if not labels and task == "species":
        return [], summary
    if not labels and task == "count":
        return [], summary
    lines = []
    for label in labels:
        class_name = resolve_label_class_name(label, row, task)
        if not class_name:
            summary["invalid_box_count"] += 1
            continue
        if class_name not in class_to_index:
            if task == "species":
                summary["skipped_non_target_count"] += 1
            else:
                summary["invalid_box_count"] += 1
            continue
        box_values = normalize_yolo_box(label, image_width, image_height)
        if box_values is None:
            summary["invalid_box_count"] += 1
            continue
        class_id = class_to_index[class_name]
        lines.append(format_yolo_label_line(class_id, box_values))
        summary["object_count"] += 1
    return lines, summary


def resolve_label_class_name(label: dict, row: dict, task: str) -> str:
    if task == "count":
        return "colony"
    label_class = str(label.get("class", "")).strip()
    if label_class:
        return label_class
    return str(row.get("primary_class", "")).strip()


def normalize_yolo_box(label: dict, image_width: int, image_height: int) -> tuple[float, float, float, float] | None:
    try:
        x = float(label["x"])
        y = float(label["y"])
        width = float(label["width"])
        height = float(label["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if image_width <= 0 or image_height <= 0:
        return None
    if width <= 0 or height <= 0:
        return None
    x2 = x + width
    y2 = y + height
    if x < 0 or y < 0 or x2 > float(image_width) or y2 > float(image_height):
        return None
    center_x = (x + width * 0.5) / float(image_width)
    center_y = (y + height * 0.5) / float(image_height)
    norm_width = width / float(image_width)
    norm_height = height / float(image_height)
    if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
        return None
    if not (0.0 < norm_width <= 1.0 and 0.0 < norm_height <= 1.0):
        return None
    return center_x, center_y, norm_width, norm_height


def format_yolo_label_line(class_id: int, box_values: tuple[float, float, float, float]) -> str:
    center_x, center_y, width, height = box_values
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def load_row_labels(dataset_dir: Path, row: dict) -> list[dict]:
    json_relative = str(row.get("json_path", "")).strip()
    if not json_relative:
        return []
    json_path = dataset_dir / json_relative
    exists = run_with_io_retry(lambda: json_path.exists(), json_path)
    if not exists:
        return []
    payload = run_with_io_retry(lambda: load_agar_annotation(json_path), json_path)
    labels = payload.get("labels", [])
    return labels if isinstance(labels, list) else []


def extract_label_classes(dataset_dir: Path, row: dict) -> list[str]:
    labels = load_row_labels(dataset_dir, row)
    classes = []
    for label in labels:
        label_class = str(label.get("class", "")).strip()
        if label_class:
            classes.append(label_class)
    return classes


def make_export_image_name(image_relative: str) -> str:
    source = Path(image_relative)
    stem = "__".join(source.with_suffix("").parts)
    return f"{stem}{source.suffix.lower()}"


def transfer_image(source: Path, target: Path, image_copy_mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if image_copy_mode == "copy":
        run_with_io_retry(lambda: shutil.copy2(source, target), source)
        return
    target.symlink_to(source.resolve())


def write_dataset_metadata(output_dir: Path, class_names: list[str]) -> None:
    names = {index: name for index, name in enumerate(class_names)}
    yaml_payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_payload, f, sort_keys=False)
    (output_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def run_with_io_retry(action, path: Path, attempts: int = 4, delay_seconds: float = 1.0):
    last_error = None
    for attempt in range(attempts):
        try:
            return action()
        except RETRYABLE_IO_EXCEPTIONS as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay_seconds * (attempt + 1))
    raise RuntimeError(f"Repeated I/O failure while accessing {path}: {last_error}") from last_error
