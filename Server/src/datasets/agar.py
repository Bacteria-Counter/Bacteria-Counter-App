from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


CATEGORIES = ("countable", "empty", "uncountable")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
INVENTORY_COLUMNS = [
    "sample_id",
    "category",
    "image_path",
    "json_path",
    "image_name",
    "json_name",
    "background",
    "classes",
    "primary_class",
    "class_count",
    "colonies_number",
    "label_count",
    "image_width",
    "image_height",
    "image_channels",
    "image_readable",
    "has_pair",
    "detection_eligible",
    "species_image_eligible",
]


@dataclass(frozen=True)
class AgarRecord:
    sample_id: int | None
    category: str
    image_path: str
    json_path: str
    image_name: str
    json_name: str
    background: str
    classes: str
    primary_class: str
    class_count: int
    colonies_number: int | None
    label_count: int
    image_width: int | None
    image_height: int | None
    image_channels: int | None
    image_readable: bool
    has_pair: bool
    detection_eligible: bool
    species_image_eligible: bool


def load_agar_annotation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", [])
    classes = data.get("classes", [])
    if labels is None:
        data["labels"] = []
    if classes is None:
        data["classes"] = []
    return data


def discover_records(dataset_dir: Path, read_images: bool = True) -> pd.DataFrame:
    rows = []
    data_dir = find_agar_data_dir(dataset_dir)
    if data_dir is None:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    for category in CATEGORIES:
        category_dir = data_dir / category
        if not category_dir.exists():
            continue
        image_paths = sorted(
            path for path in category_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            json_path = image_path.with_suffix(".json")
            rows.append(record_from_pair(dataset_dir, category, image_path, json_path, read_images))
    if not rows:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)
    return pd.DataFrame([record.__dict__ for record in rows], columns=INVENTORY_COLUMNS)


def find_agar_data_dir(dataset_dir: Path) -> Path | None:
    direct_candidates = [dataset_dir / "data", dataset_dir]
    for candidate in direct_candidates:
        if any((candidate / category).is_dir() for category in CATEGORIES):
            return candidate

    for path in dataset_dir.rglob("*"):
        if path.is_dir() and path.name in CATEGORIES:
            parent = path.parent
            if any((parent / category).is_dir() for category in CATEGORIES):
                return parent
    return None


def record_from_pair(
    dataset_dir: Path,
    category: str,
    image_path: Path,
    json_path: Path,
    read_images: bool,
) -> AgarRecord:
    has_pair = json_path.exists()
    data = load_agar_annotation(json_path) if has_pair else {}
    classes = [str(value) for value in data.get("classes", [])]
    labels = data.get("labels", [])
    colonies_number = data.get("colonies_number")
    width, height, channels, readable = image_metadata(image_path) if read_images else (None, None, None, False)
    primary_class = classes[0] if classes else ""
    detection_eligible = category in {"countable", "empty"} and colonies_number is not None and int(colonies_number) >= 0
    species_image_eligible = bool(primary_class) and category in {"countable", "uncountable"}
    return AgarRecord(
        sample_id=data.get("sample_id"),
        category=category,
        image_path=str(image_path.relative_to(dataset_dir)),
        json_path=str(json_path.relative_to(dataset_dir)) if has_pair else "",
        image_name=image_path.name,
        json_name=json_path.name,
        background=str(data.get("background", "")),
        classes="|".join(classes),
        primary_class=primary_class,
        class_count=len(classes),
        colonies_number=None if colonies_number is None else int(colonies_number),
        label_count=len(labels),
        image_width=width,
        image_height=height,
        image_channels=channels,
        image_readable=readable,
        has_pair=has_pair,
        detection_eligible=detection_eligible,
        species_image_eligible=species_image_eligible,
    )


def image_metadata(path: Path) -> tuple[int | None, int | None, int | None, bool]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None, None, False
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return int(width), int(height), int(channels), True


def build_inventory(dataset_dir: Path, read_images: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    inventory = discover_records(dataset_dir, read_images=read_images)
    summary = summarize_inventory(inventory)
    return inventory, summary


def summarize_inventory(inventory: pd.DataFrame) -> dict[str, Any]:
    if inventory.empty:
        return {"total_images": 0}
    return {
        "total_images": int(len(inventory)),
        "categories": to_int_dict(inventory["category"].value_counts()),
        "backgrounds": to_int_dict(inventory["background"].value_counts()),
        "classes": to_int_dict(inventory["primary_class"].replace("", pd.NA).dropna().value_counts()),
        "paired_json": int(inventory["has_pair"].sum()),
        "readable_images": int(inventory["image_readable"].sum()),
        "detection_eligible": int(inventory["detection_eligible"].sum()),
        "species_image_eligible": int(inventory["species_image_eligible"].sum()),
        "colonies_number": numeric_summary(inventory["colonies_number"].dropna()),
        "label_count": numeric_summary(inventory["label_count"]),
    }


def numeric_summary(values: pd.Series) -> dict[str, float | int]:
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def to_int_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.to_dict().items()}


def split_inventory(
    inventory: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_columns: tuple[str, ...] = ("category", "primary_class"),
) -> pd.DataFrame:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
    if inventory.empty:
        return pd.DataFrame(columns=[*INVENTORY_COLUMNS, "split"])

    rng = random.Random(seed)
    rows = []
    working = inventory.copy()
    working["_stratum"] = working[list(stratify_columns)].fillna("").astype(str).agg("|".join, axis=1)

    for _, group in working.groupby("_stratum", sort=True):
        records = group.drop(columns=["_stratum"]).to_dict("records")
        rng.shuffle(records)
        n_total = len(records)
        n_train = round(n_total * train_ratio)
        n_val = round(n_total * val_ratio)

        if n_total >= 3:
            n_train = max(1, min(n_train, n_total - 2))
            n_val = max(1, min(n_val, n_total - n_train - 1))
        elif n_total == 2:
            n_train = 1
            n_val = 0
        else:
            n_train = 1
            n_val = 0

        for index, record in enumerate(records):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"
            record["split"] = split
            rows.append(record)

    return pd.DataFrame(rows).sort_values(["split", "category", "primary_class", "image_name"]).reset_index(drop=True)


def write_split_files(split_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_dir / "all.csv", index=False)
    for split in ("train", "val", "test"):
        split_df[split_df["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)

    task_filters = {
        "detect_count": split_df["detection_eligible"],
        "species_image": split_df["species_image_eligible"],
        "countable_only": split_df["category"] == "countable",
    }
    for task_name, mask in task_filters.items():
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        task_df = split_df[mask].copy()
        task_df.to_csv(task_dir / "all.csv", index=False)
        for split in ("train", "val", "test"):
            task_df[task_df["split"] == split].to_csv(task_dir / f"{split}.csv", index=False)
