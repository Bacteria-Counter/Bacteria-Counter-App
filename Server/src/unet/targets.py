from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.classical.detect_count import read_image
from src.datasets.agar import load_agar_annotation


@dataclass(frozen=True)
class PseudoMaskConfig:
    ellipse_shrink_ratio: float = 0.9
    min_radius: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


def load_unet_count_split(split_csv: Path, max_images: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(split_csv)
    if "detection_eligible" in frame.columns:
        frame = frame[frame["detection_eligible"].map(is_truthy)].copy()
    if max_images > 0:
        frame = frame.head(max_images).copy()
    return frame.reset_index(drop=True)


def build_unet_count_manifest(
    dataset_dir: Path,
    split_csv: Path,
    output_dir: Path,
    config: PseudoMaskConfig | None = None,
    max_images: int = 0,
) -> pd.DataFrame:
    active_config = config or PseudoMaskConfig()
    split_df = load_unet_count_split(split_csv, max_images=max_images)
    if split_df.empty:
        raise ValueError(f"No detection/count rows found in {split_csv}")
    rows = []
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for row in split_df.to_dict("records"):
        image_relative = str(row["image_path"])
        image_path = dataset_dir / image_relative
        image = read_image(image_path)
        height, width = image.shape[:2]
        labels = load_labels(dataset_dir, row)
        mask = render_colony_pseudomask((height, width), labels, active_config)
        mask_name = make_mask_name(image_relative)
        mask_path = masks_dir / mask_name
        cv2.imwrite(str(mask_path), mask)
        rows.append(
            {
                "image_path": image_relative,
                "json_path": str(row.get("json_path", "")),
                "mask_path": str(mask_path.relative_to(output_dir)),
                "category": str(row.get("category", "")),
                "primary_class": str(row.get("primary_class", "")),
                "colonies_number": int(row["colonies_number"]),
                "label_count": int(row.get("label_count", 0)),
                "image_width": int(width),
                "image_height": int(height),
                "mask_kind": "empty_zero" if int(row["colonies_number"]) == 0 else "pseudo_box_ellipse",
            }
        )
    return pd.DataFrame(rows)


def write_unet_count_targets(
    dataset_dir: Path,
    split_csv: Path,
    output_dir: Path,
    config: PseudoMaskConfig | None = None,
    max_images: int = 0,
) -> dict:
    active_config = config or PseudoMaskConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_unet_count_manifest(
        dataset_dir=dataset_dir,
        split_csv=split_csv,
        output_dir=output_dir,
        config=active_config,
        max_images=max_images,
    )
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    summary = {
        "image_count": int(len(manifest)),
        "mask_kind_counts": to_int_dict(manifest["mask_kind"].value_counts()),
        "category_counts": to_int_dict(manifest["category"].value_counts()),
        "pseudo_mask_config": active_config.to_dict(),
        "mask_type": "pseudo_masks_from_box_annotations",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def render_colony_pseudomask(
    image_shape: tuple[int, int],
    labels: list[dict],
    config: PseudoMaskConfig | None = None,
) -> np.ndarray:
    active_config = config or PseudoMaskConfig()
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for label in labels:
        ellipse = normalize_label_ellipse(label, width, height, active_config)
        if ellipse is None:
            continue
        center, axes = ellipse
        cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 255, thickness=-1)
    return mask


def load_labels(dataset_dir: Path, row: dict) -> list[dict]:
    json_relative = str(row.get("json_path", "")).strip()
    if not json_relative:
        return []
    data = load_agar_annotation(dataset_dir / json_relative)
    labels = data.get("labels", [])
    return labels if isinstance(labels, list) else []


def normalize_label_ellipse(
    label: dict,
    image_width: int,
    image_height: int,
    config: PseudoMaskConfig,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    try:
        x = float(label["x"])
        y = float(label["y"])
        width = float(label["width"])
        height = float(label["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    center_x = int(round(np.clip(x + width * 0.5, 0.0, max(float(image_width - 1), 0.0))))
    center_y = int(round(np.clip(y + height * 0.5, 0.0, max(float(image_height - 1), 0.0))))
    radius_x = max(config.min_radius, int(round(width * 0.5 * config.ellipse_shrink_ratio)))
    radius_y = max(config.min_radius, int(round(height * 0.5 * config.ellipse_shrink_ratio)))
    return (center_x, center_y), (radius_x, radius_y)


def make_mask_name(image_relative: str) -> str:
    source = Path(image_relative)
    stem = "__".join(source.with_suffix("").parts)
    return f"{stem}_mask.png"


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def to_int_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.to_dict().items()}
