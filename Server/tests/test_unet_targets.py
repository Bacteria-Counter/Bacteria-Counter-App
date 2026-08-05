from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.unet.targets import (
    PseudoMaskConfig,
    build_unet_count_manifest,
    render_colony_pseudomask,
    write_unet_count_targets,
)


def write_sample(root: Path, category: str, sample_id: int, colonies_number: int, labels: list[dict]) -> tuple[Path, Path]:
    category_dir = root / "data" / category
    category_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 64, 3), 180, dtype=np.uint8)
    image_path = category_dir / f"{sample_id}.png"
    json_path = category_dir / f"{sample_id}.json"
    cv2.imwrite(str(image_path), image)
    payload = {
        "background": "bright",
        "classes": ["A"],
        "colonies_number": colonies_number,
        "labels": labels,
        "sample_id": sample_id,
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return image_path, json_path


def write_split_csv(root: Path, rows: list[dict]) -> Path:
    import pandas as pd

    path = root / "split.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_render_colony_pseudomask_draws_ellipses() -> None:
    mask = render_colony_pseudomask(
        (40, 50),
        [{"x": 10, "y": 12, "width": 14, "height": 10}],
        PseudoMaskConfig(ellipse_shrink_ratio=0.9, min_radius=2),
    )

    assert mask.shape == (40, 50)
    assert int(mask.sum()) > 0
    assert int((mask > 0).sum()) < 14 * 10


def test_build_unet_count_manifest_marks_empty_and_countable_masks(tmp_path: Path) -> None:
    image_a, json_a = write_sample(
        tmp_path,
        "countable",
        1,
        2,
        [{"x": 10, "y": 10, "width": 8, "height": 8}, {"x": 30, "y": 18, "width": 10, "height": 12}],
    )
    image_b, json_b = write_sample(tmp_path, "empty", 2, 0, [])
    split_csv = write_split_csv(
        tmp_path,
        [
            {
                "image_path": str(image_a.relative_to(tmp_path)),
                "json_path": str(json_a.relative_to(tmp_path)),
                "category": "countable",
                "primary_class": "A",
                "colonies_number": 2,
                "label_count": 2,
                "detection_eligible": True,
            },
            {
                "image_path": str(image_b.relative_to(tmp_path)),
                "json_path": str(json_b.relative_to(tmp_path)),
                "category": "empty",
                "primary_class": "A",
                "colonies_number": 0,
                "label_count": 0,
                "detection_eligible": True,
            },
        ],
    )

    manifest = build_unet_count_manifest(tmp_path, split_csv, tmp_path / "prepared")

    assert len(manifest) == 2
    assert set(manifest["mask_kind"]) == {"pseudo_box_ellipse", "empty_zero"}


def test_write_unet_count_targets_saves_manifest_masks_and_summary(tmp_path: Path) -> None:
    image_path, json_path = write_sample(
        tmp_path,
        "countable",
        1,
        1,
        [{"x": 8, "y": 9, "width": 12, "height": 10}],
    )
    split_csv = write_split_csv(
        tmp_path,
        [
            {
                "image_path": str(image_path.relative_to(tmp_path)),
                "json_path": str(json_path.relative_to(tmp_path)),
                "category": "countable",
                "primary_class": "A",
                "colonies_number": 1,
                "label_count": 1,
                "detection_eligible": True,
            }
        ],
    )

    summary = write_unet_count_targets(tmp_path, split_csv, tmp_path / "prepared")

    assert summary["image_count"] == 1
    assert summary["mask_type"] == "pseudo_masks_from_box_annotations"
    assert (tmp_path / "prepared" / "manifest.csv").exists()
    assert len(list((tmp_path / "prepared" / "masks").glob("*.png"))) == 1
