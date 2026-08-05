from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--image-size", default=1536, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--confidence-threshold", default=0.001, type=float)
    parser.add_argument("--iou-threshold", default=0.6, type=float)
    parser.add_argument("--max-det", default=1000, type=int)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is required for YOLO evaluation. Install ultralytics in Colab before running this script.") from exc

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model_path))
    val_kwargs = {
        "data": str(args.data_yaml),
        "split": str(args.split),
        "imgsz": int(args.image_size),
        "batch": int(args.batch_size),
        "workers": int(args.workers),
        "conf": float(args.confidence_threshold),
        "iou": float(args.iou_threshold),
        "max_det": int(args.max_det),
        "save_json": bool(args.save_json),
        "plots": bool(args.plots),
        "project": str(args.output_dir.parent),
        "name": args.output_dir.name,
        "exist_ok": True,
        "verbose": False,
    }
    if str(args.device).strip():
        val_kwargs["device"] = str(args.device).strip()
    metrics = model.val(**val_kwargs)
    summary = summarize_metrics(metrics)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "evaluation_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print("AGAR YOLO species evaluation summary")
    for key, value in summary.items():
        if key == "class_maps":
            continue
        print(f"{key}: {value}")
    print(f"Saved evaluation outputs to: {args.output_dir}")


def summarize_metrics(metrics) -> dict:
    box = getattr(metrics, "box", None)
    summary = {
        "fitness": float(getattr(metrics, "fitness", 0.0)),
        "box_map": float(getattr(box, "map", 0.0)) if box is not None else 0.0,
        "box_map50": float(getattr(box, "map50", 0.0)) if box is not None else 0.0,
        "box_map75": float(getattr(box, "map75", 0.0)) if box is not None else 0.0,
        "class_maps": [float(value) for value in list(getattr(box, "maps", []))] if box is not None else [],
    }
    return summary


if __name__ == "__main__":
    main()
