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
    parser.add_argument("--model", default="yolov8n.pt", type=str)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--image-size", default=1536, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--optimizer", default="auto", type=str)
    parser.add_argument("--lr0", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--close-mosaic", default=10, type=int)
    parser.add_argument("--cache", default="False", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default="", type=str)
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is required for YOLO training. Install ultralytics in Colab before running this script.") from exc

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = resolve_resume_checkpoint(args.output_dir, args.resume, args.resume_from)
    run_kwargs = {
        "data": str(args.data_yaml),
        "epochs": int(args.epochs),
        "imgsz": int(args.image_size),
        "batch": int(args.batch_size),
        "workers": int(args.workers),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "optimizer": str(args.optimizer),
        "lr0": float(args.lr0),
        "weight_decay": float(args.weight_decay),
        "close_mosaic": int(args.close_mosaic),
        "cache": parse_cache_flag(args.cache),
        "project": str(args.output_dir.parent),
        "name": args.output_dir.name,
        "exist_ok": True,
    }
    if str(args.device).strip():
        run_kwargs["device"] = str(args.device).strip()
    if resume_checkpoint is not None:
        model = YOLO(str(resume_checkpoint))
        run_kwargs["resume"] = True
    else:
        model = YOLO(args.model)
    model.train(**run_kwargs)
    config_path = args.output_dir / "train_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    best_weights = args.output_dir / "weights" / "best.pt"
    last_weights = args.output_dir / "weights" / "last.pt"
    print("AGAR YOLO training summary")
    print(f"output_dir: {args.output_dir}")
    print(f"resume_checkpoint: {resume_checkpoint if resume_checkpoint is not None else 'none'}")
    print(f"best_weights: {best_weights}")
    print(f"last_weights: {last_weights}")
    print(f"Saved training config to: {config_path}")


def parse_cache_flag(value: str) -> bool | str:
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no"}:
        return False
    return value


def resolve_resume_checkpoint(output_dir: Path, resume: bool, resume_from: str) -> Path | None:
    explicit = str(resume_from).strip()
    if explicit:
        explicit_path = Path(explicit)
        if not explicit_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {explicit_path}")
        return explicit_path
    if not resume:
        return None
    auto_checkpoint = output_dir / "weights" / "last.pt"
    if not auto_checkpoint.exists():
        raise FileNotFoundError(f"Could not auto-resume because checkpoint does not exist: {auto_checkpoint}")
    return auto_checkpoint


if __name__ == "__main__":
    main()
