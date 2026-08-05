from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.species_classification import (
    DEFAULT_CLASSIFIER_TYPES,
    SpeciesFeatureConfig,
    evaluate_species_classifier,
    save_species_evaluation,
    save_species_model,
    save_species_selection,
    select_species_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-sizes", default="224,256,320")
    parser.add_argument("--center-crop-ratios", default="0.60,0.72,0.85")
    parser.add_argument("--classifier-types", default=",".join(DEFAULT_CLASSIFIER_TYPES))
    parser.add_argument("--max-train-images", default=0, type=int)
    parser.add_argument("--max-val-images", default=0, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "sweep_results.csv"
    records = load_existing_records(results_path) if args.resume else []
    completed_ids = {int(record["config_id"]) for record in records}
    classifier_types = parse_classifier_types(args.classifier_types)
    configs = list(build_grid(args))
    best_record = max(records, key=ranking_key_from_record) if records else None

    for index, config in enumerate(configs, start=1):
        if index in completed_ids:
            print(f"{index}/{len(configs)} skipped")
            continue
        model, train_features, val_features, selection = select_species_classifier(
            dataset_dir=args.dataset_dir,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            config=config,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            classifier_types=classifier_types,
        )
        train_evaluation = evaluate_species_classifier(
            dataset_dir=args.dataset_dir,
            split_csv=args.train_csv,
            model=model,
            max_images=args.max_train_images,
        )
        val_evaluation = evaluate_species_classifier(
            dataset_dir=args.dataset_dir,
            split_csv=args.val_csv,
            model=model,
            max_images=args.max_val_images,
        )
        record = {
            "config_id": index,
            "image_size": config.image_size,
            "center_crop_ratio": config.center_crop_ratio,
            "classifier_type": model.classifier_type,
            "train_accuracy": train_evaluation.metrics["accuracy"],
            "train_macro_f1": train_evaluation.metrics["macro_f1"],
            "val_accuracy": val_evaluation.metrics["accuracy"],
            "val_macro_f1": val_evaluation.metrics["macro_f1"],
            "val_macro_precision": val_evaluation.metrics["macro_precision"],
            "val_macro_recall": val_evaluation.metrics["macro_recall"],
            "generalization_gap": train_evaluation.metrics["accuracy"] - val_evaluation.metrics["accuracy"],
        }
        records.append(record)
        write_records(records, results_path)
        if best_record is None or ranking_key_from_record(record) > ranking_key_from_record(best_record):
            best_record = record
            save_best_run(
                output_dir=args.output_dir,
                model=model,
                train_features=train_features,
                val_features=val_features,
                selection=selection,
                train_evaluation=train_evaluation,
                val_evaluation=val_evaluation,
                record=record,
            )
        print(
            f"{index}/{len(configs)} "
            f"classifier={model.classifier_type} "
            f"val_accuracy={val_evaluation.metrics['accuracy']} "
            f"val_macro_f1={val_evaluation.metrics['macro_f1']} "
            f"gap={train_evaluation.metrics['accuracy'] - val_evaluation.metrics['accuracy']}"
        )

    write_records(records, results_path)
    if best_record is None:
        raise RuntimeError("No species configurations were evaluated")

    print("Best classical AGAR species image validation summary")
    print(f"config_id: {best_record['config_id']}")
    print(f"image_size: {best_record['image_size']}")
    print(f"center_crop_ratio: {best_record['center_crop_ratio']}")
    print(f"classifier_type: {best_record['classifier_type']}")
    print(f"train_accuracy: {best_record['train_accuracy']}")
    print(f"train_macro_f1: {best_record['train_macro_f1']}")
    print(f"val_accuracy: {best_record['val_accuracy']}")
    print(f"val_macro_f1: {best_record['val_macro_f1']}")
    print(f"Saved sweep outputs to: {args.output_dir}")


def build_grid(args: argparse.Namespace) -> list[SpeciesFeatureConfig]:
    return [
        SpeciesFeatureConfig(image_size=image_size, center_crop_ratio=center_crop_ratio)
        for image_size in parse_int_list(args.image_sizes)
        for center_crop_ratio in parse_float_list(args.center_crop_ratios)
    ]


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def write_records(records: list[dict], path: Path) -> None:
    frame = pd.DataFrame(records).sort_values(
        ["val_macro_f1", "val_accuracy", "val_macro_precision"],
        ascending=[False, False, False],
    )
    frame.to_csv(path, index=False)


def save_best_run(
    output_dir: Path,
    model,
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    selection: pd.DataFrame,
    train_evaluation,
    val_evaluation,
    record: dict,
) -> None:
    best_dir = output_dir / "best_run"
    best_dir.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(best_dir / "train_features.csv", index=False)
    val_features.to_csv(best_dir / "val_features.csv", index=False)
    save_species_selection(selection, best_dir)
    save_species_model(model, best_dir / "model.json")
    save_species_evaluation(train_evaluation, best_dir / "train")
    save_species_evaluation(val_evaluation, best_dir / "val")
    with (best_dir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def ranking_key_from_record(record: dict) -> tuple[float, float, float]:
    return (
        float(record["val_macro_f1"]),
        float(record["val_accuracy"]),
        -float(record["generalization_gap"]),
    )


def parse_classifier_types(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
