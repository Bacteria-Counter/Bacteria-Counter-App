from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--center-crop-ratio", default=0.72, type=float)
    parser.add_argument("--max-train-images", default=0, type=int)
    parser.add_argument("--max-val-images", default=0, type=int)
    parser.add_argument("--classifier-types", default=",".join(DEFAULT_CLASSIFIER_TYPES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SpeciesFeatureConfig(
        image_size=args.image_size,
        center_crop_ratio=args.center_crop_ratio,
    )
    model, train_features, val_features, selection = select_species_classifier(
        dataset_dir=args.dataset_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        config=config,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
        classifier_types=parse_classifier_types(args.classifier_types),
    )
    train_features.to_csv(args.output_dir / "train_features.csv", index=False)
    val_features.to_csv(args.output_dir / "val_features.csv", index=False)
    save_species_selection(selection, args.output_dir)
    save_species_model(model, args.output_dir / "model.json")
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
    save_species_evaluation(train_evaluation, args.output_dir / "train")
    save_species_evaluation(val_evaluation, args.output_dir / "val")

    print("Classical AGAR species image classification summary")
    print(f"classifier_type: {model.classifier_type}")
    print(f"classes: {model.classes}")
    print(f"train_image_count: {train_evaluation.metrics['image_count']}")
    print(f"train_accuracy: {train_evaluation.metrics['accuracy']}")
    print(f"train_macro_f1: {train_evaluation.metrics['macro_f1']}")
    print(f"val_image_count: {val_evaluation.metrics['image_count']}")
    print(f"val_accuracy: {val_evaluation.metrics['accuracy']}")
    print(f"val_macro_f1: {val_evaluation.metrics['macro_f1']}")
    print(f"Saved model and metrics to: {args.output_dir}")


def parse_classifier_types(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
