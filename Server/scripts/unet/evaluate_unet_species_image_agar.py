from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--model-pt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--max-images", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for U-Net evaluation. Install torch in Colab before running this script.") from exc

    from src.classical.species_classification import SpeciesPrediction, classification_metrics
    from src.unet.torch_backend import SpeciesImageDataset, UNetEncoderClassifier, choose_device

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.model_pt, map_location="cpu")
    class_names = [str(name) for name in checkpoint["class_names"]]
    split_df = load_species_split(args.split_csv, args.max_images)
    dataset = SpeciesImageDataset(split_df, dataset_dir=args.dataset_dir, class_names=class_names, image_size=int(checkpoint["image_size"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = UNetEncoderClassifier(
        class_count=len(class_names),
        base_channels=int(checkpoint.get("base_channels", 32)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = choose_device(args.device)
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            pred_indices = logits.argmax(dim=1).cpu().tolist()
            for index in range(len(batch["image_path"])):
                predicted_class = class_names[pred_indices[index]]
                predictions.append(
                    SpeciesPrediction(
                        image_path=str(batch["image_path"][index]),
                        true_class=str(batch["true_class"][index]),
                        predicted_class=predicted_class,
                        distance=0.0,
                        correct=predicted_class == str(batch["true_class"][index]),
                        category=str(batch["category"][index]),
                    )
                )
    metrics = classification_metrics(predictions, class_names)
    pd.DataFrame([prediction.__dict__ for prediction in predictions]).to_csv(args.output_dir / "predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("UNet AGAR species image evaluation summary")
    print(f"image_count: {metrics['image_count']}")
    print(f"accuracy: {metrics['accuracy']}")
    print(f"macro_f1: {metrics['macro_f1']}")
    print(f"Saved predictions and metrics to: {args.output_dir}")


def load_species_split(path: Path, max_images: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "species_image_eligible" in frame.columns:
        frame = frame[frame["species_image_eligible"].map(is_truthy)].copy()
    frame = frame[frame["primary_class"].fillna("").astype(str) != ""].copy()
    if max_images > 0:
        frame = frame.head(max_images).copy()
    if frame.empty:
        raise ValueError(f"No species-image rows found in {path}")
    return frame.reset_index(drop=True)


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    main()
