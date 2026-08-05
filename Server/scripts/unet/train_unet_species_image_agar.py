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
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--max-train-images", default=0, type=int)
    parser.add_argument("--max-val-images", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for U-Net training. Install torch in Colab before running this script.") from exc

    from src.classical.species_classification import SpeciesPrediction, classification_metrics
    from src.unet.torch_backend import SpeciesImageDataset, UNetEncoderClassifier, choose_device

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_df = load_species_split(args.train_csv, args.max_train_images)
    val_df = load_species_split(args.val_csv, args.max_val_images)
    class_names = sorted(train_df["primary_class"].astype(str).unique().tolist())
    train_dataset = SpeciesImageDataset(train_df, dataset_dir=args.dataset_dir, class_names=class_names, image_size=args.image_size)
    val_dataset = SpeciesImageDataset(val_df, dataset_dir=args.dataset_dir, class_names=class_names, image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = choose_device(args.device)
    model = UNetEncoderClassifier(
        class_count=len(class_names),
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_state = None
    best_metrics = None
    best_macro_f1 = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics, val_predictions = run_eval_epoch(model, val_loader, device, class_names, classification_metrics, SpeciesPrediction)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_metrics = row
            best_state = {
                "model_type": "unet_species_image_classifier",
                "image_size": args.image_size,
                "base_channels": args.base_channels,
                "dropout": args.dropout,
                "class_names": class_names,
                "state_dict": model.state_dict(),
            }
            pd.DataFrame([prediction.__dict__ for prediction in val_predictions]).to_csv(
                args.output_dir / "val_predictions.csv",
                index=False,
            )
            with (args.output_dir / "val_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint")
    torch.save(best_state, args.output_dir / "model.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    with (args.output_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print("UNet AGAR species image training summary")
    for key, value in best_metrics.items():
        print(f"{key}: {value}")
    print(f"Saved best checkpoint to: {args.output_dir / 'model.pt'}")


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


def run_train_epoch(model, loader, optimizer, device) -> float:
    import torch

    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def run_eval_epoch(model, loader, device, class_names, metrics_fn, prediction_cls):
    import torch

    model.eval()
    total_loss = 0.0
    total_items = 0
    predictions = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            batch_size = int(images.size(0))
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            pred_indices = logits.argmax(dim=1).cpu().tolist()
            true_indices = labels.cpu().tolist()
            for index in range(len(batch["image_path"])):
                predicted_class = class_names[pred_indices[index]]
                true_class = class_names[true_indices[index]]
                predictions.append(
                    prediction_cls(
                        image_path=str(batch["image_path"][index]),
                        true_class=true_class,
                        predicted_class=predicted_class,
                        distance=0.0,
                        correct=predicted_class == true_class,
                        category=str(batch["category"][index]),
                    )
                )
    metrics = metrics_fn(predictions, class_names)
    return total_loss / max(total_items, 1), metrics, predictions


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    main()
