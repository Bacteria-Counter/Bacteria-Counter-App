from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--train-manifest-csv", required=True, type=Path)
    parser.add_argument("--val-manifest-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--max-train-images", default=0, type=int)
    parser.add_argument("--max-val-images", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    try:
        if torch is None:
            raise ModuleNotFoundError("torch")
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for U-Net training. Install torch in Colab before running this script.") from exc

    from src.unet.torch_backend import (
        CountSegmentationDataset,
        UNetSegmenter,
        choose_device,
        segmentation_loss,
        segmentation_metrics_from_logits,
    )

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = load_manifest(args.train_manifest_csv, args.max_train_images)
    val_manifest = load_manifest(args.val_manifest_csv, args.max_val_images)
    train_dataset = CountSegmentationDataset(
        train_manifest,
        dataset_dir=args.dataset_dir,
        manifest_root=args.train_manifest_csv.parent,
        image_size=args.image_size,
    )
    val_dataset = CountSegmentationDataset(
        val_manifest,
        dataset_dir=args.dataset_dir,
        manifest_root=args.val_manifest_csv.parent,
        image_size=args.image_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = choose_device(args.device)
    model = UNetSegmenter(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_state = None
    best_metrics = None
    best_dice = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device, segmentation_loss)
        val_loss, val_metrics = run_eval_epoch(model, val_loader, device, segmentation_loss, segmentation_metrics_from_logits)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        history.append(row)
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_metrics = row
            best_state = {
                "model_type": "unet_count_segmenter",
                "image_size": args.image_size,
                "base_channels": args.base_channels,
                "state_dict": model.state_dict(),
            }
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
        )
    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint")
    torch.save(best_state, args.output_dir / "model.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print("UNet AGAR counting training summary")
    for key, value in best_metrics.items():
        print(f"{key}: {value}")
    print(f"Saved best checkpoint to: {args.output_dir / 'model.pt'}")


def load_manifest(path: Path, max_images: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if max_images > 0:
        frame = frame.head(max_images).copy()
    if frame.empty:
        raise ValueError(f"No rows found in {path}")
    return frame


def run_train_epoch(model, loader, optimizer, device, loss_fn) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def run_eval_epoch(model, loader, device, loss_fn, metric_fn) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    metric_sums = {"dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = loss_fn(logits, masks)
            metrics = metric_fn(logits, masks)
            batch_size = int(images.size(0))
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            for key in metric_sums:
                metric_sums[key] += metrics[key] * batch_size
    averaged = {key: value / max(total_items, 1) for key, value in metric_sums.items()}
    return total_loss / max(total_items, 1), averaged


if __name__ == "__main__":
    main()
