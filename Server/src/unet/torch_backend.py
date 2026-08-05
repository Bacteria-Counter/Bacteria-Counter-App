from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from src.classical.detect_count import read_image

class CountSegmentationDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, dataset_dir: Path, manifest_root: Path, image_size: int) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.manifest_root = manifest_root
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        image = read_image(self.dataset_dir / str(row["image_path"]))
        mask = cv2.imread(str(self.manifest_root / str(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {self.manifest_root / str(row['mask_path'])}")
        image_tensor = image_to_tensor(image, self.image_size)
        mask_tensor = mask_to_tensor(mask, self.image_size)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(row["image_path"]),
            "true_count": int(row["colonies_number"]),
            "category": str(row.get("category", "")),
            "primary_class": str(row.get("primary_class", "")),
        }


class SpeciesImageDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, dataset_dir: Path, class_names: list[str], image_size: int) -> None:
        self.split_df = split_df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.class_names = class_names
        self.class_to_index = {name: index for index, name in enumerate(class_names)}
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.split_df)

    def __getitem__(self, index: int) -> dict:
        row = self.split_df.iloc[index]
        image = read_image(self.dataset_dir / str(row["image_path"]))
        image_tensor = image_to_tensor(image, self.image_size)
        class_name = str(row["primary_class"])
        if class_name not in self.class_to_index:
            raise KeyError(f"Unknown class name: {class_name}")
        return {
            "image": image_tensor,
            "label": int(self.class_to_index[class_name]),
            "image_path": str(row["image_path"]),
            "true_class": class_name,
            "category": str(row.get("category", "")),
        }


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = DoubleConv(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNetSegmenter(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 1) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels=in_channels, base_channels=base_channels)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.head(x)


class UNetEncoderClassifier(nn.Module):
    def __init__(self, class_count: int, in_channels: int = 3, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_channels=in_channels, base_channels=base_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(base_channels * 16, class_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, x5 = self.encoder(x)
        return self.head(self.pool(x5))


def image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw)


def mask_to_tensor(mask: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    normalized = (resized > 0).astype(np.float32)[None, :, :]
    return torch.from_numpy(normalized)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)


def segmentation_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    pred_sum = predictions.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))
    union = pred_sum + target_sum - intersection
    dice = ((2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    accuracy = (predictions.eq(targets)).float().mean().item()
    return {"dice": float(dice), "iou": float(iou), "pixel_accuracy": float(accuracy)}


def choose_device(device_name: str = "") -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
