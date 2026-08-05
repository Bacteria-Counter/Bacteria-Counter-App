"""Trains CSRNet on our AGAR + SAM-Copy-Paste-synthetic colony dataset
(density-map regression instead of box detection). See
generate_csrnet_density_maps.py for how the target maps were made.

Density-map regression counts by summing a predicted heatmap rather than
drawing boxes -- no NMS, no double-counting from overlapping boxes, which is
exactly the failure mode this project keeps running into with YOLO on
dense/overlapping colonies. This is a genuinely different architecture, not
another preprocessing tweak.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from csrnet_model import CSRNet

DATA_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/csrnet_dataset")
CKPT_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/repos/bacterial-colony-detection/csrnet_runs")
CKPT_DIR.mkdir(exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ColonyDensityDataset(Dataset):
    def __init__(self, split):
        self.img_dir = DATA_DIR / "images" / split
        self.den_dir = DATA_DIR / "density" / split
        self.files = sorted(f.stem for f in self.img_dir.glob("*.jpg"))
        self.transform = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stem = self.files[idx]
        img = cv2.imread(str(self.img_dir / f"{stem}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        density = np.load(self.den_dir / f"{stem}.npy")
        img_t = self.transform(img)
        density_t = torch.from_numpy(density).unsqueeze(0)  # 1xHxW
        return img_t, density_t


def evaluate(model, loader, device):
    model.eval()
    total_abs_err = 0.0
    n = 0
    with torch.no_grad():
        for imgs, densities in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            for i in range(imgs.shape[0]):
                pred_count = preds[i].sum().item()
                gt_count = densities[i].sum().item()
                total_abs_err += abs(pred_count - gt_count)
                n += 1
    return total_abs_err / n


def main(epochs=50, batch_size=8, lr=1e-5, device_str=None):
    if device_str is None:
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    train_ds = ColonyDensityDataset("train")
    val_ds = ColonyDensityDataset("val")
    print(f"train={len(train_ds)} images, val={len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CSRNet().to(device)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_mae = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for imgs, densities in train_loader:
            imgs, densities = imgs.to(device), densities.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, densities)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_time = time.time() - t0
        mae = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs}  loss={running_loss/len(train_ds):.4f}  "
              f"val_MAE={mae:.2f}  time={train_time:.0f}s")

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), CKPT_DIR / "best.pt")
            print(f"  -> new best, saved (MAE={mae:.2f})")
        torch.save(model.state_dict(), CKPT_DIR / "last.pt")

    print(f"Done. Best val MAE: {best_mae:.2f}")


if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    main(epochs=epochs)
