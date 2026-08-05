from __future__ import annotations

import pytest

import scripts.unet.train_unet_count_agar as train_unet_count_agar


def test_train_script_exposes_guarded_torch_symbol() -> None:
    assert hasattr(train_unet_count_agar, "torch")


torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset

run_eval_epoch = train_unet_count_agar.run_eval_epoch


class TinyCountDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        value = float(index)
        return {
            "image": torch.full((3, 16, 16), value, dtype=torch.float32),
            "mask": torch.zeros((1, 16, 16), dtype=torch.float32),
        }


class ZeroLogitModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.size(0), 1, x.size(2), x.size(3)), dtype=x.dtype, device=x.device)


def test_run_eval_epoch_returns_metrics_without_name_error() -> None:
    loader = DataLoader(TinyCountDataset(), batch_size=2, shuffle=False)
    model = ZeroLogitModel()
    loss, metrics = run_eval_epoch(
        model,
        loader,
        torch.device("cpu"),
        lambda logits, masks: torch.nn.functional.binary_cross_entropy_with_logits(logits, masks),
        lambda logits, masks: {"dice": 1.0, "iou": 1.0, "pixel_accuracy": 1.0},
    )

    assert loss >= 0.0
    assert metrics == {"dice": 1.0, "iou": 1.0, "pixel_accuracy": 1.0}
