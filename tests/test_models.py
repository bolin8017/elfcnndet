"""Smoke test the model factory shape."""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from elfcnndet.models import make_cnn


def test_make_cnn_returns_lightning_module() -> None:
    assert isinstance(make_cnn(), pl.LightningModule)


def test_forward_accepts_long_tensor_in_byte_value_range() -> None:
    """Embedding requires long indices; trainer up-casts uint8 features at materialize time."""
    model = make_cnn().eval()
    x = torch.randint(0, 256, (4, 256), dtype=torch.long)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (4, 2)


def test_training_step_returns_scalar_loss() -> None:
    """training_step must return a scalar tensor with grad for Lightning's autograd."""
    model = make_cnn()
    x = torch.randint(0, 256, (4, 256), dtype=torch.long)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loss = model.training_step((x, y), 0)
    assert loss.dim() == 0
    assert loss.requires_grad
    assert loss.item() > 0  # CrossEntropyLoss is non-negative


def test_configure_optimizers_returns_adam_with_pinned_lr() -> None:
    """configure_optimizers must return Adam at lr=1e-3 (pinned for reproducibility)."""
    optimizer = make_cnn().configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 1e-3
