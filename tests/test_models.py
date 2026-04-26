"""Smoke test the model factory shape."""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from elfcnndet.models import make_cnn


def test_make_cnn_returns_lightning_module() -> None:
    assert isinstance(make_cnn(), pl.LightningModule)


def test_forward_accepts_uint8_byte_indices() -> None:
    model = make_cnn().eval()
    x = torch.randint(0, 256, (4, 256), dtype=torch.long)
    with torch.inference_mode():
        out = model(x)
    assert out.shape == (4, 2)
