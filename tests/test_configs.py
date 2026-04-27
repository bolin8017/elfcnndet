"""Pydantic config classes for elfcnndet stages — used by maldet 1.1 introspect-schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elfcnndet.configs import EvaluateConfig, PredictConfig, TrainConfig


def test_train_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.epochs == 10
    assert cfg.batch_size == 32
    assert cfg.lr == 1e-3
    assert cfg.embed_dim == 128
    assert cfg.hidden_dim == 256
    assert cfg.patience == 5
    assert cfg.random_state == 42


def test_train_config_rejects_extras() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(unknown=1)


def test_train_config_rejects_zero_epochs() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(epochs=0)


def test_train_config_rejects_zero_batch_size() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(batch_size=0)


def test_train_config_rejects_nonpositive_lr() -> None:
    with pytest.raises(ValidationError):
        TrainConfig(lr=0.0)
    with pytest.raises(ValidationError):
        TrainConfig(lr=-1e-3)


def test_evaluate_config_threshold_range() -> None:
    EvaluateConfig(threshold=0.0)
    EvaluateConfig(threshold=1.0)
    with pytest.raises(ValidationError):
        EvaluateConfig(threshold=-0.01)
    with pytest.raises(ValidationError):
        EvaluateConfig(threshold=1.01)


def test_evaluate_config_rejects_extras() -> None:
    with pytest.raises(ValidationError):
        EvaluateConfig(unknown=1)


def test_predict_config_defaults_and_extras() -> None:
    cfg = PredictConfig()
    assert cfg.batch_size == 256
    with pytest.raises(ValidationError):
        PredictConfig(unknown=1)
