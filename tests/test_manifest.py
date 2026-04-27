"""Tests for maldet.toml shape — guard against accidental drift."""

from __future__ import annotations

from pathlib import Path

from maldet.manifest import load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_loads_via_maldet() -> None:
    m = load_manifest(REPO_ROOT / "maldet.toml")
    assert m.detector.name == "elfcnndet"
    assert m.detector.version == "2.0.0"
    assert m.detector.framework == "lightning"


def test_manifest_resources_supports_multi_gpu() -> None:
    m = load_manifest(REPO_ROOT / "maldet.toml")
    assert m.resources.supports == ["cpu", "gpu1", "gpu2"]


def test_manifest_lifecycle_supports_ddp() -> None:
    m = load_manifest(REPO_ROOT / "maldet.toml")
    assert m.lifecycle.supports_distributed == "ddp"
    assert set(m.lifecycle.stages) == {"train", "evaluate", "predict"}


def test_manifest_stages_reference_local_extractor_and_lightning_trainer() -> None:
    m = load_manifest(REPO_ROOT / "maldet.toml")
    train = m.stages["train"]
    assert train.extractor == "elfcnndet.features:Text256Extractor"
    assert train.model == "elfcnndet.models:make_cnn"
    assert train.trainer == "maldet.trainers.lightning_trainer:LightningTrainer"


def test_manifest_io_contract() -> None:
    """[input] / [output] / [compat] fields the operator + trainer depend on.

    classes order matters for binary metrics — flipping it would silently
    invert AUC/precision."""
    m = load_manifest(REPO_ROOT / "maldet.toml")
    assert m.input.binary_format == "elf"
    assert m.input.required_sections == [".text"]
    assert m.output.classes == ["Malware", "Benign"]
    assert m.compat.min_maldet == "1.0"
