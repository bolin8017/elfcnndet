"""Tests for maldet.toml shape — guard against accidental drift."""

from __future__ import annotations

import json
from pathlib import Path

from maldet.manifest import load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_loads_via_maldet() -> None:
    """The manifest detector version must match the package version (single source of truth)."""
    import tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    expected_version = pyproject["project"]["version"]

    m = load_manifest(REPO_ROOT / "maldet.toml")
    assert m.detector.name == "elfcnndet"
    assert m.detector.version == expected_version
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

    Under maldet 2.0, ``classes`` is alphabetical and ``positive_class`` is
    explicit — together they pin the binary CM orientation in the platform
    UI, so flipping either one would silently invert metrics."""
    m = load_manifest(REPO_ROOT / "maldet.toml")
    assert m.input.binary_format == "elf"
    assert m.input.required_sections == [".text"]
    assert m.output.classes == ["Benign", "Malware"]
    assert m.output.positive_class == "Malware"
    assert m.compat.min_maldet == "2.2"
    assert m.compat.schema_version == 2


def test_manifest_has_config_class_per_stage() -> None:
    m = load_manifest(REPO_ROOT / "maldet.toml")
    assert m.stages["train"].config_class == "elfcnndet.configs:TrainConfig"
    assert m.stages["evaluate"].config_class == "elfcnndet.configs:EvaluateConfig"
    assert m.stages["predict"].config_class == "elfcnndet.configs:PredictConfig"
    assert m.stages["train"].params_schema == {}
    assert m.stages["evaluate"].params_schema == {}
    assert m.stages["predict"].params_schema == {}


def test_introspect_schema_for_train_config_is_valid_json_schema(tmp_path: Path) -> None:
    """Round-trip: TrainConfig → introspect-schema → JSON Schema with the right shape."""
    from maldet.cli import app
    from typer.testing import CliRunner

    out = tmp_path / "train_schema.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["introspect-schema", "--config-class", "elfcnndet.configs:TrainConfig", "--out", str(out)],
    )
    assert result.exit_code == 0, (result.stdout, result.stderr)
    schema = json.loads(out.read_text())
    assert schema.get("additionalProperties") is False
    assert "epochs" in schema["properties"]
    assert schema["properties"]["epochs"]["minimum"] == 1
    assert schema["properties"]["lr"]["exclusiveMinimum"] == 0.0


def test_introspect_schema_for_evaluate_config_has_no_properties(tmp_path: Path) -> None:
    """Guard the v4.1.0 footgun-removal: EvaluateConfig must remain empty.

    A future re-addition of threshold (or any other field) would be caught
    here. The TrainConfig round-trip test next to this one shows the same
    pattern for fields that legitimately exist.
    """
    from maldet.cli import app
    from typer.testing import CliRunner

    out = tmp_path / "evaluate_schema.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "introspect-schema",
            "--config-class",
            "elfcnndet.configs:EvaluateConfig",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, (result.stdout, result.stderr)
    schema = json.loads(out.read_text())
    assert schema.get("additionalProperties") is False
    assert schema.get("properties", {}) == {}, "EvaluateConfig must expose no params"
