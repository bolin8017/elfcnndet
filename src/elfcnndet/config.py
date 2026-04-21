"""Configuration for the 1D-CNN ELF detector."""

from pathlib import Path

from maldet.config import (
    BaseDetectorConfig,
    DataConfig as BaseDataConfig,
)
from pydantic import BaseModel, ConfigDict, field_validator


class DataConfig(BaseDataConfig):
    dataset: Path = Path("./data/samples")


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    section_name: str = ".text"
    size: int = 256

    @field_validator("size")
    @classmethod
    def _positive_size(cls, v: int) -> int:
        if v <= 0 or v > 65536:
            raise ValueError("size must be in (0, 65536]")
        return v


class ModelConfig(BaseModel):
    """1D-CNN hyperparameters."""

    model_config = ConfigDict(extra="allow", frozen=True)

    embedding_dim: int = 32
    conv1_out: int = 64
    conv2_out: int = 128
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    val_split: float = 0.1
    random_state: int = 42

    @field_validator("epochs")
    @classmethod
    def _at_least_one_epoch(cls, v: int) -> int:
        if v < 1:
            raise ValueError("epochs must be >= 1")
        return v

    @field_validator("batch_size")
    @classmethod
    def _at_least_one_batch(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v

    @field_validator("val_split")
    @classmethod
    def _fraction(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("val_split must be in [0, 1)")
        return v


class ElfCnnDetectorConfig(BaseDetectorConfig):
    data: DataConfig = DataConfig()
    feature: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    seed: int = 42
