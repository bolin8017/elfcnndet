"""1D-CNN ELF malware detector with multi-GPU DataParallel.

Detects available GPUs via `torch.cuda.device_count()`:
- 0 GPUs → CPU
- 1 GPU  → single-GPU training
- ≥2 GPUs → nn.DataParallel splits the batch across all allocated GPUs

On lolday, set `resource_profile="gpu2"` when submitting the Job; the
platform allocates 2× nvidia.com/gpu to the pod, which PyTorch then sees
as `device_count() == 2`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from maldet import BaseDetector
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ElfCnnDetectorConfig
from .model import ByteCNN
from .text256 import extract_text256

MODEL_FILENAME = "model.pt"
META_FILENAME = "meta.json"


class ElfCnnDetector(BaseDetector):
    """1D-CNN detector over the first 256 bytes of `.text`."""

    config_class = ElfCnnDetectorConfig

    def __init__(self, config: ElfCnnDetectorConfig | None = None) -> None:
        super().__init__(config)
        torch.manual_seed(self.config.seed)
        # Both guards are necessary: a CUDA-capable host whose driver is
        # older than the torch build returns `device_count > 0` even though
        # `is_available()` is False — actually .to("cuda") then crashes. The
        # platform pods always satisfy both (driver matches image), so the
        # combined check just adds safety for dev laptops.
        cuda_ok = torch.cuda.is_available()
        self._device_count = torch.cuda.device_count() if cuda_ok else 0
        self._device = torch.device("cuda" if cuda_ok and self._device_count > 0 else "cpu")
        self._model: nn.Module | None = None
        self.logger.info(
            "detector_initialized",
            device=str(self._device),
            gpu_device_count=self._device_count,
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size,
        )
        self._log_gpu_tags()

    def _log_gpu_tags(self) -> None:
        """Emit MLflow run tags describing the allocated GPU topology."""
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return
        try:
            import mlflow
            mlflow.set_tag("gpu.device_count", self._device_count)
            if self._device_count > 0:
                names = [torch.cuda.get_device_name(i) for i in range(self._device_count)]
                mlflow.set_tag("gpu.device_names", ",".join(names))
        except Exception as exc:   # noqa: BLE001 — MLflow is best-effort
            self.logger.warning("mlflow_gpu_tag_failed", reason=str(exc))

    def train(self) -> Path:
        df = self._load_csv(self.config.data.train)
        X, y = self._build_tensors(df)
        self.logger.info("training_started", samples=X.shape[0])

        base_model = ByteCNN(
            embedding_dim=self.config.model.embedding_dim,
            conv1_out=self.config.model.conv1_out,
            conv2_out=self.config.model.conv2_out,
        )
        if self._device_count >= 2:
            model: nn.Module = nn.DataParallel(base_model)
            self.logger.info("data_parallel_enabled", gpus=self._device_count)
        else:
            model = base_model
        model.to(self._device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.model.lr)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=self.config.model.batch_size,
            shuffle=True,
        )

        t0 = time.time()
        model.train()
        for epoch in range(1, self.config.model.epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(self._device, non_blocking=True)
                yb = yb.to(self._device, non_blocking=True)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += xb.size(0)
            avg_loss = epoch_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            self._log_metric_step("train_loss", avg_loss, step=epoch)
            self._log_metric_step("train_acc", train_acc, step=epoch)
            self.logger.info(
                "epoch_done", epoch=epoch, loss=avg_loss, train_acc=train_acc
            )
        train_time = time.time() - t0
        self._log_metric_step("train_time_seconds", train_time)

        model_dir = self.config.output.model
        self.ensure_directory_exists(model_dir)
        # Save the inner state_dict (strip DataParallel's `module.` prefix).
        inner = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(inner.state_dict(), model_dir / MODEL_FILENAME)
        meta = {
            "embedding_dim": self.config.model.embedding_dim,
            "conv1_out": self.config.model.conv1_out,
            "conv2_out": self.config.model.conv2_out,
            "feature_size": self.config.feature.size,
        }
        (model_dir / META_FILENAME).write_text(pd.Series(meta).to_json())
        self._model = model
        self.logger.info(
            "training_completed",
            model_path=str(model_dir / MODEL_FILENAME),
            duration_seconds=round(train_time, 2),
        )
        return model_dir

    def evaluate(self) -> dict[str, Any]:
        if self._model is None:
            self._load_model()
        df = self._load_csv(self.config.data.test)
        X, y = self._build_tensors(df)
        y_pred = self._batch_predict(X)

        metrics = {
            "accuracy": float(accuracy_score(y.cpu(), y_pred)),
            "precision": float(precision_score(y.cpu(), y_pred, zero_division=0)),
            "recall": float(recall_score(y.cpu(), y_pred, zero_division=0)),
            "f1": float(f1_score(y.cpu(), y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y.cpu(), y_pred).tolist(),
            "n_samples": int(X.shape[0]),
            "gpu_device_count": int(self._device_count),
        }
        self.logger.info(
            "evaluation_completed", accuracy=metrics["accuracy"], f1=metrics["f1"]
        )
        return metrics

    def predict(self) -> Path:
        if self._model is None:
            self._load_model()
        df = self._load_csv(self.config.data.predict)
        X, _ = self._build_tensors(df, require_labels=False)
        y_pred = self._batch_predict(X)
        # Softmax malware-class probability for the score column.
        with torch.no_grad():
            self._model.eval()
            probs = []
            loader = DataLoader(TensorDataset(X), batch_size=self.config.model.batch_size)
            for (xb,) in loader:
                xb = xb.to(self._device, non_blocking=True)
                logits = self._model(xb)
                probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            scores = np.concatenate(probs)

        out_df = pd.DataFrame({
            "file_name": df["file_name"].values,
            "pred_label": ["Malware" if p == 1 else "Benign" for p in y_pred],
            "pred_score": scores,
        })

        out_path = self.config.output.prediction
        self.ensure_directory_exists(out_path)
        if out_path.is_dir() or out_path.suffix == "":
            out_path = out_path / "predictions.csv"
        out_df.to_csv(out_path, index=False)
        self.logger.info("prediction_completed", output=str(out_path))
        return out_path

    # ------------------------------------------------------------------ helpers

    def _batch_predict(self, X: torch.Tensor) -> np.ndarray:
        assert self._model is not None
        loader = DataLoader(TensorDataset(X), batch_size=self.config.model.batch_size)
        self._model.eval()
        preds: list[int] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self._device, non_blocking=True)
                preds.extend(self._model(xb).argmax(1).cpu().tolist())
        return np.asarray(preds, dtype=np.int64)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"dataset CSV not found: {path}")
        df = pd.read_csv(path)
        if "file_name" not in df.columns:
            raise ValueError(f"{path}: CSV missing 'file_name' column")
        return df

    def _build_tensors(
        self, df: pd.DataFrame, require_labels: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dataset_root = self.config.data.dataset
        size = self.config.feature.size

        vectors: list[np.ndarray] = []
        labels: list[int] = []
        kept_rows: list[int] = []
        for idx, row in df.iterrows():
            sha = row["file_name"]
            sample_path = dataset_root / sha[:2] / sha
            try:
                vec = extract_text256(sample_path, size=size)
            except (FileNotFoundError, ValueError) as exc:
                self.logger.warning("sample_skipped", file=sha, reason=str(exc))
                continue
            vectors.append(vec)
            kept_rows.append(idx)
            if require_labels:
                label = row.get("label")
                if label not in ("Malware", "Benign"):
                    raise ValueError(
                        f"row {idx}: label must be 'Malware' or 'Benign', got {label!r}"
                    )
                labels.append(1 if label == "Malware" else 0)

        if not vectors:
            raise RuntimeError("no valid samples after feature extraction")

        df.drop(index=[i for i in df.index if i not in kept_rows], inplace=True)
        df.reset_index(drop=True, inplace=True)

        X = torch.from_numpy(np.stack(vectors).astype(np.int64))
        y = torch.tensor(labels, dtype=torch.int64) if require_labels else torch.tensor([])
        return X, y

    def _load_model(self) -> None:
        model_dir = self.config.output.model
        meta_path = model_dir / META_FILENAME
        weights_path = model_dir / MODEL_FILENAME
        if not weights_path.exists():
            raise FileNotFoundError(f"trained model not found: {weights_path}")

        meta = pd.read_json(meta_path, typ="series") if meta_path.exists() else pd.Series({})
        base = ByteCNN(
            embedding_dim=int(meta.get("embedding_dim", self.config.model.embedding_dim)),
            conv1_out=int(meta.get("conv1_out", self.config.model.conv1_out)),
            conv2_out=int(meta.get("conv2_out", self.config.model.conv2_out)),
        )
        base.load_state_dict(torch.load(weights_path, map_location=self._device))
        if self._device_count >= 2:
            model: nn.Module = nn.DataParallel(base)
        else:
            model = base
        model.to(self._device)
        self._model = model
        self.logger.info("model_loaded", path=str(weights_path))

    def _log_metric_step(self, key: str, value: float, step: int | None = None) -> None:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return
        try:
            import mlflow
            if step is None:
                mlflow.log_metric(key, value)
            else:
                mlflow.log_metric(key, value, step=step)
        except Exception as exc:   # noqa: BLE001
            self.logger.debug("mlflow_log_metric_failed", key=key, reason=str(exc))
