"""ByteCNN — 1D-CNN over the first 256 bytes of an ELF .text section.

Byte values 0..255 are embedding indices (Embedding(256, 32)); the time axis is
fixed at 256 from Text256Extractor. Two output logits are ordered
(Malware, Benign) to match maldet.toml [output].classes.

Factory entrypoint: ``make_cnn`` — referenced from maldet.toml stages.train.model.
"""

from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn


class ByteCNN(pl.LightningModule):
    """1D-CNN classifier over byte-indexed ELF .text features.

    Inputs: ``long`` tensor of shape ``(N, 256)`` with values in ``[0, 255]``
    (typically up-cast from Text256Extractor's ``uint8`` output by
    LightningTrainer at materialize time).

    Outputs: raw logits of shape ``(N, 2)`` — softmax/argmax is the caller's
    job. Loss is ``CrossEntropyLoss`` with class index 0 = Malware, 1 = Benign
    (matches ``maldet.toml [output].classes`` order).

    Defaults: Adam optimizer at lr=1e-3, no scheduler. Override via constructor
    kwargs (``embedding_dim``/``conv1_out``/``conv2_out``).
    """

    def __init__(self, embedding_dim: int = 32, conv1_out: int = 64, conv2_out: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, conv1_out, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv2_out, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        h = emb.transpose(1, 2)
        h = torch.relu(self.conv1(h))
        h = self.pool(h)
        h = torch.relu(self.conv2(h))
        h = self.gap(h).squeeze(-1)
        return self.fc(h)

    def training_step(self, batch, batch_idx):  # type: ignore[no-untyped-def]
        x, y = batch
        loss = self.loss(self.forward(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def predict(self, x):  # type: ignore[no-untyped-def]
        """Sklearn-compatible argmax prediction over ``forward()``.

        ``maldet.evaluators.binary.BinaryClassification`` and
        ``maldet.builtins.predictors.BatchPredictor`` call ``model.predict``
        / ``model.predict_proba`` directly; Lightning modules don't expose
        those by default, so adapt them here.
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.long, device=self.device)
            logits = self.forward(x_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, x):  # type: ignore[no-untyped-def]
        """Softmax probabilities, classes ordered (Malware, Benign)."""
        self.eval()
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.long, device=self.device)
            logits = self.forward(x_t)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


def make_cnn(**kwargs) -> ByteCNN:  # type: ignore[no-untyped-def]
    return ByteCNN(**kwargs)
