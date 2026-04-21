"""1D-CNN architecture — byte embedding + conv stack + linear head."""

from __future__ import annotations

import torch
from torch import nn


class ByteCNN(nn.Module):
    """Input: (B, L) long tensor in [0, 255]. Output: (B, 2) logits."""

    def __init__(
        self,
        num_classes: int = 2,
        embedding_dim: int = 32,
        conv1_out: int = 64,
        conv2_out: int = 128,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, conv1_out, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv2_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) long
        emb = self.embed(x)                     # (B, L, E)
        h = emb.transpose(1, 2)                 # (B, E, L)
        h = torch.relu(self.conv1(h))
        h = self.pool(h)
        h = torch.relu(self.conv2(h))
        h = self.gap(h).squeeze(-1)             # (B, C2)
        return self.fc(h)                       # (B, num_classes)
