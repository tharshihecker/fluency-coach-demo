"""Neural model architectures saved by the research notebooks."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Residual MLP architecture used by the five-fold research workflow."""

    def __init__(self, n_features: int, dropout: float = 0.12) -> None:
        super().__init__()
        self.start = nn.Linear(n_features, 64)
        self.block = nn.Sequential(
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        self.out = nn.Sequential(nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.start(x)
        hidden = hidden + 0.5 * self.block(hidden)
        return self.out(hidden).squeeze(-1)
