"""Classification heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Two-layer MLP producing ``num_classes`` logits.

    Hidden width is ``max(int(hidden_channels * hidden_multiplier), min_hidden)``
    so it scales with the trunk width. Output is raw logits (no sigmoid) for
    downstream BCEWithLogitsLoss.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        hidden_multiplier: float = 0.5,
        dropout: float = 0.04,
        min_hidden: int = 8,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        h = max(int(hidden_channels * hidden_multiplier), min_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
