"""GINE encoder for per-beat graph subgraphs."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool


class SubgraphEncoder(nn.Module):
    """GINE message-passing stack followed by global mean pool per subgraph.

    Each layer is a post-LN residual block: ``norm(h + dropout(conv(h)))``.
    ``edge_index`` is passed through unchanged, leaving GINE's canonical
    self-message semantics intact.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 128,
        mlp_hidden: int = 256,
        num_layers: int = 3,
        edge_dim: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {hidden_channels}")
        if mlp_hidden <= 0:
            raise ValueError(f"mlp_hidden must be > 0, got {mlp_hidden}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, mlp_hidden),
                nn.ReLU(),
                nn.Linear(mlp_hidden, hidden_channels),
            )
            self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.hidden_channels = hidden_channels

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        num_subgraphs: int,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h_out = conv(h, edge_index, edge_attr=edge_attr)
            h = norm(h + self.dropout(h_out))
        return global_mean_pool(h, batch, size=num_subgraphs)
