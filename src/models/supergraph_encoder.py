"""GATv2 encoder for the supergraph of subgraph embeddings.

Receives pre-computed edge attributes (assembled upstream by HierGNNModel)
and runs GATv2Conv message passing.  Lead identity is added to each node
via a learned embedding.

"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv


class SupergraphEncoder(nn.Module):
    """Stack of GATv2Conv layers over the supergraph with edge step bias."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_edge_step: int = 1,
        num_leads: int = 12,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
            )
        if max_edge_step < 1:
            raise ValueError(f"max_edge_step must be >= 1, got {max_edge_step}")

        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.max_edge_step = max_edge_step
        self.num_leads = num_leads

        self.lead_embedding = nn.Embedding(num_leads, hidden_channels)
        self.step_embedding = nn.Embedding(2 * max_edge_step + 3, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                    edge_dim=hidden_channels,
                    share_weights=False,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.dropout = nn.Dropout(dropout)

    def _bucketize(self, edge_step: torch.Tensor) -> torch.Tensor:
        """Map signed step values to embedding bucket ids.

        Bucket layout: ``0=underflow``, ``1..(2K+1)`` for ``[-K..K]``,
        ``2K+2=overflow``.
        """
        if edge_step.numel() == 0:
            return edge_step.new_zeros((0,), dtype=torch.long)

        step = edge_step.to(dtype=torch.long)
        k = self.max_edge_step
        bucket = torch.full_like(step, fill_value=2 * k + 2)
        in_range = (step >= -k) & (step <= k)
        bucket[step < -k] = 0
        bucket[in_range] = step[in_range] + k + 1
        return bucket

    def embed_step(self, edge_step: torch.Tensor) -> torch.Tensor:
        """Map raw signed step values to learned embeddings ``[E, H]``."""
        return self.step_embedding(self._bucketize(edge_step))

    def forward(
        self,
        x: torch.Tensor,
        super_edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        subgraph_lead: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[Tensor, Tensor]]]:
        """Run GATv2 message passing over the supergraph.

        Returns ``(h, attentions)`` where ``attentions`` is the per-layer
        attention ``[(edge_index, alpha), ...]`` (``alpha`` is ``[E, heads]``).
        The attention is GATv2Conv's own per-edge softmax, computed unconditionally
        as part of the normal forward pass, so surfacing it costs nothing extra and
        callers that don't need it (the training/eval hot path) simply discard it.
        """
        if x.size(0) == 0:
            return x, []

        lead_ids = subgraph_lead.to(dtype=torch.long)
        h = x + self.lead_embedding(lead_ids)

        attentions: list[tuple[Tensor, Tensor]] = []
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h_out, (attn_edge_index, alpha) = conv(
                h,
                super_edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
            attentions.append((attn_edge_index, alpha))
            h = norm(h + self.dropout(h_out))
        return h, attentions
