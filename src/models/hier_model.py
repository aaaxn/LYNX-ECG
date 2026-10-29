"""Full hierarchical GNN model: subgraph → supergraph → attention pool → head.

Forward inputs are passed as plain tensors (no batched ``Data`` object) so the
model is decoupled from the dataset-collation layer.

Output: :class:`ModelOutput` containing logits and auxiliary outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as nn_func
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import softmax as group_softmax

from models.heads import ClassificationHead
from models.subgraph_encoder import SubgraphEncoder
from models.supergraph_encoder import SupergraphEncoder


@dataclass
class AttentionMaps:
    """Faithful, class-agnostic attention surfaced from one forward pass.

    These are the attention/gate weights the model *actually used* to pool its
    hierarchy for the given batch, not a post-hoc attribution. Per-class node
    importance (which turning points drive a *specific* diagnosis) is a separate
    concern handled by gradient attribution, because the GINE subgraph encoder
    has no attention and the pooling weights below do not depend on the class.

    Fields (``B`` exams, ``L`` leads, ``N`` subgraphs = beats across leads):

    - ``inter_lead`` ``[B, L]``: softmax weight per ``(exam, lead)`` from the
      inter-lead pool, naming *which leads* the exam embedding is built from. Weights
      sum to 1 over an exam's present leads; absent leads are 0. (An exam with no
      present leads, which cannot arise from a real batch, yields a zero row.)
    - ``lead_present`` ``[B, L]`` bool: which ``(exam, lead)`` slots received any
      subgraph (the mask the inter-lead softmax respects).
    - ``intra_lead_gate`` ``[N]``: attentional-pool gate weight per subgraph
      within its ``(exam, lead)`` group, naming *which beats* within a lead matter.
      Weights sum to 1 within each non-empty group.
    - ``supergraph`` per GATv2 layer ``(edge_index [2, E], alpha [E, heads])``:
      within-lead beat-to-beat attention.
    - ``subgraph_exam`` / ``subgraph_lead`` ``[N]``: the index tensors needed to
      map ``intra_lead_gate`` and ``supergraph`` node ids back to ``(exam, lead)``.
    """

    inter_lead: torch.Tensor
    lead_present: torch.Tensor
    intra_lead_gate: torch.Tensor
    supergraph: list[tuple[torch.Tensor, torch.Tensor]]
    subgraph_exam: torch.Tensor
    subgraph_lead: torch.Tensor


@dataclass
class ModelOutput:
    logits: torch.Tensor
    attention: AttentionMaps | None = None


def _gate_mlp(hidden_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_channels, max(hidden_channels // 2, 8)),
        nn.ReLU(),
        nn.Linear(max(hidden_channels // 2, 8), 1),
    )


class HierGNNModel(nn.Module):
    """End-to-end hierarchical model for multi-label ECG classification.

    The constructor arguments are captured verbatim in ``self.hparams`` so
    checkpoints are self-describing. ``training.checkpoint`` persists the
    dict and rebuilds the exact architecture without callers re-deriving it
    from tensor shapes.
    """

    def __init__(
        self,
        num_classes: int,
        num_leads: int = 12,
        in_channels: int = 2,
        edge_dim: int = 2,
        hidden_channels: int = 128,
        beat_encoder_layers: int = 3,
        beat_encoder_mlp_hidden: int = 256,
        dropout: float = 0.10,
        gatv2_heads: int = 4,
        gatv2_layers: int = 2,
        gatv2_max_edge_step: int = 1,
        head_hidden_multiplier: float = 0.5,
        head_dropout: float = 0.04,
    ) -> None:
        hparams = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        super().__init__()
        self.hparams: dict[str, Any] = hparams
        self.num_classes = num_classes
        self.num_leads = num_leads
        self.hidden_channels = hidden_channels

        self.subgraph_encoder = SubgraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            mlp_hidden=beat_encoder_mlp_hidden,
            num_layers=beat_encoder_layers,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.supergraph_encoder = SupergraphEncoder(
            hidden_channels=hidden_channels,
            num_heads=gatv2_heads,
            num_layers=gatv2_layers,
            max_edge_step=gatv2_max_edge_step,
            num_leads=num_leads,
            dropout=dropout,
        )

        self.intra_lead_pool = AttentionalAggregation(gate_nn=_gate_mlp(hidden_channels))
        assert self.intra_lead_pool.gate_nn is not None
        self.inter_lead_attn = nn.Sequential(
            nn.Linear(hidden_channels, max(hidden_channels // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_channels // 2, 8), 1),
        )
        self.head = ClassificationHead(
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            hidden_multiplier=head_hidden_multiplier,
            dropout=head_dropout,
        )

    def _intra_lead_pool(
        self,
        super_emb: torch.Tensor,
        subgraph_exam: torch.Tensor,
        subgraph_lead: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool subgraph embeddings into one vector per ``(exam, lead)``.

        Returns ``(lead_emb, lead_present, gate)`` where ``lead_emb`` is
        ``[batch_size, num_leads, hidden]``, ``lead_present`` is
        ``[batch_size, num_leads]`` (bool mask of which lead-slots received any
        subgraph; the others are zero rows that the inter-lead pool must mask
        out), and ``gate`` is the ``[N]`` per-subgraph attentional-pool weight.

        ``gate`` is re-derived with the *same* formula
        :class:`AttentionalAggregation` uses internally (``softmax(gate_nn(x))``
        grouped by ``(exam, lead)``). It is not exposed by the PyG module, so we
        recompute it here rather than reaching into the module's internals;
        ``test_intra_lead_gate_matches_attentional_aggregation`` locks this
        re-derivation against the module so it cannot drift. (The duplicate
        ``gate_nn`` evaluation is not worth manual dedup: measured A/B showed
        no win, because ``torch.compile`` CSE already merges the two calls.)
        """
        flat_index = subgraph_exam * self.num_leads + subgraph_lead
        dim_size = batch_size * self.num_leads
        lead_emb_flat = self.intra_lead_pool(super_emb, flat_index, dim_size=dim_size)
        lead_emb = lead_emb_flat.view(batch_size, self.num_leads, self.hidden_channels)

        lead_present = torch.zeros(
            (batch_size, self.num_leads), dtype=torch.bool, device=super_emb.device
        )
        lead_present.view(-1).index_fill_(0, flat_index, True)

        gate_logits = self.intra_lead_pool.gate_nn(super_emb)
        gate = group_softmax(gate_logits, flat_index, num_nodes=dim_size).squeeze(-1)
        return lead_emb, lead_present, gate

    def _inter_lead_pool(
        self,
        lead_emb: torch.Tensor,
        lead_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention pool over leads, masking out absent leads.

        Returns ``(pooled, attn_weights)``: the pooled ``[B, hidden]`` exam
        embedding, and the ``[B, L]`` per-lead softmax weights (0 on absent
        leads), naming *which leads* each exam embedding is built from.
        """
        attn_logits = self.inter_lead_attn(lead_emb)
        mask = lead_present.unsqueeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn_weights = nn_func.softmax(attn_logits, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        pooled = (attn_weights * lead_emb).sum(dim=1)
        return pooled, attn_weights.squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        subgraph_node_batch: torch.Tensor,
        super_edge_index: torch.Tensor,
        super_edge_step: torch.Tensor,
        subgraph_lead: torch.Tensor,
        subgraph_exam: torch.Tensor,
        batch_size: int,
        subgraph_beat: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> ModelOutput:
        """Run the full hierarchy and return :class:`ModelOutput`.

        ``return_attention`` is an opt-in explanation path: when set, the output
        carries an :class:`AttentionMaps` with the inter-lead, intra-lead, and
        supergraph attention the model used. It does not change the logits (in
        eval mode), and the flag is False on the training/eval hot path so it
        costs nothing there. Intended for inference-time explanation.
        """
        subgraph_emb = self.subgraph_encoder(
            x, edge_index, edge_attr, subgraph_node_batch, num_subgraphs=subgraph_lead.size(0)
        )

        super_edge_attr = (
            self.supergraph_encoder.embed_step(super_edge_step)
            if super_edge_index.numel() > 0
            else None
        )
        super_emb, supergraph_attn = self.supergraph_encoder(
            subgraph_emb,
            super_edge_index,
            super_edge_attr,
            subgraph_lead,
        )

        lead_emb, lead_present, intra_gate = self._intra_lead_pool(
            super_emb, subgraph_exam, subgraph_lead, batch_size
        )

        exam_emb, inter_weights = self._inter_lead_pool(lead_emb, lead_present)

        logits = self.head(exam_emb)

        attention: AttentionMaps | None = None
        if return_attention:
            attention = AttentionMaps(
                inter_lead=inter_weights,
                lead_present=lead_present,
                intra_lead_gate=intra_gate,
                supergraph=supergraph_attn,
                subgraph_exam=subgraph_exam,
                subgraph_lead=subgraph_lead,
            )

        return ModelOutput(
            logits=logits,
            attention=attention,
        )
