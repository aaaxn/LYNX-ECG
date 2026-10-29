"""Shared test fixtures for unit tests that exercise the HierData batch protocol.

``MinimalBatch`` mirrors the fields that :func:`training.inference.forward_batch`
reads from a ``HierData`` batch (plus ``y`` and ``num_graphs``).  Any change to
that protocol, whether adding a tensor field or renaming a slot, must be reflected here
and in ``forward_batch`` simultaneously, which makes the coupling visible and
intentional.

Fields mirrored from ``HierData`` / ``forward_batch``:
    x, edge_index, edge_attr:          subgraph node features and edges
    subgraph_node_batch:                subgraph membership per node
    super_edge_index, super_edge_step: synthesised supergraph edges
    subgraph_lead, subgraph_exam:      per-subgraph lead / exam indices
    subgraph_beat:                     per-subgraph beat index
    y:                                 multi-hot float label vector
    num_graphs:                        scalar exam count (PyG batch protocol)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.hier_model import ModelOutput


@dataclass
class MinimalBatch:
    """Minimal stand-in for a collated ``HierData`` batch.

    Initialised with a single exam (``num_graphs=1``) and one node, zero edges.
    Only ``y`` is provided at construction; all tensor fields are synthesised in
    ``__post_init__`` so tests stay readable.

    Mirrors the fields consumed by :func:`training.inference.forward_batch`:
    ``x``, ``edge_index``, ``edge_attr``, ``subgraph_node_batch``,
    ``super_edge_index``, ``super_edge_step``, ``subgraph_lead``,
    ``subgraph_exam``, ``subgraph_beat``, and the scalar ``num_graphs``.
    """

    y: torch.Tensor

    def __post_init__(self) -> None:
        self.x = torch.zeros(1, 2)
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.empty((0, 2))
        self.subgraph_node_batch = torch.zeros(1, dtype=torch.long)
        self.super_edge_index = torch.empty((2, 0), dtype=torch.long)
        self.super_edge_step = torch.empty(0, dtype=torch.long)
        self.subgraph_lead = torch.zeros(1, dtype=torch.long)
        self.subgraph_exam = torch.zeros(1, dtype=torch.long)
        self.subgraph_beat = torch.zeros(1, dtype=torch.long)
        self.num_graphs = 1

    def to(self, device: torch.device, non_blocking: bool = False) -> MinimalBatch:
        del non_blocking
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self


class MinimalModel(nn.Module):
    """Minimal stub model for unit tests.

    Returns all-zero logits (shape ``[batch_size, 2]``) for every batch.
    Logits are intentionally zero rather than a non-trivial constant (the
    original ``FakeModel`` in ``test_inference`` returned ``0.5``).  Zero
    logits are sufficient for every call site: no test asserts on a specific
    logit value, and BCE with mixed labels (e.g. ``[1, 0]`` vs ``[0, 1]``)
    still produces a positive average loss, so assertions such as
    ``avg_loss > 0.0`` remain valid.

    A ``_dummy`` parameter with a zero gradient is included so that
    ``torch.optim.AdamW(model.parameters())`` receives a non-empty parameter
    list, required by tests that drive the full ``training.train.train()`` loop.

    The ``+ self._dummy * 0`` term keeps ``_dummy`` in the computation graph,
    satisfying ``loss.backward()`` without altering any output value.
    """

    def __init__(self) -> None:
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs) -> ModelOutput:
        batch_size = int(kwargs["batch_size"])
        logits = torch.zeros(batch_size, 2) + self._dummy * 0
        return ModelOutput(
            logits=logits,
        )
