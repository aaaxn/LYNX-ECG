"""Tiny hierarchical-GNN graph + model builders for ``explain.*`` unit tests.

``ring_subgraph_inputs`` builds ``num_subgraphs`` independent ``k_nodes``-node
rings (a ring per subgraph): node features, a directed ring ``edge_index`` +
``edge_attr`` per subgraph, and the ``subgraph_node_batch`` membership vector.
It intentionally stops short of the exam-level fields (``super_edge_index``,
``subgraph_lead``, ...) so callers that need a non-default subgraph/exam
layout (e.g. a multi-exam batch) can lay those out themselves on top of it.

``single_exam_ring_inputs`` is the common case: one exam, 2 leads x 2 beats =
4 ring subgraphs, wired into a single connected supergraph.

``tiny_hier_model`` is the smallest ``HierGNNModel`` configuration that is
structurally valid for both builders above (5 classes, 1 GINE layer, 1 GATv2
layer with 2 heads).
"""

from __future__ import annotations

import torch

from models.hier_model import HierGNNModel


def ring_subgraph_inputs(num_subgraphs: int = 4, k_nodes: int = 4) -> dict:
    """Node features + intra-subgraph ring edges for ``num_subgraphs`` rings.

    Subgraph ``s`` owns nodes ``[s*k_nodes, (s+1)*k_nodes)``, connected as an
    undirected ring (stored as paired directed edges). Edge attributes are
    deterministic functions of the within-ring position so tests can pin
    exact values.
    """
    x = torch.randn(num_subgraphs * k_nodes, 2)
    edges, edge_attrs, node_batch = [], [], []
    for s in range(num_subgraphs):
        offset = s * k_nodes
        for i in range(k_nodes):
            edges.append([offset + i, offset + (i + 1) % k_nodes])
            edges.append([offset + (i + 1) % k_nodes, offset + i])
            edge_attrs.append([0.1 * (i + 1), 0.05 * (i + 1)])
            edge_attrs.append([0.1 * (i + 1), 0.05 * (i + 1)])
        node_batch.extend([s] * k_nodes)
    return {
        "x": x,
        "edge_index": torch.tensor(edges, dtype=torch.long).T,
        "edge_attr": torch.tensor(edge_attrs, dtype=torch.float32),
        "subgraph_node_batch": torch.tensor(node_batch, dtype=torch.long),
    }


def single_exam_ring_inputs(k_nodes: int = 4) -> dict:
    """One exam, 2 leads x 2 beats = 4 ring subgraphs of ``k_nodes`` nodes each."""
    base = ring_subgraph_inputs(num_subgraphs=4, k_nodes=k_nodes)
    base.update(
        super_edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
        super_edge_step=torch.tensor([1, -1, 1, -1]),
        subgraph_lead=torch.tensor([0, 0, 1, 1]),
        subgraph_exam=torch.tensor([0, 0, 0, 0]),
        subgraph_beat=torch.tensor([0, 1, 0, 1]),
        batch_size=1,
    )
    return base


def tiny_hier_model(num_leads: int = 2) -> HierGNNModel:
    """Smallest structurally-valid ``HierGNNModel``: 5 classes, hidden=16."""
    return HierGNNModel(
        num_classes=5,
        num_leads=num_leads,
        hidden_channels=16,
        beat_encoder_layers=1,
        beat_encoder_mlp_hidden=32,
        gatv2_heads=2,
        gatv2_layers=1,
    )
