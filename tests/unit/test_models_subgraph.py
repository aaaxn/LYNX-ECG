"""Unit tests for SubgraphEncoder."""

from __future__ import annotations

import pytest
import torch

from models.subgraph_encoder import SubgraphEncoder

pytestmark = pytest.mark.unit


def _two_subgraphs():
    """Two tiny subgraphs (5 nodes each, ring topology) batched together."""
    n = 5
    x = torch.randn(2 * n, 2)
    e0 = torch.tensor([[i, (i + 1) % n] for i in range(n)] + [[(i + 1) % n, i] for i in range(n)]).T
    e1 = e0 + n
    edge_index = torch.cat([e0, e1], dim=1).long()
    edge_attr = torch.randn(edge_index.size(1), 2)
    batch = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    return x, edge_index, edge_attr, batch


def test_subgraph_encoder_forward_shape():
    enc = SubgraphEncoder(
        in_channels=2, hidden_channels=32, mlp_hidden=64, num_layers=2, edge_dim=2
    )
    x, edge_index, edge_attr, batch = _two_subgraphs()
    out = enc(x, edge_index, edge_attr, batch, num_subgraphs=2)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_subgraph_encoder_num_subgraphs_fixes_output_rows():
    """The pool must size its output from num_subgraphs, not batch.max()+1.

    Inferring from the batch vector is a GPU sync and would silently drop
    rows for trailing subgraphs; an explicit count keeps row i aligned with
    subgraph i regardless of the batch vector's max.
    """
    enc = SubgraphEncoder(
        in_channels=2, hidden_channels=32, mlp_hidden=64, num_layers=2, edge_dim=2
    )
    x, edge_index, edge_attr, batch = _two_subgraphs()
    out = enc(x, edge_index, edge_attr, batch, num_subgraphs=3)
    assert out.shape == (3, 32)
    assert torch.all(out[2] == 0)


def test_subgraph_encoder_invalid_layers():
    with pytest.raises(ValueError, match="num_layers"):
        SubgraphEncoder(num_layers=0)


def test_subgraph_encoder_backward_updates_params():
    torch.manual_seed(0)
    enc = SubgraphEncoder(
        in_channels=2, hidden_channels=16, mlp_hidden=32, num_layers=1, edge_dim=2
    )
    x, edge_index, edge_attr, batch = _two_subgraphs()
    out = enc(x, edge_index, edge_attr, batch, num_subgraphs=2)
    loss = out.pow(2).sum()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
