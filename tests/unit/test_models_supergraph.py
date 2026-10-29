"""Unit tests for SupergraphEncoder."""

from __future__ import annotations

import pytest
import torch

from models.supergraph_encoder import SupergraphEncoder

pytestmark = pytest.mark.unit


def test_bucketize_signed_in_range():
    enc = SupergraphEncoder(hidden_channels=16, num_heads=4, max_edge_step=2)
    steps = torch.tensor([-2, -1, 0, 1, 2])
    buckets = enc._bucketize(steps)
    assert buckets.tolist() == [1, 2, 3, 4, 5]


def test_bucketize_signed_under_overflow():
    enc = SupergraphEncoder(hidden_channels=16, num_heads=4, max_edge_step=1)
    steps = torch.tensor([-5, -2, -1, 0, 1, 2, 5])
    assert enc._bucketize(steps).tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_supergraph_encoder_forward_shape():
    enc = SupergraphEncoder(
        hidden_channels=32, num_heads=4, num_layers=2, max_edge_step=1, num_leads=3
    )
    n = 6
    x = torch.randn(n, 32)
    super_edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
    super_edge_step = torch.tensor([1, -1, 1, -1, 1, -1])
    subgraph_lead = torch.tensor([0, 0, 1, 1, 2, 2])
    edge_attr = enc.embed_step(super_edge_step)
    out, attns = enc(x, super_edge_index, edge_attr, subgraph_lead)
    assert out.shape == (n, 32)
    assert torch.isfinite(out).all()
    assert len(attns) == 2


def test_supergraph_encoder_no_edges():
    """Empty supergraph (e.g., single-beat lead) still returns a valid forward."""
    enc = SupergraphEncoder(hidden_channels=16, num_heads=4, num_layers=1, max_edge_step=1)
    x = torch.randn(3, 16)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    leads = torch.tensor([0, 1, 2])
    out, attns = enc(x, edge_index, None, leads)
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()
    assert len(attns) == 1
    attn_edge_index, alpha = attns[0]
    assert attn_edge_index.shape == (2, 0)
    assert alpha.shape == (0, 4)


def test_supergraph_encoder_invalid_heads_div():
    with pytest.raises(ValueError, match="divisible"):
        SupergraphEncoder(hidden_channels=33, num_heads=4)


def test_supergraph_encoder_returns_attention_with_output():
    """forward yields per-layer (edge_index, alpha) alongside the node embeddings h."""
    torch.manual_seed(0)
    hidden, heads, layers = 32, 4, 2
    enc = SupergraphEncoder(
        hidden_channels=hidden, num_heads=heads, num_layers=layers, max_edge_step=1, num_leads=3
    )
    enc.eval()

    n = 6
    x = torch.randn(n, hidden)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=torch.long)
    edge_step = torch.tensor([1, -1, 1, -1, 1, -1])
    leads = torch.tensor([0, 0, 1, 1, 2, 2])
    edge_attr = enc.embed_step(edge_step)

    with torch.no_grad():
        h, attns = enc(x, edge_index, edge_attr, leads)

    assert h.shape == (n, hidden)
    assert torch.isfinite(h).all()
    assert len(attns) == layers
    for attn_edge_index, alpha in attns:
        assert attn_edge_index.shape == (2, edge_index.size(1))
        assert alpha.shape == (edge_index.size(1), heads)
        assert torch.isfinite(alpha).all()


def test_supergraph_encoder_attention_empty_nodes():
    """Empty supergraph (no nodes) returns empty attention list, not a crash."""
    enc = SupergraphEncoder(hidden_channels=16, num_heads=4, num_layers=2)
    enc.eval()
    x = torch.empty((0, 16))
    edge_index = torch.empty((2, 0), dtype=torch.long)
    leads = torch.empty((0,), dtype=torch.long)

    out, attns = enc(x, edge_index, None, leads)
    assert out.shape == (0, 16)
    assert attns == []
