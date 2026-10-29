"""Unit tests for HierGNNModel, end-to-end forward + backward with a tiny batch."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import softmax as group_softmax

from models.hier_model import AttentionMaps, HierGNNModel, ModelOutput

pytestmark = pytest.mark.unit


def _hand_batch(num_classes: int = 5, num_leads: int = 3):
    """Build a tiny synthetic batch by hand:

    - 2 exams (batch_size=2)
    - exam 0: 2 leads with 2 beats each → 4 subgraphs
    - exam 1: 1 lead with 1 beat → 1 subgraph (intentionally minimal)
    Total 5 subgraphs, k_nodes=4 nodes per subgraph.
    """
    k_nodes = 4
    x = torch.randn(5 * k_nodes, 2)
    edges = []
    edge_attrs = []
    node_batch = []
    for s in range(5):
        offset = s * k_nodes
        ring_fwd = [[offset + i, offset + (i + 1) % k_nodes] for i in range(k_nodes)]
        ring_bwd = [[offset + (i + 1) % k_nodes, offset + i] for i in range(k_nodes)]
        edges.extend(ring_fwd + ring_bwd)
        edge_attrs.extend([[0.1 * (i + 1), 0.05 * (i + 1)] for i in range(2 * k_nodes)])
        node_batch.extend([s] * k_nodes)
    edge_index = torch.tensor(edges, dtype=torch.long).T
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    subgraph_node_batch = torch.tensor(node_batch, dtype=torch.long)

    super_edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    super_edge_step = torch.tensor([1, -1, 1, -1])
    subgraph_lead = torch.tensor([0, 0, 1, 1, 0])
    subgraph_exam = torch.tensor([0, 0, 0, 0, 1])
    batch_size = 2

    subgraph_beat = torch.tensor([0, 1, 0, 1, 0])

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "subgraph_node_batch": subgraph_node_batch,
        "super_edge_index": super_edge_index,
        "super_edge_step": super_edge_step,
        "subgraph_lead": subgraph_lead,
        "subgraph_exam": subgraph_exam,
        "subgraph_beat": subgraph_beat,
        "batch_size": batch_size,
    }


def test_hier_model_forward_shape():
    torch.manual_seed(0)
    model = HierGNNModel(
        num_classes=5,
        num_leads=3,
        in_channels=2,
        edge_dim=2,
        hidden_channels=16,
        beat_encoder_layers=1,
        beat_encoder_mlp_hidden=32,
        gatv2_heads=2,
        gatv2_layers=1,
        gatv2_max_edge_step=1,
    )
    batch = _hand_batch(num_classes=5, num_leads=3)
    out = model(**batch)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (2, 5)
    assert torch.isfinite(out.logits).all()


def test_hier_model_backward_updates_params():
    torch.manual_seed(0)
    model = HierGNNModel(
        num_classes=5,
        num_leads=3,
        hidden_channels=16,
        beat_encoder_layers=1,
        beat_encoder_mlp_hidden=32,
        gatv2_heads=2,
        gatv2_layers=1,
    )
    batch = _hand_batch(num_classes=5, num_leads=3)
    target = torch.zeros(2, 5)
    target[0, 0] = 1.0
    target[1, 2] = 1.0
    out = model(**batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.logits, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_hier_model_lead_mask_handles_absent_leads():
    """Exam 1 only has lead 0. The inter-lead pool must not mix in absent leads."""
    torch.manual_seed(0)
    model = HierGNNModel(
        num_classes=3,
        num_leads=4,
        hidden_channels=16,
        beat_encoder_layers=1,
        beat_encoder_mlp_hidden=16,
        gatv2_heads=2,
        gatv2_layers=1,
    )
    batch = _hand_batch(num_classes=3, num_leads=4)
    out = model(**batch)
    assert out.logits.shape == (2, 3)
    assert torch.isfinite(out.logits).all()


def test_inter_lead_pool_ignores_absent_lead_rows():
    """Perturbing the rows of absent leads must not change the exam embedding."""
    torch.manual_seed(0)
    model = HierGNNModel(num_classes=3, num_leads=4, hidden_channels=16, beat_encoder_layers=1)
    model.eval()

    batch_size, num_leads, hidden = 2, 4, 16
    lead_emb = torch.randn(batch_size, num_leads, hidden)
    lead_present = torch.tensor(
        [[True, True, True, True], [True, False, False, False]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        out_a, _weights_a = model._inter_lead_pool(lead_emb, lead_present)

    lead_emb_perturbed = lead_emb.clone()
    lead_emb_perturbed[1, 1:, :] += 100.0
    with torch.no_grad():
        out_b, _weights_b = model._inter_lead_pool(lead_emb_perturbed, lead_present)

    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_intra_lead_pool_presence_mask_handles_duplicate_subgraphs():
    """Multiple subgraphs in the same (exam, lead) slot must not corrupt the mask."""
    torch.manual_seed(0)
    model = HierGNNModel(num_classes=3, num_leads=4, hidden_channels=16, beat_encoder_layers=1)
    model.eval()

    batch_size, hidden = 2, 16
    subgraph_exam = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)
    subgraph_lead = torch.tensor([0, 0, 0, 2, 1], dtype=torch.long)
    super_emb = torch.randn(subgraph_exam.numel(), hidden)

    _lead_emb, lead_present, _gate = model._intra_lead_pool(
        super_emb, subgraph_exam, subgraph_lead, batch_size
    )

    expected = torch.tensor(
        [[True, False, True, False], [False, True, False, False]],
        dtype=torch.bool,
    )
    assert torch.equal(lead_present, expected)


def test_intra_lead_gate_matches_attentional_aggregation():
    """The re-derived intra-lead gate must reproduce PyG's AttentionalAggregation.

    ``_intra_lead_pool`` recomputes the pool's gate weights (PyG does not expose
    them) as ``softmax(gate_nn(x))`` grouped by ``(exam, lead)``. This test locks
    that formula against the module itself: pooling ``gate * x`` by group must
    equal the module's output. If a PyG version changes the gate formula, this
    fails instead of silently returning wrong attention.
    """
    torch.manual_seed(0)
    hidden = 16
    gate_nn = nn.Sequential(nn.Linear(hidden, 8), nn.ReLU(), nn.Linear(8, 1))
    pool = AttentionalAggregation(gate_nn=gate_nn)
    pool.eval()

    x = torch.randn(7, hidden)
    index = torch.tensor([0, 0, 1, 1, 1, 3, 3])
    dim_size = 4

    with torch.no_grad():
        module_out = pool(x, index, dim_size=dim_size)
        gate = group_softmax(gate_nn(x), index, num_nodes=dim_size)
        recon = torch.zeros(dim_size, hidden).index_add_(0, index, gate * x)

    assert module_out.shape == (dim_size, hidden)
    assert torch.allclose(module_out, recon, atol=1e-6)
    for g in (0, 1, 3):
        assert torch.allclose(gate[index == g].sum(), torch.tensor(1.0), atol=1e-6)


def test_hier_model_return_attention_invariance_and_shapes():
    """return_attention surfaces faithful maps WITHOUT changing the logits."""
    torch.manual_seed(0)
    num_classes, num_leads = 5, 3
    model = HierGNNModel(
        num_classes=num_classes,
        num_leads=num_leads,
        hidden_channels=16,
        beat_encoder_layers=1,
        beat_encoder_mlp_hidden=32,
        gatv2_heads=2,
        gatv2_layers=2,
    )
    model.eval()
    batch = _hand_batch(num_classes=num_classes, num_leads=num_leads)

    with torch.no_grad():
        plain = model(**batch)
        explained = model(**batch, return_attention=True)

    assert plain.attention is None
    assert torch.allclose(plain.logits, explained.logits, atol=1e-6)

    attn = explained.attention
    assert isinstance(attn, AttentionMaps)

    n_subgraphs = batch["subgraph_exam"].numel()
    batch_size = batch["batch_size"]

    assert attn.inter_lead.shape == (batch_size, num_leads)
    assert attn.lead_present.shape == (batch_size, num_leads)
    assert torch.all(attn.inter_lead[~attn.lead_present] == 0.0)
    present_mass = attn.inter_lead.sum(dim=1)
    assert torch.allclose(present_mass, torch.ones(batch_size), atol=1e-6)

    assert attn.intra_lead_gate.shape == (n_subgraphs,)
    flat_index = batch["subgraph_exam"] * num_leads + batch["subgraph_lead"]
    for grp in flat_index.unique():
        mass = attn.intra_lead_gate[flat_index == grp].sum()
        assert torch.allclose(mass, torch.tensor(1.0), atol=1e-6)

    assert len(attn.supergraph) == 2
    n_super_edges = batch["super_edge_index"].size(1)
    for edge_index, alpha in attn.supergraph:
        assert edge_index.shape == (2, n_super_edges)
        assert alpha.shape == (n_super_edges, 2)
        assert torch.isfinite(alpha).all()

    assert torch.equal(attn.subgraph_exam, batch["subgraph_exam"])
    assert torch.equal(attn.subgraph_lead, batch["subgraph_lead"])
