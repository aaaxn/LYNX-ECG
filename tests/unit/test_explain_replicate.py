"""Unit tests for the replicate-exam primitive (explain.replicate).

``replicate_inputs`` tiles a single-exam forward-kwargs dict as ``n``
structurally disjoint batch copies. These tests pin the offset arithmetic
(shapes + exact values) and the independence guarantee that batched Integrated
Gradients relies on: copy ``i``'s logits must equal what a
standalone forward on copy ``i`` alone would produce, regardless of what the
other copies' ``x`` rows contain.
"""

from __future__ import annotations

import pytest
import torch

from explain.replicate import replicate_inputs
from tests.fixtures.hier_graph import (
    single_exam_ring_inputs as _single_exam_inputs,
)
from tests.fixtures.hier_graph import (
    tiny_hier_model as _model,
)

pytestmark = pytest.mark.unit


def _two_subgraph_inputs() -> dict:
    """One exam, 2 subgraphs, 5 nodes, 4 directed edges, 1 super-edge.

    Node layout: subgraph 0 = nodes {0, 1, 2}, subgraph 1 = nodes {3, 4}.
    Small enough that the offset arithmetic in the shape test can be checked
    by hand.
    """
    x = torch.randn(5, 2)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=torch.long)
    edge_attr = torch.randn(4, 3)
    subgraph_node_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    super_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    super_edge_step = torch.tensor([1], dtype=torch.long)
    subgraph_lead = torch.tensor([0, 1], dtype=torch.long)
    subgraph_exam = torch.tensor([0, 0], dtype=torch.long)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "subgraph_node_batch": subgraph_node_batch,
        "super_edge_index": super_edge_index,
        "super_edge_step": super_edge_step,
        "subgraph_lead": subgraph_lead,
        "subgraph_exam": subgraph_exam,
        "batch_size": 1,
    }


def test_replicate_shapes_and_offsets():
    inputs = _two_subgraph_inputs()
    n = 3

    rep = replicate_inputs(inputs, n)

    assert rep["x"].shape == (15, 2)
    assert rep["edge_index"].shape == (2, 12)
    copy0 = inputs["edge_index"]
    copy1 = rep["edge_index"][:, 4:8]
    assert torch.equal(copy1, copy0 + 5)
    copy2 = rep["edge_index"][:, 8:12]
    assert torch.equal(copy2, copy0 + 10)

    assert int(rep["subgraph_node_batch"].max()) == 5
    assert rep["edge_attr"].shape == (12, 3)
    assert rep["super_edge_index"].shape == (2, 3)
    assert torch.equal(rep["super_edge_step"], inputs["super_edge_step"].repeat(3))
    assert torch.equal(rep["subgraph_lead"], inputs["subgraph_lead"].repeat(3))

    assert torch.equal(rep["subgraph_exam"], torch.tensor([0, 0, 1, 1, 2, 2]))
    assert rep["batch_size"] == 3


def test_replicated_copies_are_independent():
    torch.manual_seed(0)
    model = _model()
    model.eval()
    inputs = _single_exam_inputs()
    n_nodes = inputs["x"].shape[0]

    with torch.no_grad():
        baseline_logits = model(**inputs).logits[0]

    rep = replicate_inputs(inputs, 3)
    x_rep = rep["x"].clone()
    x_rep[n_nodes : 2 * n_nodes] = 0.0
    rep["x"] = x_rep

    with torch.no_grad():
        rep_logits = model(**rep).logits

    assert torch.allclose(rep_logits[0], baseline_logits, atol=1e-5)
    assert torch.allclose(rep_logits[2], baseline_logits, atol=1e-5)
    assert not torch.allclose(rep_logits[1], baseline_logits, atol=1e-5)


def test_subgraph_beat_present_is_tiled():
    inputs = _single_exam_inputs()
    rep = replicate_inputs(inputs, 3)
    assert torch.equal(rep["subgraph_beat"], inputs["subgraph_beat"].repeat(3))


def test_subgraph_beat_absent_stays_absent():
    inputs = _two_subgraph_inputs()
    rep = replicate_inputs(inputs, 3)
    assert "subgraph_beat" not in rep


def test_subgraph_beat_none_stays_none():
    inputs = _two_subgraph_inputs()
    inputs["subgraph_beat"] = None
    rep = replicate_inputs(inputs, 3)
    assert rep["subgraph_beat"] is None


def test_rejects_unknown_keys():
    inputs = _two_subgraph_inputs()
    inputs["bogus"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected keys"):
        replicate_inputs(inputs, 2)
