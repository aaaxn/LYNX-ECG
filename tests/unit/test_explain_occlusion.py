"""Unit tests for structural deletion (explain.occlusion).

Structural deletion removes a node's or subgraph's *presence* regardless of its
feature magnitude, and its incident edges go with it. These tests cover
node-level deletion with gap bridging (delete_nodes), subgraph-level deletion
(delete_subgraphs), their composition (delete_nodes_with_cascade), the
empty-subgraph and empty-exam guards, and the target-score helper.
"""

from __future__ import annotations

import pytest
import torch

from explain.occlusion import (
    EmptyExamError,
    EmptySubgraphError,
    delete_nodes,
    delete_nodes_with_cascade,
    delete_subgraphs,
    target_scores,
)
from tests.fixtures.hier_graph import (
    single_exam_ring_inputs as _single_exam_inputs,
)
from tests.fixtures.hier_graph import (
    tiny_hier_model as _model,
)

pytestmark = pytest.mark.unit

K_NODES = 4


def _path_graph_inputs() -> dict:
    """Deterministic 4-node path 0-1-2-3 (undirected), all in one subgraph.

    Column 0 of ``x`` is the node id, so relabelling is checkable by value.
    """
    x = torch.tensor([[0.0, 0.0], [1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
    und = [(0, 1), (1, 2), (2, 3)]
    edges, edge_attrs = [], []
    for a, b in und:
        edges += [[a, b], [b, a]]
        edge_attrs += [[float(a), float(b)], [float(b), float(a)]]
    return {
        "x": x,
        "edge_index": torch.tensor(edges, dtype=torch.long).T,
        "edge_attr": torch.tensor(edge_attrs, dtype=torch.float32),
        "subgraph_node_batch": torch.zeros(4, dtype=torch.long),
        "super_edge_index": torch.empty((2, 0), dtype=torch.long),
        "super_edge_step": torch.empty(0, dtype=torch.long),
        "subgraph_lead": torch.zeros(1, dtype=torch.long),
        "subgraph_exam": torch.zeros(1, dtype=torch.long),
        "subgraph_beat": torch.zeros(1, dtype=torch.long),
        "batch_size": 1,
    }


def test_delete_relabels_nodes_drops_incident_edges_and_bridges_the_gap():
    model = _model()
    inputs = _path_graph_inputs()
    remove = torch.tensor([False, True, False, False])

    out = delete_nodes(model, inputs, remove)

    assert torch.equal(out["x"][:, 0], torch.tensor([0.0, 2.0, 3.0]))
    assert torch.equal(out["subgraph_node_batch"], torch.zeros(3, dtype=torch.long))
    edges = {
        tuple(c): tuple(a)
        for c, a in zip(out["edge_index"].T.tolist(), out["edge_attr"].tolist(), strict=True)
    }
    assert edges == {
        (1, 2): (2.0, 3.0),
        (2, 1): (3.0, 2.0),
        (0, 1): (1.0, 3.0),
        (1, 0): (3.0, 1.0),
    }
    assert int(out["edge_index"].max()) < out["x"].shape[0]


def _forward_beats_inputs(extra_edges: list[tuple[int, int]] | None = None) -> dict:
    """Two forward-only beat graphs of 5 and 3 nodes with derived [dt, dy] edges.

    Each beat has the consecutive edges i -> i+1 that a natural visibility graph
    always contains, plus ``extra_edges`` (global node ids).
    """
    t = torch.tensor([0.0, 0.1, 0.25, 0.3, 0.5, 0.0, 0.2, 0.4])
    y = torch.tensor([0.0, 1.0, -0.5, 0.2, 0.1, 0.3, 0.9, 0.0])
    sgnb = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), *(extra_edges or [])]
    edge_index = torch.tensor(pairs, dtype=torch.long).T
    src, dst = edge_index
    return {
        "x": torch.stack([y, t], dim=1),
        "edge_index": edge_index,
        "edge_attr": torch.stack([t[dst] - t[src], y[dst] - y[src]], dim=1),
        "subgraph_node_batch": sgnb,
        "super_edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "super_edge_step": torch.tensor([1, -1]),
        "subgraph_lead": torch.zeros(2, dtype=torch.long),
        "subgraph_exam": torch.zeros(2, dtype=torch.long),
        "subgraph_beat": torch.tensor([0, 1]),
        "batch_size": 1,
    }


def test_bridge_gets_the_endpoint_differences_of_the_removed_run():
    inputs = _forward_beats_inputs()
    t, y = inputs["x"][:, 1], inputs["x"][:, 0]
    remove = torch.zeros(8, dtype=torch.bool)
    remove[[1, 2]] = True

    out = delete_nodes(_model(), inputs, remove)

    edges = {
        tuple(c): a
        for c, a in zip(out["edge_index"].T.tolist(), out["edge_attr"].tolist(), strict=True)
    }
    assert (1, 0) not in edges
    assert edges[(0, 1)] == pytest.approx([float(t[3] - t[0]), float(y[3] - y[0])])
    assert len(edges) == 3 + 1


def test_bridge_is_not_duplicated_and_runs_at_beat_ends_are_not_bridged():
    inputs = _forward_beats_inputs(extra_edges=[(0, 3)])
    remove = torch.zeros(8, dtype=torch.bool)
    remove[[1, 2]] = True
    remove[4] = True
    remove[5] = True

    out = delete_nodes(_model(), inputs, remove)

    assert sorted(map(tuple, out["edge_index"].T.tolist())) == [(0, 1), (2, 3)]


def test_bridge_requires_the_consecutive_edges_of_the_run():
    inputs = _forward_beats_inputs()
    keep_edge = ~((inputs["edge_index"][0] == 1) & (inputs["edge_index"][1] == 2))
    inputs["edge_index"] = inputs["edge_index"][:, keep_edge]
    inputs["edge_attr"] = inputs["edge_attr"][keep_edge]
    remove = torch.zeros(8, dtype=torch.bool)
    remove[[1, 2]] = True

    with pytest.raises(ValueError, match="consecutive edge"):
        delete_nodes(_model(), inputs, remove)


def test_delete_does_not_mutate_inputs():
    model = _model()
    inputs = _path_graph_inputs()
    x_before = inputs["x"].clone()
    e_before = inputs["edge_index"].clone()
    delete_nodes(model, inputs, torch.tensor([True, False, False, False]))
    assert torch.equal(inputs["x"], x_before)
    assert torch.equal(inputs["edge_index"], e_before)


def test_delete_empty_mask_is_identity_for_prediction():
    torch.manual_seed(0)
    model = _model()
    model.eval()
    inputs = _single_exam_inputs()
    n = inputs["x"].shape[0]

    out = delete_nodes(model, inputs, torch.zeros(n, dtype=torch.bool))
    assert out["x"].shape == inputs["x"].shape
    with torch.no_grad():
        p_full = torch.sigmoid(model(**inputs).logits[0])
        p_keep = torch.sigmoid(model(**out).logits[0])
    assert torch.allclose(p_full, p_keep, atol=1e-6)


def test_emptying_a_subgraph_raises():
    model = _model()
    inputs = _single_exam_inputs()
    remove = torch.zeros(inputs["x"].shape[0], dtype=torch.bool)
    remove[:K_NODES] = True
    with pytest.raises(EmptySubgraphError, match="empty 1 of 4"):
        delete_nodes(model, inputs, remove)


def test_rejects_wrong_mask_shape():
    model = _model()
    inputs = _single_exam_inputs()
    with pytest.raises(ValueError, match="remove_mask must have shape"):
        delete_nodes(model, inputs, torch.zeros(3, dtype=torch.bool))


def test_target_scores_reads_multiple_classes_from_one_forward():
    torch.manual_seed(0)
    model = _model()
    model.eval()
    inputs = _single_exam_inputs()

    scores = target_scores(model, inputs, target_classes=(0, 2))
    with torch.no_grad():
        logits = model(**inputs).logits[0]

    assert list(scores) == [0, 2]
    assert scores[0].logit == pytest.approx(float(logits[0]))
    assert scores[2].probability == pytest.approx(float(torch.sigmoid(logits[2])))

    with pytest.raises(ValueError, match="non-empty and unique"):
        target_scores(model, inputs, target_classes=())
    with pytest.raises(ValueError, match="non-empty and unique"):
        target_scores(model, inputs, target_classes=(0, 0))
    with pytest.raises(ValueError, match="must lie"):
        target_scores(model, inputs, target_classes=(model.num_classes,))


def _three_subgraph_path_inputs() -> dict:
    """3 subgraphs of 2 nodes each; supergraph path 0-1-2 (subgraph 1 in the middle).

    Column 0 of ``x`` is the node id, so relabelling is checkable by value.
    Deleting the middle subgraph exercises dropping *both* its incident
    supergraph edges, not just one.
    """
    x = torch.tensor([[0.0, 0.0], [1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4], [5.0, 0.5]])
    intra_pairs = [(0, 1), (2, 3), (4, 5)]
    edges, edge_attrs = [], []
    for a, b in intra_pairs:
        edges += [[a, b], [b, a]]
        edge_attrs += [[float(a), float(b)], [float(b), float(a)]]
    return {
        "x": x,
        "edge_index": torch.tensor(edges, dtype=torch.long).T,
        "edge_attr": torch.tensor(edge_attrs, dtype=torch.float32),
        "subgraph_node_batch": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        "super_edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        "super_edge_step": torch.tensor([1, -1, 1, -1]),
        "subgraph_lead": torch.tensor([0, 0, 1], dtype=torch.long),
        "subgraph_exam": torch.tensor([0, 0, 0], dtype=torch.long),
        "subgraph_beat": torch.tensor([0, 1, 0], dtype=torch.long),
        "batch_size": 1,
    }


def test_delete_subgraphs_relabels_and_drops_incident_supergraph_edges():
    model = _model()
    inputs = _three_subgraph_path_inputs()
    remove = torch.tensor([False, True, False])

    out = delete_subgraphs(model, inputs, remove)

    assert torch.equal(out["x"][:, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))
    assert torch.equal(out["subgraph_node_batch"], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(out["subgraph_lead"], torch.tensor([0, 1]))
    assert torch.equal(out["subgraph_exam"], torch.tensor([0, 0]))
    assert torch.equal(out["subgraph_beat"], torch.tensor([0, 0]))
    surviving = {tuple(c) for c in out["edge_index"].T.tolist()}
    assert surviving == {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert out["super_edge_index"].shape == (2, 0)
    assert out["super_edge_step"].shape == (0,)


def test_delete_subgraphs_empty_mask_is_identity_for_prediction():
    torch.manual_seed(0)
    model = _model()
    model.eval()
    inputs = _single_exam_inputs()
    n_sub = inputs["subgraph_lead"].shape[0]

    out = delete_subgraphs(model, inputs, torch.zeros(n_sub, dtype=torch.bool))
    with torch.no_grad():
        p_full = torch.sigmoid(model(**inputs).logits[0])
        p_keep = torch.sigmoid(model(**out).logits[0])
    assert torch.allclose(p_full, p_keep, atol=1e-6)


def test_emptying_an_exam_of_all_subgraphs_raises():
    model = _model()
    inputs = _single_exam_inputs()
    n_sub = inputs["subgraph_lead"].shape[0]
    with pytest.raises(EmptyExamError, match="empty 1 of 1 exam"):
        delete_subgraphs(model, inputs, torch.ones(n_sub, dtype=torch.bool))


def test_delete_nodes_with_cascade_removes_empty_subgraphs_and_partial_nodes():
    model = _model()
    inputs = _three_subgraph_path_inputs()
    remove = torch.tensor([False, False, True, True, True, False])

    result = delete_nodes_with_cascade(model, inputs, remove)
    out = result.inputs

    assert result.removed_nodes == 3
    assert result.removed_subgraphs == 1
    assert torch.equal(result.removed_subgraph_ids, torch.tensor([1]))
    assert torch.equal(out["x"][:, 0], torch.tensor([0.0, 1.0, 5.0]))
    assert torch.equal(out["subgraph_node_batch"], torch.tensor([0, 0, 1]))
    assert torch.equal(out["subgraph_lead"], torch.tensor([0, 1]))
    assert torch.equal(out["subgraph_beat"], torch.tensor([0, 0]))
    assert out["super_edge_index"].shape == (2, 0)


def test_delete_nodes_with_cascade_reports_no_removed_subgraphs_for_partial_deletion():
    model = _model()
    inputs = _three_subgraph_path_inputs()
    remove = torch.tensor([True, False, False, False, False, False])

    result = delete_nodes_with_cascade(model, inputs, remove)

    assert result.removed_nodes == 1
    assert result.removed_subgraphs == 0
    assert result.removed_subgraph_ids.numel() == 0
    assert result.inputs["subgraph_lead"].shape[0] == 3


def test_delete_nodes_with_cascade_rejects_fully_empty_exam():
    model = _model()
    inputs = _three_subgraph_path_inputs()

    with pytest.raises(EmptyExamError, match="empty 1 of 1 exam"):
        delete_nodes_with_cascade(
            model,
            inputs,
            torch.ones(inputs["x"].shape[0], dtype=torch.bool),
        )


def test_delete_nodes_with_cascade_rejects_empty_exam_in_multi_exam_batch():
    model = _model()
    inputs = _three_subgraph_path_inputs()
    inputs["subgraph_exam"] = torch.tensor([0, 0, 1])
    inputs["batch_size"] = 2
    remove = torch.tensor([False, False, False, False, True, True])

    with pytest.raises(EmptyExamError, match="empty 1 of 2 exam"):
        delete_nodes_with_cascade(model, inputs, remove)
