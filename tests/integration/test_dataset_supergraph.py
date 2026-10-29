"""Integration tests for ExamGraphDataset + supergraph synthesis + PyG batching."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from tests.fixtures.synthetic_graph_hdf5 import build_tiny_hdf5 as _build_tiny_h5
from tests.fixtures.synthetic_graph_hdf5 import make_beat_graph as _make_graph
from training.dataset import (
    ExamGraphDataset,
    GraphIndexArrays,
    HierData,
    _build_super_edges,
    _stream_graph_exam_boundaries,
)

pytestmark = pytest.mark.integration


def test_dataset_derives_edge_attr_from_node_data(tmp_path):
    """The contract derives delta_t / delta_y from node_t / node_x exactly."""
    h5_path = _build_tiny_h5(tmp_path)
    ds = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2)
    item = ds[0]
    assert item.edge_attr.shape == (item.edge_index.size(1), 2)

    graphs = [
        _make_graph(exam_id="E001", lead=lead, beat_index=beat)
        for lead in (0, 1)
        for beat in (0, 1, 2)
    ]
    expected = np.concatenate(
        [
            np.column_stack(
                [
                    g.node_t[g.edge_index[1]] - g.node_t[g.edge_index[0]],
                    g.node_x[g.edge_index[1], 0] - g.node_x[g.edge_index[0], 0],
                ]
            ).astype(np.float32)
            for g in graphs
        ],
        axis=0,
    )
    np.testing.assert_array_equal(item.edge_attr.numpy(), expected)


def test_dataset_rejects_nonbinary_stored_labels_before_cast(tmp_path):
    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r+") as f:
        f["exam_labels"][0, 0] = 2

    with pytest.raises(ValueError, match="binary"):
        GraphIndexArrays.from_hdf5(h5_path)


def test_dataset_rejects_num_leads_mismatch(tmp_path):
    """Regression: lead ids >= num_leads must fail loudly at the data
    boundary, not surface as an opaque embedding-lookup failure (or silent
    clamp) inside the model forward. The file stores leads {0, 1}."""
    h5_path = _build_tiny_h5(tmp_path)
    with pytest.raises(ValueError, match="num_leads mismatch"):
        ExamGraphDataset(h5_path, exam_indices=[0], num_leads=1)


def test_dataset_rejects_later_exam_graph_local_edge_corruption(tmp_path):
    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r+") as f:
        graph_idx = 6
        edge_start = int(f["graph_edge_ptr"][graph_idx])
        node_count = int(f["graph_node_ptr"][graph_idx + 1] - f["graph_node_ptr"][graph_idx])
        f["edge_index"][1, edge_start] = node_count

    ds = ExamGraphDataset(h5_path, exam_indices=[0, 1], num_leads=2)
    _ = ds[0]
    with pytest.raises(ValueError, match="beat-graph boundaries"):
        ds[1]


def test_dataset_rejects_later_exam_edge_shifted_into_next_graph(tmp_path):
    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r+") as f:
        graph_idx = 6
        edge_start = int(f["graph_edge_ptr"][graph_idx])
        node_count = int(f["graph_node_ptr"][graph_idx + 1] - f["graph_node_ptr"][graph_idx])
        f["edge_index"][:, edge_start] = np.array([node_count, node_count + 1], dtype=np.int16)

    ds = ExamGraphDataset(h5_path, exam_indices=[0, 1], num_leads=2)
    _ = ds[0]
    with pytest.raises(ValueError, match="outside graph-local node range"):
        ds[1]


def test_dataset_rejects_later_exam_ordering_without_edge_attributes(tmp_path):
    from config.train_models import SubgraphRepresentationConfig

    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r+") as f:
        graph_idx = 6
        edge_start = int(f["graph_edge_ptr"][graph_idx])
        f["edge_index"][1, edge_start] = f["edge_index"][0, edge_start]

    ds = ExamGraphDataset(
        h5_path,
        exam_indices=[0, 1],
        num_leads=2,
        subgraph_representation=SubgraphRepresentationConfig(
            adjacency="stored_visibility", edge_attributes="zero"
        ),
    )
    _ = ds[0]
    with pytest.raises(ValueError, match=r"src < dst"):
        ds[1]


def test_dataset_getitem_shapes(tmp_path):
    h5_path = _build_tiny_h5(tmp_path)
    ds = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2)
    item = ds[0]
    assert isinstance(item, HierData)
    assert item.subgraph_lead.numel() == 6
    assert item.subgraph_exam.numel() == 6
    assert item.subgraph_lead.tolist() == [0, 0, 0, 1, 1, 1]
    assert item.super_edge_index.size(1) == 8
    assert item.super_edge_step.tolist().count(1) == 4
    assert item.super_edge_step.tolist().count(-1) == 4
    assert item.y.shape == (5,)
    assert item.y[0].item() == 1.0


def test_super_edges_signed_bidirectional():
    """Helper-only test: 3 subgraphs of one lead → ((0,1),(1,0),(1,2),(2,1)) with steps (+1,-1,+1,-1)."""
    graph_lead = np.array([0, 0, 0])
    graph_beat = np.array([0, 1, 2])
    edge_index, step = _build_super_edges(graph_lead, graph_beat)
    assert edge_index.shape == (2, 4)
    pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (1, 2) in pairs and (2, 1) in pairs
    fwd_steps = [step[i] for i, p in enumerate(pairs) if p[0] < p[1]]
    rev_steps = [step[i] for i, p in enumerate(pairs) if p[0] > p[1]]
    assert all(s == 1 for s in fwd_steps)
    assert all(s == -1 for s in rev_steps)


def test_super_edges_two_leads_disjoint():
    """Each lead's chain must be disjoint. Lead 0 indices {0,1}, lead 1 indices {2,3}."""
    graph_lead = np.array([0, 0, 1, 1])
    graph_beat = np.array([0, 1, 0, 1])
    edge_index, _ = _build_super_edges(graph_lead, graph_beat)
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=True))
    assert pairs == {(0, 1), (1, 0), (2, 3), (3, 2)}


def test_pyg_batch_offsets_super_edge_index(tmp_path):
    """PyG Batch.from_data_list must offset super_edge_index by num_subgraphs per Data."""
    h5_path = _build_tiny_h5(tmp_path)
    ds = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2)
    batch = Batch.from_data_list([ds[0], ds[1]])
    assert int(batch.super_edge_index.max()) == 11
    assert batch.subgraph_exam.tolist() == [0] * 6 + [1] * 6
    assert batch.subgraph_lead.max().item() < 2
    assert int(batch.num_graphs) == 2


def test_dataset_end_to_end_with_model(tmp_path):
    """Plumb a tiny dataset through DataLoader+Model and assert finite logits."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from models.hier_model import HierGNNModel

    h5_path = _build_tiny_h5(tmp_path)
    ds = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2)
    loader = PyGDataLoader(ds, batch_size=2, shuffle=False)
    model = HierGNNModel(
        num_classes=5,
        num_leads=2,
        hidden_channels=16,
        beat_encoder_layers=1,
        beat_encoder_mlp_hidden=32,
        gatv2_heads=2,
        gatv2_layers=1,
    )
    batch = next(iter(loader))
    out = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        subgraph_node_batch=batch.subgraph_node_batch,
        super_edge_index=batch.super_edge_index,
        super_edge_step=batch.super_edge_step,
        subgraph_lead=batch.subgraph_lead,
        subgraph_exam=batch.subgraph_exam,
        batch_size=int(batch.num_graphs),
    )
    assert out.logits.shape == (2, 5)
    assert torch.isfinite(out.logits).all()


def test_symmetrize_edges_doubles_edges_and_negates_attrs(tmp_path):
    """flag ON must append the reverse of every stored edge, with delta_t /
    delta_y negated (both are antisymmetric under src/dst swap)."""
    h5_path = _build_tiny_h5(tmp_path)
    ds_off = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2)
    ds_on = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2, symmetrize_edges=True)

    item_off = ds_off[0]
    item_on = ds_on[0]

    e_off = item_off.edge_index.size(1)
    e_on = item_on.edge_index.size(1)
    assert e_on == 2 * e_off

    forward_ei = item_on.edge_index[:, :e_off]
    reverse_ei = item_on.edge_index[:, e_off:]
    assert torch.equal(forward_ei, item_off.edge_index)
    assert torch.equal(reverse_ei, forward_ei.flip(0))

    forward_attr = item_on.edge_attr[:e_off]
    reverse_attr = item_on.edge_attr[e_off:]
    assert torch.equal(forward_attr, item_off.edge_attr)
    torch.testing.assert_close(reverse_attr, -forward_attr)


def test_symmetrize_edges_model_forward(tmp_path):
    """A real HierGNNModel forward pass over a symmetrized item must not error
    and must produce finite logits."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from models.hier_model import HierGNNModel

    h5_path = _build_tiny_h5(tmp_path)
    ds = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2, symmetrize_edges=True)
    loader = PyGDataLoader(ds, batch_size=2, shuffle=False)
    model = HierGNNModel(
        num_classes=5,
        num_leads=2,
        hidden_channels=16,
        beat_encoder_layers=1,
        beat_encoder_mlp_hidden=32,
        gatv2_heads=2,
        gatv2_layers=1,
    )
    batch = next(iter(loader))
    out = model(
        x=batch.x,
        edge_index=batch.edge_index,
        edge_attr=batch.edge_attr,
        subgraph_node_batch=batch.subgraph_node_batch,
        super_edge_index=batch.super_edge_index,
        super_edge_step=batch.super_edge_step,
        subgraph_lead=batch.subgraph_lead,
        subgraph_exam=batch.subgraph_exam,
        batch_size=int(batch.num_graphs),
    )
    assert out.logits.shape == (2, 5)
    assert torch.isfinite(out.logits).all()


def test_symmetrize_first_node_receives_messages(tmp_path):
    """With the flag OFF, directed (src < dst) storage means at least one node
    never appears as a message destination (GINEConv's default src->dst flow
    gives it zero neighbor messages). With the flag ON, every node with any
    incident stored edge appears as a destination at least once."""
    h5_path = _build_tiny_h5(tmp_path)
    ds_off = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2)
    ds_on = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2, symmetrize_edges=True)

    item_off = ds_off[0]
    item_on = ds_on[0]

    all_nodes_off = set(range(item_off.x.size(0)))
    dst_nodes_off = set(item_off.edge_index[1].tolist())
    incident_nodes_off = set(item_off.edge_index[0].tolist()) | dst_nodes_off
    assert incident_nodes_off - dst_nodes_off, "expected at least one message-starved node"
    assert dst_nodes_off <= all_nodes_off

    all_nodes_on = set(range(item_on.x.size(0)))
    dst_nodes_on = set(item_on.edge_index[1].tolist())
    incident_nodes_on = set(item_on.edge_index[0].tolist()) | dst_nodes_on
    assert incident_nodes_on - dst_nodes_on == set()
    assert dst_nodes_on <= all_nodes_on


def test_graph_index_arrays_build_exact_boundaries_with_chunked_runs(tmp_path):
    h5_path = _build_tiny_h5(tmp_path)

    arrays = GraphIndexArrays.from_hdf5(h5_path)
    with h5py.File(h5_path, "r") as f:
        starts, ends = _stream_graph_exam_boundaries(
            f["graph_exam_index"], num_exams=3, chunk_size=4
        )

    expected_starts = np.array([0, 6, 12], dtype=np.int64)
    expected_ends = np.array([6, 12, 18], dtype=np.int64)
    np.testing.assert_array_equal(arrays.exam_to_graph_start, expected_starts)
    np.testing.assert_array_equal(arrays.exam_to_graph_end, expected_ends)
    np.testing.assert_array_equal(starts, expected_starts)
    np.testing.assert_array_equal(ends, expected_ends)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("negative", r"graph_exam_index\[0\]=-1 is outside"),
        ("high", r"outside \[0, 3\)"),
        ("gap", r"cover every exam.*first missing exam index 1"),
        ("order", r"non-decreasing.*1 -> 0"),
    ),
)
def test_graph_index_arrays_reject_malformed_exam_ownership(tmp_path, corruption, message):
    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r+") as f:
        ownership = f["graph_exam_index"][:]
        if corruption == "negative":
            ownership[0] = -1
        elif corruption == "high":
            ownership[-1] = 3
        elif corruption == "gap":
            ownership[ownership == 1] = 2
        else:
            first_exam_one = int(np.flatnonzero(ownership == 1)[0])
            ownership[first_exam_one - 1] = 1
            ownership[first_exam_one] = 0
        f["graph_exam_index"][...] = ownership

    with pytest.raises(ValueError, match=message):
        GraphIndexArrays.from_hdf5(h5_path)


@pytest.mark.parametrize("malformation", ("fractional", "rank_two"))
def test_graph_index_arrays_reject_non_int64_or_non_vector_ownership(tmp_path, malformation):
    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r+") as f:
        ownership = f["graph_exam_index"][:]
        del f["graph_exam_index"]
        if malformation == "fractional":
            malformed = ownership.astype(np.float64)
            malformed[0] += 0.5
        else:
            malformed = ownership[:, np.newaxis]
        f.create_dataset("graph_exam_index", data=malformed)

    with pytest.raises(ValueError, match="graph_exam_index"):
        GraphIndexArrays.from_hdf5(h5_path)


def test_shared_graph_index_arrays_match_self_loading(tmp_path):
    """Two datasets sharing a GraphIndexArrays must return items identical to self-loading ones."""
    h5_path = _build_tiny_h5(tmp_path)

    arrays = GraphIndexArrays.from_hdf5(h5_path)
    shared = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2, arrays=arrays)
    standalone = ExamGraphDataset(h5_path, exam_indices=[0, 1, 2], num_leads=2)

    a = shared[1]
    b = standalone[1]
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)
    assert torch.equal(a.y, b.y)
    assert a.y.dtype == b.y.dtype == torch.float32
