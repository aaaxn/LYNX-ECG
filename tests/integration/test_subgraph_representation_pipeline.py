"""End-to-end contracts for load-time subgraph representation materialization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from tests.fixtures.synthetic_graph_hdf5 import build_tiny_hdf5

pytestmark = pytest.mark.integration


def _representation(adjacency: str, edge_attributes: str):
    from config.train_models import SubgraphRepresentationConfig

    return SubgraphRepresentationConfig(
        adjacency=adjacency,
        edge_attributes=edge_attributes,
    )


def _dataset(h5_path: str, adjacency: str, edge_attributes: str, *, symmetrize: bool = False):
    from training.dataset import ExamGraphDataset

    return ExamGraphDataset(
        h5_path,
        exam_indices=[0, 1, 2],
        num_leads=2,
        subgraph_representation=_representation(adjacency, edge_attributes),
        symmetrize_edges=symmetrize,
    )


def test_all_four_cells_preserve_non_edge_exam_tensors(tmp_path) -> None:
    h5_path = build_tiny_hdf5(tmp_path)
    items = {
        (adjacency, attrs): _dataset(h5_path, adjacency, attrs)[0]
        for adjacency in ("stored_visibility", "temporal_chain")
        for attrs in ("derived", "zero")
    }
    baseline = items[("stored_visibility", "derived")]

    invariant_fields = (
        "x",
        "subgraph_node_batch",
        "super_edge_index",
        "super_edge_step",
        "subgraph_lead",
        "subgraph_exam",
        "subgraph_beat",
        "y",
    )
    for item in items.values():
        for field in invariant_fields:
            assert torch.equal(getattr(item, field), getattr(baseline, field)), field
        assert item.x.size(0) == baseline.x.size(0)
        assert item.subgraph_lead.numel() == baseline.subgraph_lead.numel()

    assert torch.equal(
        items[("stored_visibility", "zero")].edge_index,
        baseline.edge_index,
    )
    assert torch.equal(
        items[("temporal_chain", "zero")].edge_index,
        items[("temporal_chain", "derived")].edge_index,
    )
    assert not torch.equal(
        items[("temporal_chain", "derived")].edge_index,
        baseline.edge_index,
    )


def test_default_dataset_path_is_tensor_identical_to_explicit_stored_visibility(tmp_path) -> None:
    from training.dataset import ExamGraphDataset

    h5_path = build_tiny_hdf5(tmp_path)
    default_item = ExamGraphDataset(h5_path, [0], num_leads=2)[0]
    explicit_item = _dataset(h5_path, "stored_visibility", "derived")[0]

    for field in (
        "x",
        "edge_index",
        "edge_attr",
        "subgraph_node_batch",
        "super_edge_index",
        "super_edge_step",
        "subgraph_lead",
        "subgraph_exam",
        "subgraph_beat",
        "y",
    ):
        assert torch.equal(getattr(default_item, field), getattr(explicit_item, field)), field


def test_dataset_chain_edges_are_exact_and_graph_local(tmp_path) -> None:
    h5_path = build_tiny_hdf5(tmp_path)
    item = _dataset(h5_path, "temporal_chain", "derived")[0]

    assert item.edge_index.shape == (2, 6 * 7)
    expected = []
    for graph_idx in range(6):
        start = graph_idx * 8
        expected.extend((node, node + 1) for node in range(start, start + 7))
    actual = list(zip(item.edge_index[0].tolist(), item.edge_index[1].tolist(), strict=True))
    assert actual == expected
    assert all(src < dst for src, dst in actual)
    assert all(src // 8 == dst // 8 for src, dst in actual)


def test_dataset_zero_mode_has_fixed_shape_dtype_after_edge_selection(tmp_path) -> None:
    h5_path = build_tiny_hdf5(tmp_path)

    for adjacency in ("stored_visibility", "temporal_chain"):
        item = _dataset(h5_path, adjacency, "zero")[0]
        assert item.edge_attr.shape == (item.edge_index.size(1), 2)
        assert item.edge_attr.dtype == torch.float32
        assert torch.count_nonzero(item.edge_attr).item() == 0


def test_all_four_cells_batch_and_run_real_model_forward(tmp_path) -> None:
    from models.hier_model import HierGNNModel

    h5_path = build_tiny_hdf5(tmp_path)
    model = HierGNNModel(
        num_classes=5,
        num_leads=2,
        hidden_channels=16,
        beat_encoder_layers=1,
        beat_encoder_mlp_hidden=32,
        gatv2_heads=2,
        gatv2_layers=1,
    )
    model.eval()

    for adjacency in ("stored_visibility", "temporal_chain"):
        for attrs in ("derived", "zero"):
            ds = _dataset(h5_path, adjacency, attrs)
            batch = Batch.from_data_list([ds[0], ds[1]])
            with torch.no_grad():
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


def test_effective_edge_counts_follow_adjacency_and_common_symmetrization(tmp_path) -> None:
    h5_path = build_tiny_hdf5(tmp_path)

    stored = _dataset(h5_path, "stored_visibility", "derived")
    chain = _dataset(h5_path, "temporal_chain", "derived")
    chain_symmetric = _dataset(
        h5_path,
        "temporal_chain",
        "derived",
        symmetrize=True,
    )

    np.testing.assert_array_equal(stored.effective_edges_per_exam(), np.full(3, 6 * 8))
    np.testing.assert_array_equal(chain.effective_edges_per_exam(), np.full(3, 6 * 7))
    np.testing.assert_array_equal(chain_symmetric.effective_edges_per_exam(), np.full(3, 2 * 6 * 7))


def test_build_datasets_propagates_one_resolved_mode_to_every_split(tmp_path) -> None:
    from training.loaders import build_datasets

    h5_path = build_tiny_hdf5(tmp_path)
    representation = _representation("temporal_chain", "zero")

    train_ds, val_ds, test_ds = build_datasets(
        dataset_path=h5_path,
        num_leads=2,
        train_idx=np.array([0]),
        val_idx=np.array([1]),
        test_idx=np.array([2]),
        subgraph_representation=representation,
    )

    assert train_ds.subgraph_representation == representation
    assert val_ds.subgraph_representation == representation
    assert test_ds.subgraph_representation == representation
    assert train_ds._arrays is val_ds._arrays is test_ds._arrays
    for dataset in (train_ds, val_ds, test_ds):
        item = dataset[0]
        assert item.edge_attr.dtype == torch.float32
        assert item.edge_attr.shape[1] == 2
        assert torch.count_nonzero(item.edge_attr).item() == 0
