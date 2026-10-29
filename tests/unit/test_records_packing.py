"""Tests for the packed worker→writer graph contract."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.records import GraphRecord, pack_graphs
from dataset.schema import GRAPH_SCALAR_SPECS

pytestmark = pytest.mark.unit


def _graph(
    *,
    lead: int,
    beat_index: int,
    n_nodes: int,
    n_edges: int,
    seed: int,
    exam_id: str = "e1",
) -> GraphRecord:
    rng = np.random.default_rng(seed)
    start = 100 * beat_index
    edge_src = rng.integers(0, max(1, n_nodes - 1), size=n_edges)
    edge_dst = edge_src + 1
    return GraphRecord(
        exam_id=exam_id,
        lead=lead,
        beat_index=beat_index,
        boundary_type=int(rng.integers(0, 4)),
        start_sample=start,
        end_sample=start + 99,
        r_peak_sample=start + 50,
        rpeak_polarity=1 if seed % 2 == 0 else -1,
        original_num_nodes=100,
        node_x=rng.normal(size=(n_nodes, 2)).astype(np.float32),
        node_t=np.linspace(0.0, 0.25, n_nodes, dtype=np.float32),
        node_original_sample=np.arange(start, start + n_nodes, dtype=np.int64),
        edge_index=np.stack([edge_src, edge_dst]).astype(np.int64),
    )


def _graph_list() -> list[GraphRecord]:
    return [
        _graph(lead=lead, beat_index=b, n_nodes=5 + b, n_edges=3 * b, seed=lead * 10 + b)
        for lead in (0, 3)
        for b in range(3)
    ]


def test_pack_keeps_every_graph_value_in_order():
    graphs = _graph_list()
    packed = pack_graphs(graphs)

    assert packed.n_graphs == len(graphs)
    np.testing.assert_array_equal(packed.node_x, np.concatenate([g.node_x for g in graphs]))
    np.testing.assert_array_equal(packed.node_t, np.concatenate([g.node_t for g in graphs]))
    np.testing.assert_array_equal(
        packed.node_original_sample, np.concatenate([g.node_original_sample for g in graphs])
    )
    np.testing.assert_array_equal(
        packed.edge_index, np.concatenate([g.edge_index for g in graphs], axis=1)
    )
    np.testing.assert_array_equal(packed.node_counts, [g.reduced_num_nodes for g in graphs])
    np.testing.assert_array_equal(packed.edge_counts, [g.num_edges for g in graphs])
    for spec in GRAPH_SCALAR_SPECS:
        assert spec.graph_field is not None
        np.testing.assert_array_equal(
            packed.scalars[spec.name], [getattr(g, spec.graph_field) for g in graphs]
        )


def test_pack_layout_and_dtypes():
    graphs = _graph_list()
    packed = pack_graphs(graphs)

    assert packed.edge_index.dtype == np.int16
    assert packed.node_x.dtype == np.float32
    assert packed.node_original_sample.dtype == np.int64
    assert int(packed.node_counts.sum()) == packed.node_x.shape[0]
    assert int(packed.edge_counts.sum()) == packed.edge_index.shape[1]
    assert set(packed.scalars) == {spec.name for spec in GRAPH_SCALAR_SPECS}
    for spec in GRAPH_SCALAR_SPECS:
        assert packed.scalars[spec.name].dtype == np.dtype(spec.dtype), spec.name
        assert packed.scalars[spec.name].shape == (len(graphs),)


def test_pack_empty_rejected():
    with pytest.raises(ValueError, match="at least one graph"):
        pack_graphs([])


@pytest.mark.parametrize(
    "edge_index",
    [
        np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ],
    ids=["rank-two-wrong-shape", "rank-one"],
)
def test_pack_rejects_edge_index_without_exact_shape(edge_index: np.ndarray):
    g = _graph(lead=0, beat_index=0, n_nodes=4, n_edges=1, seed=0)
    bad = GraphRecord(
        **{
            **{f: getattr(g, f) for f in g.__dataclass_fields__},
            "edge_index": edge_index,
        }
    )
    with pytest.raises(ValueError, match=r"edge_index must have shape \[2, E\]"):
        pack_graphs([bad])


def test_pack_rejects_endpoint_outside_graph_local_range():
    g = _graph(lead=0, beat_index=0, n_nodes=4, n_edges=1, seed=0)
    bad = GraphRecord(
        **{
            **{f: getattr(g, f) for f in g.__dataclass_fields__},
            "edge_index": np.array([[0], [4]], dtype=np.int64),
        }
    )
    with pytest.raises(ValueError, match=r"graph-local.*\[0, 4\).*exam_id='e1'.*lead=0.*beat=0"):
        pack_graphs([bad])


def test_pack_rejects_negative_endpoint():
    g = _graph(lead=0, beat_index=0, n_nodes=4, n_edges=1, seed=0)
    bad = GraphRecord(
        **{
            **{f: getattr(g, f) for f in g.__dataclass_fields__},
            "edge_index": np.array([[0], [-1]], dtype=np.int64),
        }
    )
    with pytest.raises(ValueError, match="graph-local"):
        pack_graphs([bad])


@pytest.mark.parametrize(
    "edge_index",
    [
        np.array([[0], [-40_000]], dtype=np.int64),
        np.array([[0.5], [1.5]], dtype=np.float64),
    ],
    ids=["int16-underflow", "fractional-endpoints"],
)
def test_pack_rejects_unsafe_edge_values(edge_index: np.ndarray):
    g = _graph(lead=0, beat_index=0, n_nodes=4, n_edges=1, seed=0)
    bad = GraphRecord(
        **{
            **{f: getattr(g, f) for f in g.__dataclass_fields__},
            "edge_index": edge_index,
        }
    )
    expected_error = "int16 range" if edge_index.dtype.kind in "iu" else "integer dtype"
    with pytest.raises(ValueError, match=expected_error):
        pack_graphs([bad])


@pytest.mark.parametrize(
    "edge_index",
    [
        np.array([[1], [1]], dtype=np.int64),
        np.array([[2], [1]], dtype=np.int64),
    ],
    ids=["equal", "descending"],
)
def test_pack_rejects_non_increasing_edge(edge_index: np.ndarray):
    g = _graph(lead=0, beat_index=0, n_nodes=4, n_edges=1, seed=0)
    bad = GraphRecord(
        **{
            **{f: getattr(g, f) for f in g.__dataclass_fields__},
            "edge_index": edge_index,
        }
    )
    with pytest.raises(ValueError, match=r"src < dst"):
        pack_graphs([bad])


def test_pack_rejects_edge_values_outside_int16():
    g = _graph(lead=0, beat_index=0, n_nodes=4, n_edges=1, seed=0)
    bad = GraphRecord(
        **{
            **{f: getattr(g, f) for f in g.__dataclass_fields__},
            "edge_index": np.array([[0], [40_000]], dtype=np.int64),
        }
    )
    with pytest.raises(ValueError, match="int16 range"):
        pack_graphs([bad])


def test_pack_rejects_too_many_nodes():
    g = _graph(lead=0, beat_index=0, n_nodes=4, n_edges=0, seed=0)
    big = GraphRecord(
        **{
            **{f: getattr(g, f) for f in g.__dataclass_fields__},
            "node_x": np.zeros((40_000, 2), dtype=np.float32),
        }
    )
    with pytest.raises(ValueError, match="at most"):
        pack_graphs([big])
