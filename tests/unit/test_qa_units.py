"""Unit tests for the internal chunk math added to ``dataset.qa`` deep checks.

These exercise ``_iter_graph_chunks`` and ``_check_graphs_deep`` directly
(private helpers) at a level below the full ``validate_dataset`` integration
contract in ``tests/integration/test_qa.py``, which stays frozen.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dataset.qa import QAIssue, _check_graphs_deep, _iter_graph_chunks

pytestmark = pytest.mark.unit


def _h5_ptr_arrays(path: Path, node_ptr: list[int], edge_ptr: list[int]) -> h5py.File:
    f = h5py.File(path, "w")
    f.create_dataset("graph_node_ptr", data=np.asarray(node_ptr, dtype=np.int64))
    f.create_dataset("graph_edge_ptr", data=np.asarray(edge_ptr, dtype=np.int64))
    return f


def test_iter_graph_chunks_covers_range_and_respects_span_bound(tmp_path):
    node_ptr = [0, 2, 4, 6, 8, 10, 12, 14]
    edge_ptr = [0, 1, 2, 3, 4, 5, 6, 7]
    with _h5_ptr_arrays(tmp_path / "ptr.h5", node_ptr, edge_ptr) as f:
        chunks = list(
            _iter_graph_chunks(f["graph_node_ptr"], f["graph_edge_ptr"], n_graphs=7, chunk_size=5)
        )

    assert chunks == [(0, 2), (2, 4), (4, 6), (6, 7)]

    covered: list[int] = []
    for lo, hi in chunks:
        covered.extend(range(lo, hi))
    assert covered == list(range(7))

    for lo, hi in chunks:
        assert node_ptr[hi] - node_ptr[lo] <= 5
        assert edge_ptr[hi] - edge_ptr[lo] <= 5


def test_iter_graph_chunks_giant_graph_is_its_own_chunk(tmp_path):
    node_ptr = [0, 1, 101, 102]
    edge_ptr = [0, 1, 2, 3]
    with _h5_ptr_arrays(tmp_path / "ptr.h5", node_ptr, edge_ptr) as f:
        chunks = list(
            _iter_graph_chunks(f["graph_node_ptr"], f["graph_edge_ptr"], n_graphs=3, chunk_size=10)
        )

    assert chunks == [(0, 1), (1, 2), (2, 3)]
    covered = [g for lo, hi in chunks for g in range(lo, hi)]
    assert covered == [0, 1, 2]


def test_iter_graph_chunks_empty(tmp_path):
    with _h5_ptr_arrays(tmp_path / "ptr.h5", [0], [0]) as f:
        chunks = list(
            _iter_graph_chunks(f["graph_node_ptr"], f["graph_edge_ptr"], n_graphs=0, chunk_size=5)
        )
    assert chunks == []


def test_iter_graph_chunks_single_graph(tmp_path):
    with _h5_ptr_arrays(tmp_path / "ptr.h5", [0, 3], [0, 2]) as f:
        chunks = list(
            _iter_graph_chunks(f["graph_node_ptr"], f["graph_edge_ptr"], n_graphs=1, chunk_size=5)
        )
    assert chunks == [(0, 1)]


def test_iter_graph_chunks_uniform_graphs_pack_multiple_per_chunk(tmp_path):
    node_ptr = list(range(11))
    edge_ptr = list(range(11))
    with _h5_ptr_arrays(tmp_path / "ptr.h5", node_ptr, edge_ptr) as f:
        chunks = list(
            _iter_graph_chunks(f["graph_node_ptr"], f["graph_edge_ptr"], n_graphs=10, chunk_size=4)
        )

    assert chunks == [(0, 4), (4, 8), (8, 10)]


def _build_minimal_deep_check_file(path: Path) -> None:
    """Two graphs, one chunk, each violating a *different* check:

    - graph 0: node_t_formula (node_t doesn't match the formula).
    - graph 1: edge_index_local_bounds (edge references a node outside the
      graph's local range).

    ``node_reduction_method`` is ``"none"`` and ``edge_attr_names`` is empty
    so anchor and derived-edge-attr checks stay inert. This isolates the
    two violations under test.
    """
    with h5py.File(path, "w") as f:
        f.attrs["sampling_rate"] = 500
        f.attrs["node_reduction_method"] = "none"
        f.attrs["edge_attr_names"] = []

        f.create_dataset("graph_node_ptr", data=np.array([0, 2, 4], dtype=np.int64))
        f.create_dataset("graph_edge_ptr", data=np.array([0, 1, 2], dtype=np.int64))
        f.create_dataset(
            "node_original_sample", data=np.array([100, 110, 200, 210], dtype=np.int64)
        )
        f.create_dataset("node_t", data=np.array([0.0, 0.5, 0.0, 0.02], dtype=np.float32))
        f.create_dataset("node_x", data=np.zeros((4, 1), dtype=np.float32))
        f.create_dataset("graph_start_sample", data=np.array([100, 200], dtype=np.int64))
        f.create_dataset("graph_end_sample", data=np.array([110, 210], dtype=np.int64))
        f.create_dataset("graph_r_peak_sample", data=np.array([105, 205], dtype=np.int64))
        f.create_dataset("edge_index", data=np.array([[0, 0], [1, 5]], dtype=np.int64))


def test_check_graphs_deep_mixed_violation_chunk_reports_both_graphs(tmp_path):
    path = tmp_path / "mixed.h5"
    _build_minimal_deep_check_file(path)

    issues: list[QAIssue] = []
    with h5py.File(path, "r") as f:
        n_checked = _check_graphs_deep(
            f, issues, n_graphs=2, chunk_size=1000, node_ptr_ok=True, edge_ptr_ok=True
        )

    assert n_checked == 2

    by_code = {(i.code, i.message) for i in issues}
    assert any(code == "node_t_formula" and "graph 0" in msg for code, msg in by_code)
    assert any(code == "edge_index_local_bounds" and "graph 1" in msg for code, msg in by_code)

    assert not any(code == "node_t_formula" and "graph 1" in msg for code, msg in by_code)
    assert not any(code == "edge_index_local_bounds" and "graph 0" in msg for code, msg in by_code)


def test_check_graphs_deep_guarded_when_pointers_malformed():
    issues: list[QAIssue] = []
    n_checked = _check_graphs_deep(
        f={},
        issues=issues,
        n_graphs=5,
        chunk_size=10,
        node_ptr_ok=False,
        edge_ptr_ok=True,
    )
    assert n_checked == 0
    assert issues == []
