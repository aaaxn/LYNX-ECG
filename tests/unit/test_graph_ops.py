"""Unit tests for visibility graph construction and edge_attr formulas."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.edge_attrs import compute_edge_attrs
from dataset.graph_ops import build_visibility_graph

pytestmark = pytest.mark.unit


def test_vg_src_lt_dst_invariant():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(40).astype(np.float32)
    edges = build_visibility_graph(y)
    src = edges[0]
    dst = edges[1]
    assert np.all(src < dst), "all stored edges must satisfy src < dst"


def test_vg_no_duplicates():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(40).astype(np.float32)
    edges = build_visibility_graph(y)
    pairs = edges.T
    unique = np.unique(pairs, axis=0)
    assert len(pairs) == len(unique), "edges contain duplicates"


def test_vg_lex_sorted():
    rng = np.random.default_rng(4)
    y = rng.standard_normal(30).astype(np.float32)
    edges = build_visibility_graph(y)
    pairs = edges.T
    expected_order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    assert np.array_equal(expected_order, np.arange(len(pairs)))


def test_vg_consecutive_edges_present_in_natural():
    """In NaturalVG consecutive nodes are always visible by construction."""
    y = np.array([0.0, 1.0, 0.5, 2.0, 0.3], dtype=np.float32)
    edges = build_visibility_graph(y)
    pair_set = {tuple(p) for p in edges.T.tolist()}
    for i in range(len(y) - 1):
        assert (i, i + 1) in pair_set, f"missing consecutive edge ({i}, {i + 1})"


def test_vg_short_signal_handles_two_nodes():
    """Two-node signal yields exactly one edge."""
    y = np.array([0.0, 1.0], dtype=np.float32)
    edges = build_visibility_graph(y)
    assert edges.shape == (2, 1)
    assert int(edges[0, 0]) == 0
    assert int(edges[1, 0]) == 1


def test_edge_attrs_dtype_and_shape():
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    node_t = np.array([0.0, 0.1, 0.25, 0.5], dtype=np.float32)
    node_x = np.array([[0.5, 0.0], [0.7, 0.25], [0.6, 0.5], [0.4, 1.0]], dtype=np.float32)
    out = compute_edge_attrs(edge_index, node_t, node_x, ["delta_t", "delta_y"])
    assert out.dtype == np.float32
    assert out.shape == (3, 2)


def test_edge_attrs_delta_t_uses_real_node_t():
    """delta_t must NOT be (j - i) / sampling_rate; it must be node_t[dst] - node_t[src]."""
    edge_index = np.array([[0, 1], [3, 2]], dtype=np.int64)
    node_t = np.array([0.0, 0.1, 0.4, 0.99], dtype=np.float32)
    node_x = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    out = compute_edge_attrs(edge_index, node_t, node_x, ["delta_t"])
    np.testing.assert_allclose(out[:, 0], np.array([0.99, 0.3], dtype=np.float32), rtol=1e-6)


def test_edge_attrs_delta_t_positive_with_src_lt_dst():
    """When edges respect src < dst and node_t is monotone, delta_t > 0."""
    edge_index = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int64)
    node_t = np.array([0.0, 0.05, 0.10], dtype=np.float32)
    node_x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    out = compute_edge_attrs(edge_index, node_t, node_x, ["delta_t"])
    assert np.all(out[:, 0] > 0)


def test_edge_attrs_delta_y_uses_only_amplitude_column():
    """Even if node_x has multiple feature columns, delta_y must read column 0."""
    edge_index = np.array([[0], [1]], dtype=np.int64)
    node_t = np.array([0.0, 0.1], dtype=np.float32)
    node_x = np.array([[1.0, 0.0], [5.0, 1.0]], dtype=np.float32)
    out = compute_edge_attrs(edge_index, node_t, node_x, ["delta_y"])
    np.testing.assert_allclose(out[:, 0], np.array([4.0], dtype=np.float32))


def test_edge_attrs_violates_src_lt_dst_raises():
    edge_index = np.array([[2], [1]], dtype=np.int64)
    node_t = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    node_x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="src < dst"):
        compute_edge_attrs(edge_index, node_t, node_x, ["delta_t"])


def test_edge_attrs_validate_false_skips_invariant_scan():
    """validate=False is the hot-path mode: the writer-enforced src < dst
    invariant is not re-scanned, so even violating input computes (garbage)
    output instead of raising. Correct input must give identical results."""
    node_t = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    node_x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    bad = np.array([[2], [1]], dtype=np.int64)
    out = compute_edge_attrs(bad, node_t, node_x, ["delta_t"], validate=False)
    assert out.shape == (1, 1)

    good = np.array([[0, 1], [1, 2]], dtype=np.int64)
    np.testing.assert_array_equal(
        compute_edge_attrs(good, node_t, node_x, ["delta_t", "delta_y"], validate=False),
        compute_edge_attrs(good, node_t, node_x, ["delta_t", "delta_y"]),
    )


def test_edge_attrs_unknown_attr_raises():
    edge_index = np.array([[0], [1]], dtype=np.int64)
    node_t = np.array([0.0, 0.1], dtype=np.float32)
    node_x = np.array([[1.0], [2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown edge attr"):
        compute_edge_attrs(edge_index, node_t, node_x, ["slope"])


def test_edge_attrs_empty_edge_index():
    edge_index = np.empty((2, 0), dtype=np.int64)
    node_t = np.array([0.0, 0.1], dtype=np.float32)
    node_x = np.array([[1.0], [2.0]], dtype=np.float32)
    out = compute_edge_attrs(edge_index, node_t, node_x, ["delta_t", "delta_y"])
    assert out.shape == (0, 2)
    assert out.dtype == np.float32


def test_edge_attrs_empty_attr_list():
    edge_index = np.array([[0], [1]], dtype=np.int64)
    node_t = np.array([0.0, 0.1], dtype=np.float32)
    node_x = np.array([[1.0], [2.0]], dtype=np.float32)
    out = compute_edge_attrs(edge_index, node_t, node_x, [])
    assert out.shape == (1, 0)


def test_edge_attrs_node_x_must_be_2d():
    edge_index = np.array([[0], [1]], dtype=np.int64)
    node_t = np.array([0.0, 0.1], dtype=np.float32)
    node_x = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="node_x must be 2-D"):
        compute_edge_attrs(edge_index, node_t, node_x, ["delta_y"])


def test_edge_attrs_invalid_edge_index_shape():
    edge_index = np.array([0, 1, 2], dtype=np.int64)
    node_t = np.array([0.0, 0.1, 0.2], dtype=np.float32)
    node_x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \[2, E\]"):
        compute_edge_attrs(edge_index, node_t, node_x, ["delta_t"])


def _build_visibility_graph_reference(y: np.ndarray) -> np.ndarray:
    """Reference implementation using the original np.unique(pairs, axis=0) path.

    Kept here only to verify bit-identity with the optimised implementation.
    Do NOT use outside of tests.
    """
    from ts2vg import NaturalVG

    vg = NaturalVG()
    sig = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
    vg.build(sig)
    edges = vg.edges or []
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64)
    src = np.minimum(arr[:, 0], arr[:, 1])
    dst = np.maximum(arr[:, 0], arr[:, 1])
    pairs = np.column_stack([src, dst])
    pairs = np.unique(pairs, axis=0)
    keep = pairs[:, 0] < pairs[:, 1]
    pairs = pairs[keep]
    return np.ascontiguousarray(pairs.T)


def _make_random_signals(rng: np.random.Generator) -> list[np.ndarray]:
    signals: list[np.ndarray] = []
    for n in (0, 1, 2, 3):
        signals.append(rng.standard_normal(n).astype(np.float32))
    for n in (5, 10, 20):
        signals.append(np.zeros(n, dtype=np.float32))
    y = np.tile(np.array([0.0, 1.0, 0.5], dtype=np.float32), 7)
    signals.append(y)
    for n in (4, 8, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200):
        signals.append(rng.standard_normal(n).astype(np.float32))
    signals.append(rng.standard_normal(25).astype(np.float64))
    signals.append(rng.integers(-100, 100, size=30).astype(np.float32))
    return signals


def test_vg_optimised_equals_reference():
    """Optimised 1-D-key implementation must be bit-identical to the reference
    np.unique(axis=0) implementation across ~200 randomised input signals
    (including degenerate cases: empty, 1-node, 2-node, flat, duplicate amplitudes).
    """
    rng = np.random.default_rng(42)
    signals = _make_random_signals(rng)

    for _ in range(200 - len(signals)):
        n = int(rng.integers(0, 80))
        signals.append(rng.standard_normal(n).astype(np.float32))

    for i, y in enumerate(signals):
        ref = _build_visibility_graph_reference(y)
        opt = build_visibility_graph(y)

        assert opt.dtype == np.int64, f"signal {i} (len={len(y)}): dtype mismatch {opt.dtype}"
        assert opt.ndim == 2 and opt.shape[0] == 2, (
            f"signal {i} (len={len(y)}): shape mismatch {opt.shape}"
        )
        assert opt.flags["C_CONTIGUOUS"], f"signal {i} (len={len(y)}): not C-contiguous"
        assert np.array_equal(opt, ref), (
            f"signal {i} (len={len(y)}): values differ\nref=\n{ref}\nopt=\n{opt}"
        )
