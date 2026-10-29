"""Focused contracts for the load-time subgraph representation seam."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _representation(*, adjacency: str = "stored_visibility", edge_attributes: str = "derived"):
    from config.train_models import SubgraphRepresentationConfig

    return SubgraphRepresentationConfig(
        adjacency=adjacency,
        edge_attributes=edge_attributes,
    )


def _nodes() -> tuple[np.ndarray, np.ndarray]:
    node_t = np.array([0.0, 0.2, 0.0, 0.0, 0.1, 0.4, 0.9], dtype=np.float32)
    amplitude = np.array([1.0, 3.0, 8.0, -1.0, 2.0, 1.0, 5.0], dtype=np.float32)
    node_x = np.column_stack([amplitude, np.zeros_like(amplitude)]).astype(np.float32)
    return node_t, node_x


def test_temporal_chain_is_deterministic_graph_local_and_exact() -> None:
    from training.subgraph_representation import materialize_subgraph_representation

    local_node_ptr = np.array([0, 2, 3, 7], dtype=np.int64)
    stored = np.array([[0, 3, 3], [1, 4, 6]], dtype=np.int64)
    node_t, node_x = _nodes()
    config = _representation(adjacency="temporal_chain")

    actual, attrs = materialize_subgraph_representation(
        stored,
        local_node_ptr,
        node_t,
        node_x,
        ["delta_t", "delta_y"],
        config,
    )
    repeated, _ = materialize_subgraph_representation(
        stored,
        local_node_ptr,
        node_t,
        node_x,
        ["delta_t", "delta_y"],
        config,
    )

    expected = np.array([[0, 3, 4, 5], [1, 4, 5, 6]], dtype=np.int64)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(repeated, expected)
    assert actual.shape == (2, (2 - 1) + (1 - 1) + (4 - 1))
    assert np.all(actual[0] < actual[1])
    assert not np.any(np.all(actual == np.array([[1], [3]]), axis=0))
    assert not np.any(np.all(actual == np.array([[2], [3]]), axis=0))

    expected_attrs = np.column_stack(
        [
            node_t[expected[1]] - node_t[expected[0]],
            node_x[expected[1], 0] - node_x[expected[0], 0],
        ]
    ).astype(np.float32)
    np.testing.assert_array_equal(attrs, expected_attrs)


def test_stored_visibility_preserves_edge_array_and_derives_selected_endpoints() -> None:
    from training.subgraph_representation import materialize_subgraph_representation

    local_node_ptr = np.array([0, 2, 3, 7], dtype=np.int64)
    stored = np.array([[0, 3, 3, 4], [1, 4, 6, 6]], dtype=np.int64)
    node_t, node_x = _nodes()

    actual, attrs = materialize_subgraph_representation(
        stored,
        local_node_ptr,
        node_t,
        node_x,
        ["delta_t", "delta_y"],
        _representation(),
    )

    assert actual is stored
    expected_attrs = np.column_stack(
        [
            node_t[stored[1]] - node_t[stored[0]],
            node_x[stored[1], 0] - node_x[stored[0], 0],
        ]
    ).astype(np.float32)
    np.testing.assert_array_equal(attrs, expected_attrs)


@pytest.mark.parametrize("adjacency", ["stored_visibility", "temporal_chain"])
def test_zero_attributes_keep_two_channel_float32_shape(adjacency: str) -> None:
    from training.subgraph_representation import materialize_subgraph_representation

    local_node_ptr = np.array([0, 2, 3, 7], dtype=np.int64)
    stored = np.array([[0, 3, 3], [1, 4, 6]], dtype=np.int64)
    node_t, node_x = _nodes()

    edges, attrs = materialize_subgraph_representation(
        stored,
        local_node_ptr,
        node_t,
        node_x,
        ["delta_t", "delta_y"],
        _representation(adjacency=adjacency, edge_attributes="zero"),
    )

    assert attrs.shape == (edges.shape[1], 2)
    assert attrs.dtype == np.float32
    assert np.count_nonzero(attrs) == 0


def test_stored_ordering_is_checked_without_edge_attributes() -> None:
    from training.subgraph_representation import materialize_subgraph_representation

    local_node_ptr = np.array([0, 2, 3, 7], dtype=np.int64)
    stored = np.array([[0], [0]], dtype=np.int64)
    node_t, node_x = _nodes()

    with pytest.raises(ValueError, match=r"src < dst"):
        materialize_subgraph_representation(
            stored,
            local_node_ptr,
            node_t,
            node_x,
            [],
            _representation(edge_attributes="derived"),
        )


def test_stored_ordering_is_checked_in_zero_attribute_mode() -> None:
    from training.subgraph_representation import materialize_subgraph_representation

    local_node_ptr = np.array([0, 2, 3, 7], dtype=np.int64)
    stored = np.array([[1], [0]], dtype=np.int64)
    node_t, node_x = _nodes()

    with pytest.raises(ValueError, match=r"src < dst"):
        materialize_subgraph_representation(
            stored,
            local_node_ptr,
            node_t,
            node_x,
            ["delta_t", "delta_y"],
            _representation(edge_attributes="zero"),
        )


@pytest.mark.parametrize(
    ("adjacency", "edge_attributes"),
    [
        ("stored_visibility", "derived"),
        ("stored_visibility", "zero"),
        ("temporal_chain", "derived"),
        ("temporal_chain", "zero"),
    ],
)
def test_valid_stored_edges_pass_every_representation_mode(
    adjacency: str, edge_attributes: str
) -> None:
    from training.subgraph_representation import materialize_subgraph_representation

    local_node_ptr = np.array([0, 2, 3, 7], dtype=np.int64)
    stored = np.array([[0, 3, 3], [1, 4, 6]], dtype=np.int64)
    node_t, node_x = _nodes()

    edges, attrs = materialize_subgraph_representation(
        stored,
        local_node_ptr,
        node_t,
        node_x,
        ["delta_t", "delta_y"],
        _representation(adjacency=adjacency, edge_attributes=edge_attributes),
    )

    assert edges.shape[0] == 2
    assert np.all(edges[0] < edges[1])
    assert attrs.shape[0] == edges.shape[1]


def test_zero_attributes_remain_zero_when_common_symmetrization_runs_last() -> None:
    from training.subgraph_representation import materialize_subgraph_representation

    local_node_ptr = np.array([0, 2, 3, 7], dtype=np.int64)
    stored = np.array([[0, 3, 3], [1, 4, 6]], dtype=np.int64)
    node_t, node_x = _nodes()

    edges, attrs = materialize_subgraph_representation(
        stored,
        local_node_ptr,
        node_t,
        node_x,
        ["delta_t", "delta_y"],
        _representation(adjacency="temporal_chain", edge_attributes="zero"),
        symmetrize_edges=True,
    )

    half = edges.shape[1] // 2
    assert edges.shape[1] == 2 * ((2 - 1) + (1 - 1) + (4 - 1))
    np.testing.assert_array_equal(edges[:, half:], edges[:, :half][::-1])
    assert attrs.shape == (edges.shape[1], 2)
    assert attrs.dtype == np.float32
    assert np.count_nonzero(attrs) == 0
