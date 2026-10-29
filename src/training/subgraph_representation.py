"""Materialize one checkpointed beat-graph representation at load time.

Dataset HDF5 files store canonical turning-point nodes and visibility edges.
This module is the single seam that may replace those edges with a graph-local
consecutive temporal chain and independently derive or zero the fixed-width
edge attributes. Common symmetrization is deliberately the final operation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from config.train_models import SubgraphRepresentationConfig
from dataset.edge_attrs import compute_edge_attrs


def _validate_node_ptr(local_node_ptr: np.ndarray, num_nodes: int) -> None:
    if local_node_ptr.ndim != 1 or local_node_ptr.size < 2:
        raise ValueError(
            "local_node_ptr must be a one-dimensional CSR pointer array with at least two entries"
        )
    if int(local_node_ptr[0]) != 0 or int(local_node_ptr[-1]) != num_nodes:
        raise ValueError(
            "local_node_ptr must span the complete exam-local node block: "
            f"expected [0, ..., {num_nodes}], got endpoints "
            f"[{int(local_node_ptr[0])}, ..., {int(local_node_ptr[-1])}]"
        )
    if np.any(np.diff(local_node_ptr) < 0):
        raise ValueError("local_node_ptr must be non-decreasing")


def _validate_graph_local_edges(
    edge_index: np.ndarray,
    local_node_ptr: np.ndarray,
    local_edge_ptr: np.ndarray | None = None,
) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
    if local_edge_ptr is not None:
        expected_graphs = local_node_ptr.size - 1
        if (
            local_edge_ptr.ndim != 1
            or local_edge_ptr.size != expected_graphs + 1
            or int(local_edge_ptr[0]) != 0
            or int(local_edge_ptr[-1]) != edge_index.shape[1]
            or np.any(np.diff(local_edge_ptr) < 0)
        ):
            raise ValueError(
                "local_edge_ptr must be a non-decreasing pointer array with one entry "
                "per graph boundary and must span edge_index"
            )
    if edge_index.shape[1] == 0:
        return
    num_nodes = int(local_node_ptr[-1])
    if edge_index.min() < 0 or edge_index.max() >= num_nodes:
        raise ValueError(
            f"edge_index contains an endpoint outside the exam-local node range [0, {num_nodes})"
        )
    src_graph = np.searchsorted(local_node_ptr[1:], edge_index[0], side="right")
    dst_graph = np.searchsorted(local_node_ptr[1:], edge_index[1], side="right")
    if np.any(src_graph != dst_graph):
        bad = int(np.flatnonzero(src_graph != dst_graph)[0])
        raise ValueError(
            "edge_index crosses graph-local beat-graph boundaries at column "
            f"{bad}: {int(edge_index[0, bad])}->{int(edge_index[1, bad])}"
        )
    if local_edge_ptr is not None:
        edge_graph = np.repeat(
            np.arange(local_node_ptr.size - 1, dtype=np.int64),
            np.diff(local_edge_ptr),
        )
        graph_start = local_node_ptr[edge_graph]
        graph_end = local_node_ptr[edge_graph + 1]
        outside_graph = (edge_index[0] < graph_start) | (edge_index[1] >= graph_end)
        if np.any(outside_graph):
            bad = int(np.flatnonzero(outside_graph)[0])
            graph = int(edge_graph[bad])
            raise ValueError(
                "edge_index contains an endpoint outside graph-local node range "
                f"[{int(graph_start[bad])}, {int(graph_end[bad])}) for graph {graph} "
                f"at column {bad}: {int(edge_index[0, bad])}->{int(edge_index[1, bad])}"
            )
    invalid_order = np.flatnonzero(edge_index[0] >= edge_index[1])
    if invalid_order.size:
        bad = int(invalid_order[0])
        raise ValueError(
            "edge_index violates src < dst invariant at column "
            f"{bad}: {int(edge_index[0, bad])}->{int(edge_index[1, bad])}"
        )


def _temporal_chain_edge_index(local_node_ptr: np.ndarray) -> np.ndarray:
    num_nodes = int(local_node_ptr[-1])
    if num_nodes <= 1:
        return np.empty((2, 0), dtype=np.int64)

    src = np.arange(num_nodes - 1, dtype=np.int64)
    boundary_src = local_node_ptr[1:-1] - 1
    valid_boundaries = boundary_src[(boundary_src >= 0) & (boundary_src < src.size)]
    if valid_boundaries.size:
        keep = np.ones(src.size, dtype=bool)
        keep[valid_boundaries] = False
        src = src[keep]

    edge_index = np.empty((2, src.size), dtype=np.int64)
    edge_index[0] = src
    np.add(src, 1, out=edge_index[1])
    return edge_index


def materialize_subgraph_representation(
    stored_edge_index: np.ndarray,
    local_node_ptr: np.ndarray,
    node_t: np.ndarray,
    node_x: np.ndarray,
    edge_attr_names: Sequence[str],
    representation: SubgraphRepresentationConfig,
    *,
    local_edge_ptr: np.ndarray | None = None,
    symmetrize_edges: bool = False,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exam-local ``edge_index`` and ``edge_attr`` for one resolved mode.

    ``local_node_ptr`` partitions the exam-local node block into stored beat
    graphs. When supplied, ``local_edge_ptr`` assigns each stored edge column
    to its owning graph for exact graph-local bound checks. ``temporal_chain``
    creates ``i -> i+1`` independently inside every interval, including zero
    edges for a one-node graph. Attributes are then computed from the selected
    endpoints or replaced by float32 zeros. Only after both factors are resolved
    may the common reverse-edge transform run.

    On the default ``stored_visibility + derived`` path with no
    symmetrization, ``stored_edge_index`` itself is returned rather than copied.
    """
    local_node_ptr = np.asarray(local_node_ptr, dtype=np.int64)
    num_nodes = int(node_t.shape[0])
    if node_x.shape[0] != num_nodes:
        raise ValueError(f"node_t has {num_nodes} rows but node_x has {node_x.shape[0]} rows")
    if validate:
        _validate_node_ptr(local_node_ptr, num_nodes)
        _validate_graph_local_edges(stored_edge_index, local_node_ptr, local_edge_ptr)

    if representation.adjacency == "stored_visibility":
        edge_index = stored_edge_index
    else:
        edge_index = _temporal_chain_edge_index(local_node_ptr)

    if representation.edge_attributes == "derived":
        edge_attr = compute_edge_attrs(
            edge_index,
            node_t,
            node_x,
            list(edge_attr_names),
            validate=validate,
        )
    else:
        if len(edge_attr_names) != 2:
            raise ValueError(
                "subgraph_representation.edge_attributes='zero' requires exactly two "
                "dataset edge attributes so model.edge_dim and parameterization remain fixed; "
                f"got {list(edge_attr_names)!r}"
            )
        edge_attr = np.zeros((edge_index.shape[1], 2), dtype=np.float32)

    if not symmetrize_edges:
        return edge_index, edge_attr

    reversed_edge_index = np.ascontiguousarray(edge_index[::-1])
    if representation.edge_attributes == "derived":
        reversed_edge_attr = compute_edge_attrs(
            reversed_edge_index,
            node_t,
            node_x,
            list(edge_attr_names),
            validate=False,
        )
    else:
        reversed_edge_attr = np.zeros_like(edge_attr)

    return (
        np.concatenate([edge_index, reversed_edge_index], axis=1),
        np.concatenate([edge_attr, reversed_edge_attr], axis=0),
    )
