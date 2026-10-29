"""Edge-attribute derivation from stored node arrays.

The dataset contract does not store ``edge_attr``; every consumer (training
Dataset, QA, visualization) derives it from ``node_t`` / ``node_x`` through
this module.

Invariants:

- ``delta_t`` is always ``node_t[dst] - node_t[src]`` from the real
  ``node_t``; never ``(j - i) / sampling_rate`` from reduced indices.
- ``delta_y`` always uses ``node_x[:, 0]`` (the amplitude column), regardless
  of how many feature columns ``node_x`` has.

This module is deliberately free of build-only dependencies (``ts2vg``,
``scipy``) so the training DataLoader workers can import it cheaply.
"""

from __future__ import annotations

import numpy as np


def compute_edge_attrs(
    edge_index: np.ndarray,
    node_t: np.ndarray,
    node_x: np.ndarray,
    edge_attr_names: list[str],
    *,
    validate: bool = True,
) -> np.ndarray:
    """Compute edge attributes; returns ``[E, len(edge_attr_names)]`` float32.

    Currently supports ``"delta_t"`` and ``"delta_y"`` (contract default).

    ``edge_index`` may be local to one graph or rebased to a larger node
    block, as long as it indexes into ``node_t`` / ``node_x`` and respects
    the ``src < dst`` storage invariant.

    ``validate=False`` skips the O(E) ``src < dst`` scan. The invariant is
    enforced by the HDF5 writer and verified by full dataset QA, so hot-path
    callers (the training dataset) may disable the re-check after
    validating their first item; QA and verification keep the default.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
    edge_index = np.asarray(edge_index, dtype=np.int64)
    e = int(edge_index.shape[1])

    if not edge_attr_names:
        return np.empty((e, 0), dtype=np.float32)

    if e == 0:
        return np.empty((0, len(edge_attr_names)), dtype=np.float32)

    src = edge_index[0]
    dst = edge_index[1]
    if validate and np.any(src >= dst):
        raise ValueError(
            "edge_index violates src < dst invariant; offending edges: "
            f"{[(int(src[k]), int(dst[k])) for k in np.where(src >= dst)[0][:5].tolist()]}"
        )

    node_t_arr = np.asarray(node_t)
    node_x_arr = np.asarray(node_x)
    if node_x_arr.ndim != 2:
        raise ValueError(f"node_x must be 2-D ([N, F]), got shape {node_x_arr.shape}")

    cols: list[np.ndarray] = []
    for name in edge_attr_names:
        if name == "delta_t":
            cols.append(node_t_arr[dst] - node_t_arr[src])
        elif name == "delta_y":
            cols.append(node_x_arr[dst, 0] - node_x_arr[src, 0])
        else:
            raise ValueError(f"Unknown edge attr {name!r}; supported: 'delta_t', 'delta_y'")
    return np.column_stack(cols).astype(np.float32, copy=False)
