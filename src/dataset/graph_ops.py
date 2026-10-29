"""Visibility-graph construction.

Schema invariants enforced here:

- ``edge_index`` values are **local** node indices (``0..K-1``); the writer
  is responsible for the storage dtype (int16 in the current contract).
- Edges are stored with ``edge_index[0, k] < edge_index[1, k]`` (``src < dst``).
- No duplicate edges within a single graph.

Edge attributes are not computed at build time; they are
derived at load time by :mod:`dataset.edge_attrs`.
"""

from __future__ import annotations

import numpy as np
from ts2vg import NaturalVG

_KEY_STRIDE: int = 2**32
_MAX_SAFE_INDEX: int = 2**31 - 1


def build_visibility_graph(y: np.ndarray) -> np.ndarray:
    """Build a natural visibility graph and return ``edge_index`` shape ``[2, E]``.

    Nodes sit at uniform x-coordinates ``0, 1, 2, ...`` (node order), also
    after node reduction.

    The output is sorted lexicographically by ``(src, dst)``, with
    ``src < dst`` and no duplicates.
    """
    vg = NaturalVG()
    vg.build(np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1)))
    edges = vg.edges or []
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"ts2vg returned edges with unexpected shape {arr.shape}; expected (E, 2)")
    src = np.minimum(arr[:, 0], arr[:, 1])
    dst = np.maximum(arr[:, 0], arr[:, 1])

    max_idx = int(max(src.max(), dst.max()))
    if max_idx > _MAX_SAFE_INDEX:
        raise ValueError(
            f"Node index {max_idx} exceeds the safe encoding limit of {_MAX_SAFE_INDEX}. "
            "Cannot encode (src, dst) pair into a single int64 without corrupting key order."
        )
    keys = src * np.int64(_KEY_STRIDE) + dst
    keys = np.unique(keys)
    src_out = keys // np.int64(_KEY_STRIDE)
    dst_out = keys % np.int64(_KEY_STRIDE)

    keep = src_out < dst_out
    src_out = src_out[keep]
    dst_out = dst_out[keep]

    result = np.empty((2, len(src_out)), dtype=np.int64)
    result[0] = src_out
    result[1] = dst_out
    return np.ascontiguousarray(result)
