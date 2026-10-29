"""Per-beat node-reduction algorithms.

All reducers return ``np.ndarray[int64]`` containing **local** indices into
the input beat (0..n-1), sorted and unique. Local indices are translated
to exam-absolute samples by the writer using ``graph_start_sample[g]``.

Methods:

- :func:`reduce_none`. Keep every sample.
- :func:`reduce_uniform`. ``round(linspace(0, n-1, K))``. Endpoints preserved
  by construction. Does **not** preserve R-peak.
- :func:`reduce_turning_points`. Local extrema of a lowpass-filtered signal.

These are the raw selection algorithms. Method dispatch, config wiring, the
lead-level pre-pass, and per-beat QC live in :mod:`dataset.reduction` (the
:class:`~dataset.reduction.BeatReducer` strategies).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def reduce_none(y: np.ndarray) -> np.ndarray:
    """Keep all samples. Returns ``arange(n)``."""
    n = int(np.asarray(y).shape[0])
    return np.arange(n, dtype=np.int64)


def reduce_uniform(y: np.ndarray, target_nodes: int) -> np.ndarray:
    """``round(linspace(0, n-1, K))`` with endpoints kept."""
    if target_nodes <= 0:
        raise ValueError(f"target_nodes must be > 0, got {target_nodes}")
    n = int(np.asarray(y).shape[0])
    if n <= target_nodes:
        return np.arange(n, dtype=np.int64)
    raw = np.linspace(0.0, float(n - 1), target_nodes)
    idx = np.round(raw).astype(np.int64)
    return np.unique(idx)


def _fill_gaps(selected: set[int], max_gap: int) -> None:
    """Insert uniformly-spaced indices so no consecutive gap exceeds *max_gap*.

    Mutates *selected* in place.
    """
    sorted_idx = sorted(selected)
    for lo, hi in pairwise(sorted_idx):
        gap = hi - lo
        if gap > max_gap:
            n_inserts = (gap - 1) // max_gap
            step = gap / (n_inserts + 1)
            for j in range(1, n_inserts + 1):
                selected.add(round(lo + j * step))


def reduce_turning_points(
    y_filtered: np.ndarray,
    r_peak_local_idx: int,
    *,
    max_gap_samples: int | None = None,
) -> np.ndarray:
    """Select turning points (local extrema) from the filtered signal.

    Returns local indices into the beat (same contract as all other reducers).
    The R-peak is force-injected if not naturally a turning point.

    When *max_gap_samples* is ``None`` (default), endpoints are NOT forced.
    When set, endpoints **are** forced and any consecutive gap larger than the
    threshold is filled with uniformly-spaced nodes.
    """
    y_filt = np.asarray(y_filtered, dtype=np.float64).ravel()
    n = len(y_filt)

    if n < 3:
        return np.array(sorted({0, n - 1, int(r_peak_local_idx)} & set(range(n))), dtype=np.int64)

    dy = np.diff(y_filt)

    sign_changes = dy[:-1] * dy[1:] < 0
    tp_indices = np.where(sign_changes)[0] + 1

    dy_sign = np.sign(dy)
    nz = np.nonzero(dy_sign)[0]
    if nz.size > 0:
        last_nz_idx = np.full(len(dy_sign), -1, dtype=np.intp)
        last_nz_idx[nz] = nz
        np.maximum.accumulate(last_nz_idx, out=last_nz_idx)
        fill_mask = last_nz_idx >= 0
        dy_sign[fill_mask] = dy_sign[last_nz_idx[fill_mask]]
    flat_changes = dy_sign[:-1] * dy_sign[1:] < 0
    flat_tp = np.where(flat_changes)[0] + 1

    selected = set(tp_indices.tolist()) | set(flat_tp.tolist())

    r_idx = int(r_peak_local_idx)
    if 0 <= r_idx < n:
        selected.add(r_idx)

    if len(selected) < 2:
        selected.add(int(np.argmax(y_filt)))
        selected.add(int(np.argmin(y_filt)))
    if len(selected) < 2:
        candidates = {0, n - 1} - selected
        if candidates:
            selected.add(candidates.pop())
        elif 0 not in selected:
            selected.add(0)
        else:
            selected.add(n - 1)

    if max_gap_samples is not None:
        selected.add(0)
        selected.add(n - 1)
        _fill_gaps(selected, max_gap_samples)

    return np.array(sorted(selected), dtype=np.int64)
