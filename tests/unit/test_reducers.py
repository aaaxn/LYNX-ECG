"""Unit tests for the reducers (none / uniform / turning_points)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from dataset.reducers import reduce_none, reduce_turning_points, reduce_uniform

pytestmark = pytest.mark.unit


def _qrs_like(n: int, r_idx: int | None = None, amplitude: float = 1.0) -> np.ndarray:
    """Synthetic beat: smooth baseline with a single QRS-like spike at r_idx.

    Provides a clean signal where the R-peak is the dominant geometric
    feature. P/T mounds can be added by callers.
    """
    x = np.arange(n)
    base = 0.05 * np.sin(2 * np.pi * x / max(n - 1, 1))
    sig = base.copy()
    if r_idx is None:
        r_idx = n // 2
    if 0 <= r_idx < n:
        for k in range(-2, 3):
            j = r_idx + k
            if 0 <= j < n:
                sig[j] = base[j] + amplitude * (1.0 - abs(k) / 3.0)
    return sig.astype(np.float32)


def _output_invariants(idx: np.ndarray, n: int) -> None:
    """Common assertions for any reducer's output."""
    assert idx.dtype == np.int64
    assert idx.ndim == 1
    if idx.size > 0:
        assert idx.min() >= 0
        assert idx.max() < n
    assert np.all(np.diff(idx) > 0), "indices must be strictly increasing (sorted, unique)"


def test_none_returns_arange():
    y = np.zeros(50, dtype=np.float32)
    out = reduce_none(y)
    assert np.array_equal(out, np.arange(50, dtype=np.int64))
    _output_invariants(out, 50)


def test_none_empty_signal():
    y = np.zeros(0, dtype=np.float32)
    out = reduce_none(y)
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_uniform_n_greater_than_k_returns_k():
    y = _qrs_like(70)
    out = reduce_uniform(y, target_nodes=30)
    assert out.size == 30
    _output_invariants(out, len(y))


def test_uniform_includes_endpoints():
    y = _qrs_like(70)
    out = reduce_uniform(y, target_nodes=30)
    assert out[0] == 0
    assert out[-1] == 69


def test_uniform_round_not_floor():
    """linspace(0, 9, 4) = [0, 3, 6, 9]; round() preserves these exactly."""
    y = np.zeros(10, dtype=np.float32)
    out = reduce_uniform(y, target_nodes=4)
    assert np.array_equal(out, np.array([0, 3, 6, 9], dtype=np.int64))


def test_uniform_n_equals_k_returns_all():
    y = _qrs_like(30)
    out = reduce_uniform(y, target_nodes=30)
    assert np.array_equal(out, np.arange(30, dtype=np.int64))


def test_uniform_n_less_than_k_returns_all():
    y = _qrs_like(15)
    out = reduce_uniform(y, target_nodes=30)
    assert np.array_equal(out, np.arange(15, dtype=np.int64))


def test_uniform_invalid_k_raises():
    y = _qrs_like(50)
    with pytest.raises(ValueError):
        reduce_uniform(y, target_nodes=0)
    with pytest.raises(ValueError):
        reduce_uniform(y, target_nodes=-3)


def test_tp_returns_sorted_unique_int64():
    y_filtered = _qrs_like(100, r_idx=50)
    out = reduce_turning_points(y_filtered, r_peak_local_idx=50)
    _output_invariants(out, len(y_filtered))


def test_tp_preserves_r_peak():
    n = 100
    y_filtered = np.zeros(n, dtype=np.float32)
    out = reduce_turning_points(y_filtered, r_peak_local_idx=50)
    assert 50 in out


def test_tp_captures_peaks_and_valleys():
    n = 200
    x = np.arange(n, dtype=np.float64)
    y = np.sin(2 * np.pi * x / 50.0).astype(np.float32)
    out = reduce_turning_points(y, r_peak_local_idx=50)
    assert out.size >= 5


@pytest.mark.parametrize(
    "signal",
    [
        _qrs_like(100, r_idx=50, amplitude=0.05),
        _qrs_like(100, r_idx=50, amplitude=25.0),
        np.linspace(-1.0, 1.0, 100).astype(np.float32),
        np.sin(2 * np.pi * np.arange(100) / 7.0).astype(np.float32),
        np.full(100, 0.3, dtype=np.float32),
    ],
    ids=["tiny-amplitude", "large-amplitude", "monotonic-ramp", "high-frequency", "flat"],
)
def test_tp_output_invariants_hold_across_signal_shapes(signal):
    out = reduce_turning_points(signal, r_peak_local_idx=50)
    _output_invariants(out, len(signal))


def test_tp_flat_signal_returns_at_least_two():
    n = 80
    y = np.zeros(n, dtype=np.float32)
    out = reduce_turning_points(y, r_peak_local_idx=40)
    assert 40 in out
    assert out.size >= 2
    _output_invariants(out, n)


def test_tp_flat_signal_rpeak_at_zero():
    n = 50
    y = np.zeros(n, dtype=np.float32)
    out = reduce_turning_points(y, r_peak_local_idx=0)
    assert 0 in out
    assert out.size >= 2
    _output_invariants(out, n)


def test_tp_short_signal_returns_valid():
    y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    out = reduce_turning_points(y, r_peak_local_idx=1)
    _output_invariants(out, len(y))
    assert 1 in out


def test_tp_minimum_two_nodes():
    n = 50
    y = np.linspace(0, 1, n, dtype=np.float32)
    out = reduce_turning_points(y, r_peak_local_idx=25)
    assert out.size >= 2


def test_tp_max_gap_none_matches_baseline():
    """max_gap_samples=None produces identical output to the no-arg call."""
    y = _qrs_like(200, r_idx=100)
    baseline = reduce_turning_points(y, r_peak_local_idx=100)
    with_none = reduce_turning_points(y, r_peak_local_idx=100, max_gap_samples=None)
    np.testing.assert_array_equal(baseline, with_none)


def test_tp_max_gap_forces_endpoints_and_fills():
    """On a flat signal where turning_points produces only {R-peak, argmin},
    max_gap_samples must force endpoints and fill large gaps."""
    n = 200
    y = np.zeros(n, dtype=np.float32)
    y[100] = 1.0

    out = reduce_turning_points(y, r_peak_local_idx=100, max_gap_samples=32)
    _output_invariants(out, n)

    assert 0 in out
    assert n - 1 in out

    gaps = np.diff(out)
    assert np.all(gaps <= 32), f"gaps exceeding threshold: {gaps[gaps > 32]}"


def test_tp_max_gap_no_extra_nodes_when_dense():
    """When turning points are already dense (all gaps <= threshold),
    gap-fill adds only endpoints, no interior nodes."""
    n = 200
    x = np.arange(n, dtype=np.float64)
    y = np.sin(2 * np.pi * x / 20.0).astype(np.float32)
    r_idx = 50

    baseline = reduce_turning_points(y, r_peak_local_idx=r_idx)
    with_gap = reduce_turning_points(y, r_peak_local_idx=r_idx, max_gap_samples=32)

    added = set(with_gap.tolist()) - set(baseline.tolist())
    assert added <= {0, n - 1}


def test_tp_max_gap_preserves_original_turning_points():
    """Gap-fill is purely additive. All original turning points remain."""
    y = _qrs_like(200, r_idx=100)
    baseline = reduce_turning_points(y, r_peak_local_idx=100)
    with_gap = reduce_turning_points(y, r_peak_local_idx=100, max_gap_samples=16)

    assert set(baseline.tolist()).issubset(set(with_gap.tolist()))


def _reduce_turning_points_reference(
    y_filtered: np.ndarray,
    r_peak_local_idx: int,
    *,
    max_gap_samples: int | None = None,
) -> np.ndarray:
    """Original pure-Python implementation kept verbatim as a reference oracle.

    This is the exact code that predates the numpy-vectorized forward-fill,
    preserved here so the equivalence tests can detect any behavioural drift
    without relying on git history.
    """

    y_filt = np.asarray(y_filtered, dtype=np.float64).ravel()
    n = len(y_filt)

    if n < 3:
        return np.array(sorted({0, n - 1, int(r_peak_local_idx)} & set(range(n))), dtype=np.int64)

    dy = np.diff(y_filt)

    sign_changes = dy[:-1] * dy[1:] < 0
    tp_indices = np.where(sign_changes)[0] + 1

    dy_sign = np.sign(dy)
    for i in range(1, len(dy_sign)):
        if dy_sign[i] == 0:
            dy_sign[i] = dy_sign[i - 1]
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
        sorted_idx = sorted(selected)
        for lo, hi in pairwise(sorted_idx):
            gap = hi - lo
            if gap > max_gap_samples:
                n_inserts = (gap - 1) // max_gap_samples
                step = gap / (n_inserts + 1)
                for j in range(1, n_inserts + 1):
                    selected.add(round(lo + j * step))

    return np.array(sorted(selected), dtype=np.int64)


def _make_plateau_signal(rng: np.random.Generator, n: int) -> np.ndarray:
    """Quantize a random signal so flat regions (zero slopes) are common."""
    y = rng.standard_normal(n).astype(np.float64)
    return np.round(y, 1).astype(np.float32)


@pytest.mark.parametrize("max_gap", [None, 1, 5, 20])
def test_tp_vectorized_matches_reference_randomized(max_gap: int | None) -> None:
    """Vectorized forward-fill produces exactly the same indices as the reference loop.

    Covers 300 random beats across a range of challenging cases:
    - Plateau-heavy signals (quantized to 1 d.p. → many zero-slope segments)
    - Leading flat segments (dy[0] == 0)
    - Fully constant signals
    - Very short signals (n = 0..5)
    - R-peak at the signal boundaries
    - All values of max_gap_samples in {None, 1, 5, 20}
    """
    rng = np.random.default_rng(seed=2024_06_10)
    failures: list[str] = []

    for n in range(6):
        for r_idx in range(max(n, 1)):
            y = np.zeros(n, dtype=np.float32)
            expected = _reduce_turning_points_reference(y, r_idx, max_gap_samples=max_gap)
            actual = reduce_turning_points(y, r_idx, max_gap_samples=max_gap)
            if not np.array_equal(actual, expected):
                failures.append(
                    f"short n={n} r={r_idx}: expected={expected.tolist()} got={actual.tolist()}"
                )

    for n in (10, 50, 100):
        r_idx = n // 2
        y = np.full(n, 3.14, dtype=np.float32)
        expected = _reduce_turning_points_reference(y, r_idx, max_gap_samples=max_gap)
        actual = reduce_turning_points(y, r_idx, max_gap_samples=max_gap)
        if not np.array_equal(actual, expected):
            failures.append(f"constant n={n}: expected={expected.tolist()} got={actual.tolist()}")

    for _ in range(200):
        n = int(rng.integers(10, 300))
        y = _make_plateau_signal(rng, n)
        r_idx = int(rng.integers(0, n))
        expected = _reduce_turning_points_reference(y, r_idx, max_gap_samples=max_gap)
        actual = reduce_turning_points(y, r_idx, max_gap_samples=max_gap)
        if not np.array_equal(actual, expected):
            failures.append(
                f"plateau n={n} r={r_idx}: expected={expected[:5].tolist()}... "
                f"got={actual[:5].tolist()}..."
            )

    for _ in range(50):
        n = int(rng.integers(10, 200))
        y = np.zeros(n, dtype=np.float32)
        flat_len = int(rng.integers(1, max(2, n // 3)))
        spike = min(flat_len + 1, n - 1)
        y[spike] = float(rng.uniform(0.5, 2.0))
        r_idx = int(rng.integers(0, n))
        expected = _reduce_turning_points_reference(y, r_idx, max_gap_samples=max_gap)
        actual = reduce_turning_points(y, r_idx, max_gap_samples=max_gap)
        if not np.array_equal(actual, expected):
            failures.append(
                f"leading-flat n={n} flat={flat_len} r={r_idx}: "
                f"expected={expected[:5].tolist()}... got={actual[:5].tolist()}..."
            )

    for _ in range(50):
        n = int(rng.integers(5, 150))
        y = _make_plateau_signal(rng, n)
        for r_idx in (0, n - 1):
            expected = _reduce_turning_points_reference(y, r_idx, max_gap_samples=max_gap)
            actual = reduce_turning_points(y, r_idx, max_gap_samples=max_gap)
            if not np.array_equal(actual, expected):
                failures.append(
                    f"boundary r_idx={r_idx} n={n}: "
                    f"expected={expected[:5].tolist()}... got={actual[:5].tolist()}..."
                )

    assert not failures, "\n".join(failures)
