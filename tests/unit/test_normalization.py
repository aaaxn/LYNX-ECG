"""Tests for per-exam amplitude normalization (gain-invariant node features)."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.normalization import (
    DegenerateAmplitudeError,
    normalize_amplitude,
    p95_lead_median_scale,
)

pytestmark = pytest.mark.unit


def _signal(scale: float, *, n_leads: int = 3, n_samples: int = 1000) -> np.ndarray:
    """A deterministic multi-lead signal whose amplitude is set by ``scale``."""
    rng = np.random.default_rng(0)
    return (scale * rng.standard_normal((n_leads, n_samples))).astype(np.float32)


def test_scale_is_median_across_leads_of_per_lead_p95():
    sig = _signal(1.0, n_leads=3)
    leads = [0, 1, 2]
    expected = float(
        np.median([np.quantile(np.abs(sig[lead].astype(np.float64)), 0.95) for lead in leads])
    )
    assert p95_lead_median_scale(sig, leads) == pytest.approx(expected)


def test_normalization_brings_signals_of_different_gain_to_same_scale():
    """Two recordings of the same shape differing only by a gain factor must
    normalize to (nearly) identical amplitude distributions."""
    base = _signal(1.0)
    gained = (base * 7.3).astype(np.float32)
    leads = [0, 1, 2]

    norm_base = normalize_amplitude(base, leads, method="p95_lead_median")
    norm_gained = normalize_amplitude(gained, leads, method="p95_lead_median")

    np.testing.assert_allclose(norm_base, norm_gained, rtol=1e-5, atol=1e-5)


def test_normalized_scale_is_unity():
    """After normalization the per-exam scale of the result is 1.0 by construction."""
    sig = _signal(42.0)
    leads = [0, 1, 2]
    out = normalize_amplitude(sig, leads, method="p95_lead_median")
    assert p95_lead_median_scale(out, leads) == pytest.approx(1.0, rel=1e-5)


def test_uniform_scalar_preserves_inter_lead_ratios():
    """A single per-exam scalar must not change the ratio between leads."""
    sig = _signal(1.0).copy()
    sig[1] *= 3.0
    leads = [0, 1, 2]
    out = normalize_amplitude(sig, leads, method="p95_lead_median")
    ratio_in = np.quantile(np.abs(sig[1]), 0.95) / np.quantile(np.abs(sig[0]), 0.95)
    ratio_out = np.quantile(np.abs(out[1]), 0.95) / np.quantile(np.abs(out[0]), 0.95)
    assert ratio_out == pytest.approx(ratio_in, rel=1e-5)


def test_scale_uses_only_configured_graph_leads():
    """Leads outside ``graph_leads`` must not influence the scale."""
    sig = _signal(1.0, n_leads=3).copy()
    sig[2] *= 1000.0
    leads = [0, 1]
    expected = float(
        np.median([np.quantile(np.abs(sig[lead].astype(np.float64)), 0.95) for lead in leads])
    )
    assert p95_lead_median_scale(sig, leads) == pytest.approx(expected)


def test_out_of_bounds_leads_excluded_from_statistic():
    sig = _signal(1.0, n_leads=2)
    expected = float(
        np.median([np.quantile(np.abs(sig[lead].astype(np.float64)), 0.95) for lead in (0, 1)])
    )
    assert p95_lead_median_scale(sig, [0, 1, 5]) == pytest.approx(expected)


def test_all_leads_out_of_bounds_returns_none_scale():
    sig = _signal(1.0, n_leads=2)
    assert p95_lead_median_scale(sig, [7, 8]) is None


def test_all_leads_out_of_bounds_is_identity_in_float32():
    sig = _signal(1.0, n_leads=2)
    out = normalize_amplitude(sig, [7, 8], method="p95_lead_median")
    np.testing.assert_array_equal(out, sig.astype(np.float32))


def test_flat_signal_raises_degenerate():
    sig = np.zeros((3, 500), dtype=np.float32)
    with pytest.raises(DegenerateAmplitudeError):
        p95_lead_median_scale(sig, [0, 1, 2])
    with pytest.raises(DegenerateAmplitudeError):
        normalize_amplitude(sig, [0, 1, 2], method="p95_lead_median")


def test_non_finite_signal_raises_degenerate():
    sig = _signal(1.0).copy()
    sig[:] = np.nan
    with pytest.raises(DegenerateAmplitudeError):
        p95_lead_median_scale(sig, [0, 1, 2])


def test_majority_flat_leads_still_degenerate_via_median():
    """Median across leads is 0 when at least half the graph_leads are flat."""
    sig = _signal(1.0, n_leads=3).copy()
    sig[0] = 0.0
    sig[1] = 0.0
    with pytest.raises(DegenerateAmplitudeError):
        p95_lead_median_scale(sig, [0, 1, 2])


def test_one_dead_lead_does_not_break_scale():
    """A single dead lead among healthy ones must not zero the median scale."""
    sig = _signal(1.0, n_leads=3).copy()
    sig[0] = 0.0
    scale = p95_lead_median_scale(sig, [0, 1, 2])
    assert scale is not None and scale > 0.0


def test_unknown_method_rejected():
    sig = _signal(1.0)
    with pytest.raises(ValueError, match="unknown amplitude normalization method"):
        normalize_amplitude(sig, [0, 1, 2], method="nope")


def test_output_is_float32():
    sig = _signal(1.0).astype(np.float64)
    out = normalize_amplitude(sig, [0, 1, 2], method="p95_lead_median")
    assert out.dtype == np.float32


def test_short_signal_scale_is_well_defined():
    """np.quantile is well-defined on a tiny per-lead array, so the scale stays
    finite and positive even for an unrealistically short signal."""
    sig = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    scale = p95_lead_median_scale(sig, [0, 1])
    assert scale is not None and np.isfinite(scale) and scale > 0.0
