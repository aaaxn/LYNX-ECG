"""Integration tests for polarity-aware R-peak detection on synthetic ECG.

Note on synthetic-data polarity:

``nk.ecg_simulate`` produces ECG-like waveforms whose absolute amplitude is
sometimes larger at the S-wave trough than at the R-peak. The scoring
function in :func:`dataset.rpeaks.detect_rpeaks` uses anchor amplitude
as a tie-breaker, so on these synthetic signals ``polarity_mode='auto'``
may legitimately pick ``-1`` even though the morphology is "clean".

The tests below therefore avoid pinning the auto-selected polarity to a
specific sign on synthetic data. Instead, they verify the contract:

1. Both polarities are always tested.
2. The peaks returned for ``y`` and ``-y`` are the same (symmetry).
3. The auto-selected polarity for ``y`` and ``-y`` are opposite signs.

Real-ECG polarity behaviour is exercised by the PTB-XL smoke build.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset.rpeaks import RPeakFailure, RPeakResult, detect_rpeaks

pytestmark = pytest.mark.integration


def test_clean_signal_succeeds(synthetic_ecg_clean):
    res = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    assert isinstance(res, RPeakResult)
    assert res.polarity in (1, -1)
    assert res.peaks.dtype == np.int64
    assert res.peaks.size >= 8
    assert np.all(np.diff(res.peaks) > 0)
    assert int(res.peaks.min()) >= 0
    assert int(res.peaks.max()) < synthetic_ecg_clean.size


def test_polarity_auto_y_and_minus_y_yield_same_peaks(synthetic_ecg_clean):
    """Symmetry contract: detect(y) and detect(-y) must return the same sample indices."""
    pos = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    neg = detect_rpeaks(-synthetic_ecg_clean, sampling_rate=400)
    assert isinstance(pos, RPeakResult)
    assert isinstance(neg, RPeakResult)
    np.testing.assert_array_equal(pos.peaks, neg.peaks)


def test_polarity_auto_y_and_minus_y_yield_opposite_polarity(synthetic_ecg_clean):
    """The auto-selected polarity sign must flip when the input is negated."""
    pos = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    neg = detect_rpeaks(-synthetic_ecg_clean, sampling_rate=400)
    assert isinstance(pos, RPeakResult)
    assert isinstance(neg, RPeakResult)
    assert pos.polarity == -neg.polarity


def test_flat_signal_fails(synthetic_ecg_flat):
    res = detect_rpeaks(synthetic_ecg_flat, sampling_rate=400)
    assert isinstance(res, RPeakFailure)
    assert res.reason in {
        "rpeak_no_valid_polarity",
        "rpeak_too_few_peaks",
        "rpeak_detection_exception",
    }


def test_noisy_signal_still_detected(synthetic_ecg_noisy):
    res = detect_rpeaks(synthetic_ecg_noisy, sampling_rate=400)
    assert isinstance(res, RPeakResult)
    assert res.peaks.size >= 5


def test_500hz_signal(synthetic_ecg_500hz):
    res = detect_rpeaks(synthetic_ecg_500hz, sampling_rate=500)
    assert isinstance(res, RPeakResult)
    assert res.peaks.size >= 8


def test_fast_hr_signal(synthetic_ecg_fast_hr):
    res = detect_rpeaks(synthetic_ecg_fast_hr, sampling_rate=400)
    assert isinstance(res, RPeakResult)
    assert res.peaks.size >= 18


def test_too_short_signal_fails():
    sig = np.zeros(10, dtype=np.float32)
    res = detect_rpeaks(sig, sampling_rate=400)
    assert isinstance(res, RPeakFailure)
