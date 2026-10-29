"""Tests for anti-aliased signal resampling."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.resampling import resample_signal

pytestmark = pytest.mark.unit


def test_resample_changes_length_and_preserves_low_frequency():
    """Resampling 400 -> 300 Hz scales the sample count by the rate ratio and
    keeps a low-frequency component intact (proper anti-aliased resampling, not
    naive decimation)."""
    orig_hz, target_hz = 400, 300
    t = np.arange(int(orig_hz * 2.0)) / orig_hz
    f0 = 5.0
    sig = np.sin(2.0 * np.pi * f0 * t).astype(np.float32)[None, :]

    out = resample_signal(sig, orig_hz, target_hz)

    assert out.shape[0] == 1
    assert abs(out.shape[1] - round(sig.shape[1] * target_hz / orig_hz)) <= 1
    freqs = np.fft.rfftfreq(out.shape[1], d=1.0 / target_hz)
    dominant = freqs[int(np.argmax(np.abs(np.fft.rfft(out[0]))))]
    assert abs(dominant - f0) < 1.0
