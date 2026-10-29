"""Tests for the source-level zero-phase band-pass pre-filter."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.filtering import apply_bandpass, build_bandpass_sos

pytestmark = pytest.mark.unit


def _bandpass(signal: np.ndarray, *, fs: int, hp: float, lp: float, order: int) -> np.ndarray:
    sos = build_bandpass_sos(sampling_rate=fs, highpass_hz=hp, lowpass_hz=lp, order=order)
    return apply_bandpass(signal, sos=sos)


def test_passband_preserved_stopbands_attenuated():
    """A 10 Hz tone (in band) survives; a 0.1 Hz drift and a 60 Hz tone (out of
    band) are strongly attenuated by the 0.5-40 Hz band-pass."""
    fs = 400
    t = np.arange(int(fs * 4.0)) / fs
    in_band = np.sin(2.0 * np.pi * 10.0 * t)
    baseline = np.sin(2.0 * np.pi * 0.1 * t)
    powerline = np.sin(2.0 * np.pi * 60.0 * t)
    sig = (in_band + baseline + powerline).astype(np.float32)[None, :]

    out = _bandpass(sig, fs=fs, hp=0.5, lp=40.0, order=4)

    assert out.shape == sig.shape
    assert out.dtype == np.float32
    assert np.percentile(np.abs(out[0]), 99) > 0.8
    freqs = np.fft.rfftfreq(out.shape[1], d=1.0 / fs)
    mag = np.abs(np.fft.rfft(out[0]))

    def at(f):
        return mag[int(np.argmin(np.abs(freqs - f)))]

    assert at(0.1) < 0.05 * at(10.0)
    assert at(60.0) < 0.05 * at(10.0)


def test_zero_phase_no_shift():
    """sosfiltfilt is forward-backward, so an in-band tone is not phase-shifted."""
    fs = 400
    t = np.arange(int(fs * 4.0)) / fs
    sig = np.sin(2.0 * np.pi * 10.0 * t).astype(np.float32)[None, :]

    out = _bandpass(sig, fs=fs, hp=0.5, lp=40.0, order=4)

    edge = fs
    corr = np.corrcoef(sig[0, edge:-edge], out[0, edge:-edge])[0, 1]
    assert corr > 0.99


def test_per_lead_independence():
    """Each lead is filtered independently along the sample axis."""
    fs = 400
    t = np.arange(int(fs * 4.0)) / fs
    lead0 = np.sin(2.0 * np.pi * 10.0 * t)
    lead1 = np.sin(2.0 * np.pi * 25.0 * t)
    multi = np.stack([lead0, lead1]).astype(np.float32)

    out_multi = _bandpass(multi, fs=fs, hp=0.5, lp=40.0, order=4)
    out0 = _bandpass(lead0.astype(np.float32)[None, :], fs=fs, hp=0.5, lp=40.0, order=4)

    np.testing.assert_allclose(out_multi[0], out0[0], rtol=0, atol=0)


def test_invalid_band_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        build_bandpass_sos(sampling_rate=400, highpass_hz=0.5, lowpass_hz=250.0, order=4)
    with pytest.raises(ValueError):
        build_bandpass_sos(sampling_rate=400, highpass_hz=40.0, lowpass_hz=0.5, order=4)


def test_apply_rejects_non_finite_and_wrong_shape():
    sos = build_bandpass_sos(sampling_rate=400, highpass_hz=0.5, lowpass_hz=40.0, order=4)
    with pytest.raises(ValueError, match="NaN/Inf"):
        apply_bandpass(np.full((1, 100), np.nan, dtype=np.float32), sos=sos)
    with pytest.raises(ValueError, match=r"\[n_leads, n_samples\]"):
        apply_bandpass(np.zeros((100,), dtype=np.float32), sos=sos)
