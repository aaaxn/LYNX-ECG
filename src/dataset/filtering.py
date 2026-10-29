"""Filter design and application for the dataset build pipeline.

The band-pass is a zero-phase Butterworth filter (``butter`` + ``sosfiltfilt``
with default padding, float64 compute). It runs on each raw source lead at the
source's native rate, before any resampling, so every source (code100 WFDB,
CODE-test, PTB-XL) is built in the same filtered amplitude domain.

The low-pass is the per-lead smoothing filter used by the turning_points
node reducer.

Filter design is pure in its parameters, so the ``cached_*_sos`` variants
memoize the SOS per distinct parameter set (``scipy.signal.butter`` is
expensive enough to dominate when called once per lead per exam). The cached
master arrays are kept read-only internally; callers receive a fresh writable
copy (a few dozen floats, since ``sosfiltfilt``'s Cython kernel requires writable
buffers), so no caller can corrupt the cache.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt


def build_bandpass_sos(
    *,
    sampling_rate: int,
    highpass_hz: float,
    lowpass_hz: float,
    order: int,
) -> np.ndarray:
    """Design a zero-phase Butterworth band-pass as second-order sections.

    ``order`` is the value passed to :func:`scipy.signal.butter` (a band-pass of
    order ``n`` yields ``2n`` poles / ``n`` SOS sections). Cutoffs are physical
    frequencies in Hz and must satisfy ``0 < highpass_hz < lowpass_hz < fs/2``.
    """
    fs = float(sampling_rate)
    if fs <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    if not (0.0 < float(highpass_hz) < float(lowpass_hz) < fs / 2.0):
        raise ValueError(
            "band-pass requires 0 < highpass_hz < lowpass_hz < Nyquist (fs/2); "
            f"got highpass_hz={highpass_hz}, lowpass_hz={lowpass_hz}, fs={fs}"
        )
    if int(order) <= 0:
        raise ValueError(f"order must be positive, got {order}")
    return butter(
        int(order),
        [float(highpass_hz), float(lowpass_hz)],
        btype="bandpass",
        fs=fs,
        output="sos",
    )


def build_lowpass_sos(*, sampling_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Design a zero-phase Butterworth low-pass as second-order sections.

    Used by the turning_points reducer to smooth each lead before extrema
    detection. ``cutoff_hz`` must satisfy ``0 < cutoff_hz < fs/2``.
    """
    fs = float(sampling_rate)
    if fs <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    if not (0.0 < float(cutoff_hz) < fs / 2.0):
        raise ValueError(
            f"low-pass requires 0 < cutoff_hz < Nyquist (fs/2); got cutoff_hz={cutoff_hz}, fs={fs}"
        )
    if int(order) <= 0:
        raise ValueError(f"order must be positive, got {order}")
    return butter(int(order), float(cutoff_hz), btype="low", fs=fs, output="sos")


@lru_cache(maxsize=64)
def _bandpass_sos_master(
    *, sampling_rate: int, highpass_hz: float, lowpass_hz: float, order: int
) -> np.ndarray:
    sos = build_bandpass_sos(
        sampling_rate=sampling_rate,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        order=order,
    )
    sos.setflags(write=False)
    return sos


@lru_cache(maxsize=64)
def _lowpass_sos_master(*, sampling_rate: int, cutoff_hz: float, order: int) -> np.ndarray:
    sos = build_lowpass_sos(sampling_rate=sampling_rate, cutoff_hz=cutoff_hz, order=order)
    sos.setflags(write=False)
    return sos


def cached_bandpass_sos(
    *, sampling_rate: int, highpass_hz: float, lowpass_hz: float, order: int
) -> np.ndarray:
    """Memoized :func:`build_bandpass_sos`; returns a fresh writable copy."""
    return _bandpass_sos_master(
        sampling_rate=sampling_rate,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        order=order,
    ).copy()


def cached_lowpass_sos(*, sampling_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Memoized :func:`build_lowpass_sos`; returns a fresh writable copy."""
    return _lowpass_sos_master(sampling_rate=sampling_rate, cutoff_hz=cutoff_hz, order=order).copy()


def apply_bandpass(signal: np.ndarray, *, sos: np.ndarray) -> np.ndarray:
    """Zero-phase band-pass a ``[n_leads, n_samples]`` signal along the sample axis.

    Each lead is filtered independently (``sosfiltfilt``, forward-backward, so no
    phase shift). The input is brought to float32 first, the ``ExamSample.signal``
    contract dtype, and the storage dtype of the raw packs the offline filter
    consumed, so every source enters the filter in the same numeric domain. The
    compute then happens in float64 and the result is cast back to float32,
    matching the offline filter that produced the filtered HDF5 packs.
    """
    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim != 2:
        raise ValueError(f"signal must be [n_leads, n_samples], got shape {sig.shape}")
    if not np.isfinite(sig).all():
        raise ValueError("signal contains NaN/Inf before filtering")
    filtered = sosfiltfilt(sos, sig.astype(np.float64, copy=False), axis=1)
    filtered = np.asarray(filtered, dtype=np.float32)
    if not np.isfinite(filtered).all():
        raise ValueError("signal contains NaN/Inf after filtering")
    return filtered
