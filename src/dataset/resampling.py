"""Anti-aliased signal resampling.

The dataset HDF5 stores one file-level ``sampling_rate``, and the visibility
graph's topology plus the ``delta_t`` edge feature depend on the sample spacing.
To make datasets recorded at different native rates comparable (e.g. training on
native-300 Hz CODE-100 and testing on CODE-test, which is packed at 400 Hz),
signals are resampled to a common rate at build time, *before* graph
construction.

Resampling uses polyphase filtering (:func:`scipy.signal.resample_poly`), which
applies an anti-aliasing FIR low-pass at the new Nyquist frequency, correct for
both up- and down-sampling, unlike naive decimation.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_signal(signal: np.ndarray, orig_hz: int, target_hz: int) -> np.ndarray:
    """Resample a ``[n_leads, n_samples]`` signal from ``orig_hz`` to ``target_hz``.

    The rate ratio ``target_hz / orig_hz`` is reduced to its lowest integer
    ``up / down`` terms and applied per lead along the time axis with an
    anti-aliasing polyphase filter. Returns a ``[n_leads, n_samples']`` float32
    array with ``n_samples' ≈ n_samples * target_hz / orig_hz``.
    """
    if orig_hz <= 0 or target_hz <= 0:
        raise ValueError(f"sampling rates must be positive, got orig={orig_hz}, target={target_hz}")
    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim != 2:
        raise ValueError(f"signal must be 2-D [n_leads, n_samples], got shape {sig.shape}")
    if orig_hz == target_hz:
        return sig
    divisor = gcd(int(orig_hz), int(target_hz))
    up = int(target_hz) // divisor
    down = int(orig_hz) // divisor
    resampled = resample_poly(sig, up=up, down=down, axis=1)
    return resampled.astype(np.float32, copy=False)
