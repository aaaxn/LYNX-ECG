"""Synthetic ECG signal generators for unit / integration tests.

All generators are deterministic via fixed seeds. They use NeuroKit's
``ecg_simulate`` for realistic morphology. Inverted variants negate the
signal to test polarity-aware detection.
"""

from __future__ import annotations

import numpy as np
import pytest


def _simulate(
    duration: float = 10.0,
    sampling_rate: int = 400,
    heart_rate: int = 70,
    noise: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    import neurokit2 as nk

    sig = nk.ecg_simulate(
        duration=duration,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
        noise=noise,
        random_state=seed,
    )
    return np.asarray(sig, dtype=np.float32)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@pytest.fixture(scope="module")
def synthetic_ecg_clean() -> np.ndarray:
    """10 s, 400 Hz, HR=70 bpm, noise=0.0. The ideal-input baseline."""
    return _read_only(_simulate(duration=10.0, sampling_rate=400, heart_rate=70, noise=0.0))


@pytest.fixture(scope="module")
def synthetic_ecg_inverted(synthetic_ecg_clean: np.ndarray) -> np.ndarray:
    """Independent negation of the clean beat. Used for polarity-aware detection tests."""
    return _read_only(np.negative(synthetic_ecg_clean))


@pytest.fixture(scope="module")
def synthetic_ecg_noisy() -> np.ndarray:
    """10 s, 400 Hz, HR=70 bpm, noise=0.1. Moderate baseline drift."""
    return _read_only(_simulate(duration=10.0, sampling_rate=400, heart_rate=70, noise=0.1))


@pytest.fixture(scope="module")
def synthetic_ecg_flat() -> np.ndarray:
    """10 s of zeros at 400 Hz. No R-peak detectable."""
    return _read_only(np.zeros(4000, dtype=np.float32))


@pytest.fixture(scope="module")
def synthetic_ecg_500hz() -> np.ndarray:
    """10 s, 500 Hz, HR=70 bpm, noise=0.0. For PTB-XL-like sampling."""
    return _read_only(_simulate(duration=10.0, sampling_rate=500, heart_rate=70, noise=0.0))


@pytest.fixture(scope="module")
def synthetic_ecg_fast_hr() -> np.ndarray:
    """10 s, 400 Hz, HR=150 bpm. Still inside [30, 240] bpm window."""
    return _read_only(_simulate(duration=10.0, sampling_rate=400, heart_rate=150, noise=0.0))
