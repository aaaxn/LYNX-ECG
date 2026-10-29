"""Unit tests for ``dataset.rpeaks.detect_rpeaks``.

The end-to-end output (peaks, polarity, reason) must match a reference
implementation that calls ``ecg_peaks`` internally. Additional tests cover
degenerate signals and the output contract.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from dataset.rpeaks import RPeakFailure, RPeakResult, detect_rpeaks

pytestmark = pytest.mark.unit


def _simulate(
    duration: float,
    sampling_rate: int,
    heart_rate: int,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    import neurokit2 as nk

    return np.asarray(
        nk.ecg_simulate(
            duration=duration,
            sampling_rate=sampling_rate,
            heart_rate=heart_rate,
            noise=noise,
            random_state=seed,
        ),
        dtype=np.float32,
    )


_NORMAL_CASES: list[tuple[float, int, int, float, int]] = [
    (10.0, 300, 60, 0.0, 1),
    (10.0, 300, 70, 0.0, 2),
    (10.0, 300, 90, 0.05, 3),
    (10.0, 300, 120, 0.0, 4),
    (10.0, 400, 60, 0.0, 5),
    (10.0, 400, 70, 0.0, 6),
    (10.0, 400, 90, 0.05, 7),
    (10.0, 400, 120, 0.0, 8),
    (10.0, 400, 150, 0.0, 9),
    (10.0, 500, 60, 0.0, 10),
    (10.0, 500, 70, 0.0, 11),
    (10.0, 500, 90, 0.05, 12),
    (10.0, 500, 120, 0.0, 13),
    (10.0, 500, 150, 0.0, 14),
]


def _reference_try_polarity(
    signal: np.ndarray,
    sampling_rate: int,
    method: str,
) -> np.ndarray:
    """Call nk.ecg_peaks (correct_artifacts=False) and return ECG_R_Peaks.

    This is an exact copy of the logic that existed in _try_polarity before the
    refactor, used only for the end-to-end equivalence test.
    """
    import neurokit2 as nk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, nk_info = nk.ecg_peaks(signal, sampling_rate=int(sampling_rate), method=method)
    return np.asarray(nk_info.get("ECG_R_Peaks", []), dtype=np.int64).reshape(-1)


def _detect_rpeaks_reference(
    signal: np.ndarray,
    sampling_rate: int,
    method: str = "neurokit",
    min_count: int = 3,
    hr_range_bpm: tuple[float, float] = (30.0, 240.0),
) -> RPeakResult | RPeakFailure:
    """Reproduce detect_rpeaks using ecg_peaks as the inner primitive.

    This is a self-contained copy of the detect_rpeaks logic with only the
    inner NeuroKit call swapped back to ecg_peaks, making it a stable
    reference baseline independent of the code under test.
    """
    from dataset.qc import validate_rpeaks

    def _try(sig: np.ndarray) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
        info: dict[str, Any] = {
            "num_peaks_raw": 0,
            "hr_median_bpm": None,
            "median_anchor_amplitude": None,
        }
        try:
            peaks = _reference_try_polarity(sig, sampling_rate, method)
        except Exception as e:
            return None, "rpeak_detection_exception", {**info, "exception": repr(e)}

        info["num_peaks_raw"] = int(peaks.size)
        reason = validate_rpeaks(
            peaks,
            signal_length=int(sig.size),
            sampling_rate=int(sampling_rate),
            min_count=int(min_count),
            hr_range_bpm=hr_range_bpm,
        )
        if reason is None:
            rr = np.diff(peaks).astype(np.float64) / float(sampling_rate)
            info["hr_median_bpm"] = float(60.0 / np.median(rr))
            info["median_anchor_amplitude"] = float(np.median(np.abs(sig[peaks])))
        return peaks, reason, info

    from dataset.rpeaks import _score

    sig = np.asarray(signal).reshape(-1)
    if sig.size == 0:
        return RPeakFailure(
            reason="rpeak_too_few_peaks", details={"num_peaks_raw": 0, "signal_length": 0}
        )

    pos_peaks, pos_reason, pos_info = _try(sig)
    neg_peaks, neg_reason, neg_info = _try(-sig)
    pos_valid = pos_reason is None
    neg_valid = neg_reason is None

    if not pos_valid and not neg_valid:
        return RPeakFailure(
            reason="rpeak_no_valid_polarity",
            details={
                "positive_failure": pos_reason,
                "negative_failure": neg_reason,
                "positive_info": pos_info,
                "negative_info": neg_info,
            },
        )
    if pos_valid and not neg_valid:
        return RPeakResult(
            peaks=pos_peaks,
            polarity=1,
            info={"polarity": "positive", "negative_failure": neg_reason, **pos_info},
        )
    if neg_valid and not pos_valid:
        return RPeakResult(
            peaks=neg_peaks,
            polarity=-1,
            info={"polarity": "negative", "positive_failure": pos_reason, **neg_info},
        )

    pos_score = _score(pos_info)
    neg_score = _score(neg_info)
    if pos_score >= neg_score:
        return RPeakResult(
            peaks=pos_peaks,
            polarity=1,
            info={
                "polarity": "positive",
                "selected_score": list(pos_score),
                "alternate_score": list(neg_score),
                **pos_info,
            },
        )
    return RPeakResult(
        peaks=neg_peaks,
        polarity=-1,
        info={
            "polarity": "negative",
            "selected_score": list(neg_score),
            "alternate_score": list(pos_score),
            **neg_info,
        },
    )


_E2E_NORMAL_CASES = [(dur, sr, hr, noise, seed) for dur, sr, hr, noise, seed in _NORMAL_CASES]


@pytest.mark.parametrize("dur,sr,hr,noise,seed", _E2E_NORMAL_CASES)
def test_detect_rpeaks_e2e_equivalence(
    dur: float, sr: int, hr: int, noise: float, seed: int
) -> None:
    """detect_rpeaks (using ecg_findpeaks) must match the ecg_peaks reference."""
    sig = _simulate(dur, sr, hr, noise=noise, seed=seed)

    got = detect_rpeaks(sig, sampling_rate=sr)
    ref = _detect_rpeaks_reference(sig, sampling_rate=sr)

    assert type(got) is type(ref), (
        f"sr={sr} hr={hr}: result type mismatch: got {type(got).__name__}, ref {type(ref).__name__}"
    )

    if isinstance(got, RPeakResult):
        assert isinstance(ref, RPeakResult)
        np.testing.assert_array_equal(
            got.peaks,
            ref.peaks,
            err_msg=f"sr={sr} hr={hr}: peaks differ",
        )
        assert got.polarity == ref.polarity, (
            f"sr={sr} hr={hr}: polarity {got.polarity} != {ref.polarity}"
        )
    else:
        assert isinstance(ref, RPeakFailure)
        assert got.reason == ref.reason, (
            f"sr={sr} hr={hr}: failure reason {got.reason!r} != {ref.reason!r}"
        )


@pytest.mark.parametrize("dur,sr,hr,noise,seed", _E2E_NORMAL_CASES)
def test_detect_rpeaks_e2e_equivalence_inverted(
    dur: float, sr: int, hr: int, noise: float, seed: int
) -> None:
    """Inverted polarity: detect_rpeaks must also match the reference."""
    sig = -_simulate(dur, sr, hr, noise=noise, seed=seed)

    got = detect_rpeaks(sig, sampling_rate=sr)
    ref = _detect_rpeaks_reference(sig, sampling_rate=sr)

    assert type(got) is type(ref)
    if isinstance(got, RPeakResult):
        assert isinstance(ref, RPeakResult)
        np.testing.assert_array_equal(got.peaks, ref.peaks)
        assert got.polarity == ref.polarity


def test_degenerate_flat_signal_fails() -> None:
    sig = np.zeros(4000, dtype=np.float32)
    result = detect_rpeaks(sig, sampling_rate=400)
    assert isinstance(result, RPeakFailure)
    assert result.reason in {
        "rpeak_no_valid_polarity",
        "rpeak_too_few_peaks",
        "rpeak_detection_exception",
    }


def test_degenerate_empty_signal_fails() -> None:
    result = detect_rpeaks(np.array([], dtype=np.float32), sampling_rate=400)
    assert isinstance(result, RPeakFailure)
    assert result.reason == "rpeak_too_few_peaks"


def test_degenerate_very_short_signal_fails() -> None:
    sig = np.zeros(10, dtype=np.float32)
    result = detect_rpeaks(sig, sampling_rate=400)
    assert isinstance(result, RPeakFailure)


@pytest.mark.parametrize("sr", [300, 400, 500])
def test_peaks_dtype_and_shape(sr: int) -> None:
    sig = _simulate(10.0, sr, 70, noise=0.0, seed=99)
    result = detect_rpeaks(sig, sampling_rate=sr)
    if isinstance(result, RPeakResult):
        assert result.peaks.dtype == np.int64
        assert result.peaks.ndim == 1
        assert np.all(np.diff(result.peaks) > 0), "peaks must be strictly increasing"
        assert int(result.peaks.min()) >= 0
        assert int(result.peaks.max()) < sig.size


def test_result_info_fields_present() -> None:
    sig = _simulate(10.0, 400, 70, noise=0.0, seed=77)
    result = detect_rpeaks(sig, sampling_rate=400)
    assert isinstance(result, RPeakResult)
    for field in ("polarity", "num_peaks_raw", "hr_median_bpm", "median_anchor_amplitude"):
        assert field in result.info, f"missing info field: {field!r}"
    assert result.info["polarity"] in {"positive", "negative"}
