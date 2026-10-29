"""Polarity-aware R-peak / QRS anchor detection.

Wraps NeuroKit2 (``nk.ecg_findpeaks``) to provide:

- Independent attempts on the original ``y`` and the inverted ``-y`` signal.
- Validation of each attempt against ``min_count`` and ``hr_range_bpm``.
- Score-based selection between two valid attempts (peak count, HR
  plausibility, anchor amplitude; ties go to the positive polarity).

The detected indices reference the same sample positions in either polarity
because ``-y[k]`` and ``y[k]`` share index ``k``. The graph itself is always
built from the original ``y``; only the detection pre-step uses ``-y``.

``graph_rpeak_polarity`` is ``+1`` when detection used ``y`` and ``-1`` when
it used ``-y``.

Implementation note: ``nk.ecg_findpeaks`` is used instead of the higher-level
``nk.ecg_peaks`` because ``ecg_peaks`` (with ``correct_artifacts=False``,
which we never enable) builds a pandas DataFrame via ``signal_formatpeaks``
that is immediately discarded.  Calling ``ecg_findpeaks`` directly skips that
DataFrame construction (~2.5 s per 300 exams, ~15 % of the pipeline) while
producing an identical ``ECG_R_Peaks`` array.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config.vocabulary import RPeakPolarityMode
from dataset.qc import validate_rpeaks


@dataclass(frozen=True)
class RPeakResult:
    """Successful detection. ``polarity`` is ``+1`` or ``-1``."""

    peaks: np.ndarray
    polarity: int
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RPeakFailure:
    """Detection failed for this lead. ``reason`` matches the canonical vocabulary."""

    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def _try_polarity(
    signal: np.ndarray,
    sampling_rate: int,
    min_count: int,
    hr_range_bpm: tuple[float, float],
) -> tuple[np.ndarray | None, str | None, dict[str, Any]]:
    """Run NeuroKit on ``signal`` and validate. Returns (peaks_or_None, reason_or_None, info)."""
    info: dict[str, Any] = {
        "num_peaks_raw": 0,
        "hr_median_bpm": None,
        "median_anchor_amplitude": None,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import neurokit2 as nk

            nk_info = nk.ecg_findpeaks(signal, sampling_rate=int(sampling_rate), method="neurokit")
        peaks = np.asarray(nk_info.get("ECG_R_Peaks", []), dtype=np.int64).reshape(-1)
    except Exception as e:  # pragma: no cover
        return None, "rpeak_detection_exception", {**info, "exception": repr(e)}

    info["num_peaks_raw"] = int(peaks.size)

    reason = validate_rpeaks(
        peaks,
        signal_length=int(signal.size),
        sampling_rate=int(sampling_rate),
        min_count=int(min_count),
        hr_range_bpm=hr_range_bpm,
    )
    if reason is None:
        rr = np.diff(peaks).astype(np.float64) / float(sampling_rate)
        info["hr_median_bpm"] = float(60.0 / np.median(rr))
        info["median_anchor_amplitude"] = float(np.median(np.abs(signal[peaks])))
    return peaks, reason, info


def _score(info: dict[str, Any]) -> tuple[int, float, float]:
    """Score for picking between two valid polarities (higher tuple wins).

    Order:
    1. peak count (more is better);
    2. HR plausibility, penalised by absolute distance from 70 bpm;
    3. median absolute amplitude at the anchors (taller QRS preferred).
    """
    peaks = int(info.get("num_peaks_raw") or 0)
    hr = info.get("hr_median_bpm")
    hr_score = -abs(float(hr) - 70.0) if hr is not None else -1e9
    amp = float(info.get("median_anchor_amplitude") or 0.0)
    return (peaks, hr_score, amp)


def detect_rpeaks(
    signal: np.ndarray,
    sampling_rate: int,
    *,
    polarity_mode: RPeakPolarityMode = "auto",
    min_count: int = 3,
    hr_range_bpm: tuple[float, float] = (30.0, 240.0),
) -> RPeakResult | RPeakFailure:
    """Detect R-peaks with optional polarity-aware fallback.

    For ``polarity_mode='auto'``, both ``y`` and ``-y`` are always attempted
    (no early exit on the positive polarity); the better-scoring valid
    polarity is returned. If both fail, a single ``rpeak_no_valid_polarity``
    failure is returned with both per-polarity diagnostics in ``details``.
    """
    sig = np.asarray(signal).reshape(-1)
    if sig.size == 0:
        return RPeakFailure(
            reason="rpeak_too_few_peaks",
            details={"num_peaks_raw": 0, "signal_length": 0},
        )

    if polarity_mode == "positive":
        peaks, reason, info = _try_polarity(sig, sampling_rate, min_count, hr_range_bpm)
        if reason is not None:
            return RPeakFailure(reason=reason, details={"polarity": "positive", **info})
        return RPeakResult(
            peaks=peaks,
            polarity=1,
            info={"polarity": "positive", **info},
        )

    if polarity_mode == "negative":
        peaks, reason, info = _try_polarity(-sig, sampling_rate, min_count, hr_range_bpm)
        if reason is not None:
            return RPeakFailure(reason=reason, details={"polarity": "negative", **info})
        return RPeakResult(
            peaks=peaks,
            polarity=-1,
            info={"polarity": "negative", **info},
        )

    pos_peaks, pos_reason, pos_info = _try_polarity(sig, sampling_rate, min_count, hr_range_bpm)
    neg_peaks, neg_reason, neg_info = _try_polarity(-sig, sampling_rate, min_count, hr_range_bpm)
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
