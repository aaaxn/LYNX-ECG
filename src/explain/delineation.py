"""NeuroKit adapters for model-output-blind PR-QRS-T delineation.

The graph's stored R anchors are the only heartbeat anchors accepted here.
Neither adapter detects or replaces R peaks. DWT receives the preprocessed lead
in its original orientation; prominence receives a temporary inverted copy only
when the graph records negative QRS polarity. All returned fiducials remain on
the original sample grid.

A complete morphology triplet is an algorithmically available, ordered set of
P onset, QRS onset/offset, and T onset/offset. Completeness is not clinical
fiducial accuracy. Missing or invalid values carry explicit reason codes and
are never filled from fixed windows or another delineator.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from explain.ecg_taxonomy import DurationName, ECGRegion, Polarity


class DelineationMethod(StrEnum):
    """Supported NeuroKit delineation methods."""

    DWT = "dwt"
    PROMINENCE = "prominence"


class DelineationFailureReason(StrEnum):
    """Stable reason codes for an incomplete morphology triplet."""

    DELINEATOR_ERROR = "delineator_error"

    MISSING_P_ONSET_OUTPUT = "missing_p_onset_output"
    MISSING_QRS_ONSET_OUTPUT = "missing_qrs_onset_output"
    MISSING_QRS_OFFSET_OUTPUT = "missing_qrs_offset_output"
    MISSING_T_ONSET_OUTPUT = "missing_t_onset_output"
    MISSING_T_OFFSET_OUTPUT = "missing_t_offset_output"

    P_ONSET_OUTPUT_LENGTH_MISMATCH = "p_onset_output_length_mismatch"
    QRS_ONSET_OUTPUT_LENGTH_MISMATCH = "qrs_onset_output_length_mismatch"
    QRS_OFFSET_OUTPUT_LENGTH_MISMATCH = "qrs_offset_output_length_mismatch"
    T_ONSET_OUTPUT_LENGTH_MISMATCH = "t_onset_output_length_mismatch"
    T_OFFSET_OUTPUT_LENGTH_MISMATCH = "t_offset_output_length_mismatch"

    MISSING_P_ONSET = "missing_p_onset"
    MISSING_QRS_ONSET = "missing_qrs_onset"
    MISSING_QRS_OFFSET = "missing_qrs_offset"
    MISSING_T_ONSET = "missing_t_onset"
    MISSING_T_OFFSET = "missing_t_offset"

    INVALID_P_ONSET_SAMPLE = "invalid_p_onset_sample"
    INVALID_QRS_ONSET_SAMPLE = "invalid_qrs_onset_sample"
    INVALID_QRS_OFFSET_SAMPLE = "invalid_qrs_offset_sample"
    INVALID_T_ONSET_SAMPLE = "invalid_t_onset_sample"
    INVALID_T_OFFSET_SAMPLE = "invalid_t_offset_sample"

    P_ONSET_OUT_OF_SIGNAL = "p_onset_out_of_signal"
    QRS_ONSET_OUT_OF_SIGNAL = "qrs_onset_out_of_signal"
    QRS_OFFSET_OUT_OF_SIGNAL = "qrs_offset_out_of_signal"
    T_ONSET_OUT_OF_SIGNAL = "t_onset_out_of_signal"
    T_OFFSET_OUT_OF_SIGNAL = "t_offset_out_of_signal"

    P_ONSET_NOT_BEFORE_QRS_ONSET = "p_onset_not_before_qrs_onset"
    QRS_ONSET_NOT_BEFORE_R_PEAK = "qrs_onset_not_before_r_peak"
    QRS_OFFSET_NOT_AFTER_R_PEAK = "qrs_offset_not_after_r_peak"
    QRS_OFFSET_NOT_BEFORE_T_ONSET = "qrs_offset_not_before_t_onset"
    T_ONSET_NOT_AFTER_R_PEAK = "t_onset_not_after_r_peak"
    T_ONSET_NOT_BEFORE_T_OFFSET = "t_onset_not_before_t_offset"


@dataclass(frozen=True)
class MorphologyTriplet:
    """Five absolute sample indices defining PR, QRS, and T regions."""

    p_onset_sample: int
    qrs_onset_sample: int
    qrs_offset_sample: int
    t_onset_sample: int
    t_offset_sample: int

    def durations_ms(self, *, sampling_rate: int) -> dict[DurationName, float]:
        """Return PR, QRS, and T durations in milliseconds."""
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
        scale = 1000.0 / sampling_rate
        return {
            ECGRegion.PR: (self.qrs_onset_sample - self.p_onset_sample) * scale,
            ECGRegion.QRS: (self.qrs_offset_sample - self.qrs_onset_sample) * scale,
            ECGRegion.T: (self.t_offset_sample - self.t_onset_sample) * scale,
        }


@dataclass(frozen=True)
class MorphologyLandmarks:
    """Per-landmark absolute sample indices; ``None`` marks an unreturned landmark.

    Each stored value passed the per-landmark checks (present, integer, inside
    the signal). Ordering across landmarks is not encoded here; region
    availability applies the ordering rules.
    """

    p_onset_sample: int | None = None
    qrs_onset_sample: int | None = None
    qrs_offset_sample: int | None = None
    t_onset_sample: int | None = None
    t_offset_sample: int | None = None


@dataclass(frozen=True)
class OrderingRule:
    """One strict ``left < right`` landmark relation and its violation reason."""

    left: str
    right: str
    reason: DelineationFailureReason


ORDERING_RULES: tuple[OrderingRule, ...] = (
    OrderingRule("p_onset", "qrs_onset", DelineationFailureReason.P_ONSET_NOT_BEFORE_QRS_ONSET),
    OrderingRule("qrs_onset", "r_peak", DelineationFailureReason.QRS_ONSET_NOT_BEFORE_R_PEAK),
    OrderingRule("r_peak", "qrs_offset", DelineationFailureReason.QRS_OFFSET_NOT_AFTER_R_PEAK),
    OrderingRule("qrs_offset", "t_onset", DelineationFailureReason.QRS_OFFSET_NOT_BEFORE_T_ONSET),
    OrderingRule("r_peak", "t_onset", DelineationFailureReason.T_ONSET_NOT_AFTER_R_PEAK),
    OrderingRule("t_onset", "t_offset", DelineationFailureReason.T_ONSET_NOT_BEFORE_T_OFFSET),
)

_REGION_RULES: dict[ECGRegion, tuple[tuple[str, ...], frozenset[DelineationFailureReason]]] = {
    ECGRegion.PR: (
        ("p_onset", "qrs_onset"),
        frozenset(
            {
                DelineationFailureReason.P_ONSET_NOT_BEFORE_QRS_ONSET,
                DelineationFailureReason.QRS_ONSET_NOT_BEFORE_R_PEAK,
            }
        ),
    ),
    ECGRegion.QRS: (
        ("qrs_onset", "qrs_offset"),
        frozenset(
            {
                DelineationFailureReason.QRS_ONSET_NOT_BEFORE_R_PEAK,
                DelineationFailureReason.QRS_OFFSET_NOT_AFTER_R_PEAK,
            }
        ),
    ),
    ECGRegion.T: (
        ("t_onset", "t_offset"),
        frozenset(
            {
                DelineationFailureReason.T_ONSET_NOT_AFTER_R_PEAK,
                DelineationFailureReason.T_ONSET_NOT_BEFORE_T_OFFSET,
                DelineationFailureReason.QRS_OFFSET_NOT_BEFORE_T_ONSET,
            }
        ),
    ),
}


def _landmark_values(
    landmarks: MorphologyLandmarks, *, r_peak_sample: int
) -> dict[str, int | None]:
    return {
        "r_peak": r_peak_sample,
        "p_onset": landmarks.p_onset_sample,
        "qrs_onset": landmarks.qrs_onset_sample,
        "qrs_offset": landmarks.qrs_offset_sample,
        "t_onset": landmarks.t_onset_sample,
        "t_offset": landmarks.t_offset_sample,
    }


def ordering_violations(
    landmarks: MorphologyLandmarks, *, r_peak_sample: int
) -> tuple[DelineationFailureReason, ...]:
    """Return the violated ordering relations among the present landmarks.

    Each rule runs only when both of its landmarks are present, so a partial
    beat still names each violated relation instead of hiding it behind a
    missing landmark elsewhere in the chain.
    """
    values = _landmark_values(landmarks, r_peak_sample=r_peak_sample)
    return tuple(
        rule.reason
        for rule in ORDERING_RULES
        if (left := values[rule.left]) is not None
        and (right := values[rule.right]) is not None
        and left >= right
    )


def region_availability(
    landmarks: MorphologyLandmarks, *, r_peak_sample: int
) -> dict[ECGRegion, bool]:
    """Decide per-region availability from partial landmarks and the stored R.

    A region is available when its own landmarks are present and none of its
    blocking relations in :data:`ORDERING_RULES` is violated:

    - PR needs ``p_onset < qrs_onset < r_peak``.
    - QRS needs ``qrs_onset < r_peak < qrs_offset``.
    - T needs ``r_peak < t_onset < t_offset`` and, when a QRS offset exists,
      ``qrs_offset < t_onset`` so QRS and T stay disjoint.

    A beat where all three regions are available satisfies the full ordered
    chain ``p_onset < qrs_onset < r_peak < qrs_offset < t_onset < t_offset``,
    which is exactly the ``beat_complete`` rule.
    """
    values = _landmark_values(landmarks, r_peak_sample=r_peak_sample)
    violated = frozenset(ordering_violations(landmarks, r_peak_sample=r_peak_sample))
    return {
        region: all(values[name] is not None for name in required) and not (violated & blocking)
        for region, (required, blocking) in _REGION_RULES.items()
    }


@dataclass(frozen=True)
class BeatDelineation:
    """Delineation result for one stored R anchor."""

    r_peak_sample: int
    landmarks: MorphologyLandmarks
    reasons: tuple[DelineationFailureReason, ...]

    @property
    def available_regions(self) -> dict[ECGRegion, bool]:
        return region_availability(self.landmarks, r_peak_sample=self.r_peak_sample)

    @property
    def complete(self) -> bool:
        return all(self.available_regions.values())

    @property
    def triplet(self) -> MorphologyTriplet | None:
        """Return the complete ordered triplet, or ``None`` for a partial beat."""
        if not self.complete:
            return None
        landmarks = self.landmarks
        assert landmarks.p_onset_sample is not None
        assert landmarks.qrs_onset_sample is not None
        assert landmarks.qrs_offset_sample is not None
        assert landmarks.t_onset_sample is not None
        assert landmarks.t_offset_sample is not None
        return MorphologyTriplet(
            p_onset_sample=landmarks.p_onset_sample,
            qrs_onset_sample=landmarks.qrs_onset_sample,
            qrs_offset_sample=landmarks.qrs_offset_sample,
            t_onset_sample=landmarks.t_onset_sample,
            t_offset_sample=landmarks.t_offset_sample,
        )


@dataclass(frozen=True)
class LeadDelineation:
    """Ordered delineation results for one exam-lead sequence."""

    method: DelineationMethod
    candidate: str
    sampling_rate: int
    polarity: Polarity
    polarity_corrected: bool
    r_peak_samples: tuple[int, ...]
    beats: tuple[BeatDelineation, ...]
    warning_messages: tuple[str, ...] = ()
    error_detail: str | None = None

    def __post_init__(self) -> None:
        if len(self.r_peak_samples) != len(self.beats):
            raise ValueError("r_peak_samples and beats must have equal lengths")
        if tuple(beat.r_peak_sample for beat in self.beats) != self.r_peak_samples:
            raise ValueError("beats must remain aligned to the stored R peaks")

    @property
    def complete_mask(self) -> tuple[bool, ...]:
        return tuple(beat.complete for beat in self.beats)


@dataclass(frozen=True)
class _WaveField:
    name: str
    neurokit_key: str
    missing_output: DelineationFailureReason
    length_mismatch: DelineationFailureReason
    missing_value: DelineationFailureReason
    invalid_value: DelineationFailureReason
    out_of_signal: DelineationFailureReason


_WAVE_FIELDS = (
    _WaveField(
        "p_onset",
        "ECG_P_Onsets",
        DelineationFailureReason.MISSING_P_ONSET_OUTPUT,
        DelineationFailureReason.P_ONSET_OUTPUT_LENGTH_MISMATCH,
        DelineationFailureReason.MISSING_P_ONSET,
        DelineationFailureReason.INVALID_P_ONSET_SAMPLE,
        DelineationFailureReason.P_ONSET_OUT_OF_SIGNAL,
    ),
    _WaveField(
        "qrs_onset",
        "ECG_R_Onsets",
        DelineationFailureReason.MISSING_QRS_ONSET_OUTPUT,
        DelineationFailureReason.QRS_ONSET_OUTPUT_LENGTH_MISMATCH,
        DelineationFailureReason.MISSING_QRS_ONSET,
        DelineationFailureReason.INVALID_QRS_ONSET_SAMPLE,
        DelineationFailureReason.QRS_ONSET_OUT_OF_SIGNAL,
    ),
    _WaveField(
        "qrs_offset",
        "ECG_R_Offsets",
        DelineationFailureReason.MISSING_QRS_OFFSET_OUTPUT,
        DelineationFailureReason.QRS_OFFSET_OUTPUT_LENGTH_MISMATCH,
        DelineationFailureReason.MISSING_QRS_OFFSET,
        DelineationFailureReason.INVALID_QRS_OFFSET_SAMPLE,
        DelineationFailureReason.QRS_OFFSET_OUT_OF_SIGNAL,
    ),
    _WaveField(
        "t_onset",
        "ECG_T_Onsets",
        DelineationFailureReason.MISSING_T_ONSET_OUTPUT,
        DelineationFailureReason.T_ONSET_OUTPUT_LENGTH_MISMATCH,
        DelineationFailureReason.MISSING_T_ONSET,
        DelineationFailureReason.INVALID_T_ONSET_SAMPLE,
        DelineationFailureReason.T_ONSET_OUT_OF_SIGNAL,
    ),
    _WaveField(
        "t_offset",
        "ECG_T_Offsets",
        DelineationFailureReason.MISSING_T_OFFSET_OUTPUT,
        DelineationFailureReason.T_OFFSET_OUTPUT_LENGTH_MISMATCH,
        DelineationFailureReason.MISSING_T_OFFSET,
        DelineationFailureReason.INVALID_T_OFFSET_SAMPLE,
        DelineationFailureReason.T_OFFSET_OUT_OF_SIGNAL,
    ),
)

_PROMINENCE_KWARGS = frozenset(
    {
        "max_qrs_interval",
        "max_pr_interval",
        "max_r_rise_time",
        "typical_st_segment",
        "max_p_basepoint_interval",
        "max_r_basepoint_interval",
        "max_t_basepoint_interval",
    }
)


def _validated_polarity(polarity: int) -> Polarity:
    if polarity not in (-1, 1):
        raise ValueError(f"polarity must be -1 or +1, got {polarity}")
    return -1 if polarity == -1 else 1


def _stored_r_peaks(r_peak_samples: np.ndarray, *, signal_length: int) -> np.ndarray:
    peaks = np.asarray(r_peak_samples)
    if peaks.ndim != 1:
        raise ValueError(f"r_peak_samples must be one-dimensional, got {peaks.shape}")
    if peaks.size == 0:
        raise ValueError("r_peak_samples must not be empty")
    if not np.issubdtype(peaks.dtype, np.integer):
        raise ValueError("r_peak_samples must contain integer sample indices")
    peaks = peaks.astype(np.int64, copy=False)
    if np.any(peaks < 0) or np.any(peaks >= signal_length):
        raise ValueError("stored R peaks must lie inside the signal")
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("stored R peaks must be strictly increasing")
    return peaks


def _sample_value(
    raw: Any,
    *,
    field: _WaveField,
    signal_length: int,
) -> tuple[int | None, DelineationFailureReason | None]:
    if raw is None:
        return None, field.missing_value
    try:
        numeric = float(raw)
    except (TypeError, ValueError):
        return None, field.invalid_value
    if not np.isfinite(numeric):
        return None, field.missing_value
    if not numeric.is_integer():
        return None, field.invalid_value
    sample = int(numeric)
    if not 0 <= sample < signal_length:
        return None, field.out_of_signal
    return sample, None


def adapt_neurokit_waves(
    waves: Mapping[str, Any],
    *,
    r_peak_samples: np.ndarray,
    signal_length: int,
    sampling_rate: int,
    method: DelineationMethod,
    candidate: str,
    polarity: int,
    polarity_corrected: bool,
    warning_messages: tuple[str, ...] = (),
) -> LeadDelineation:
    """Validate and align NeuroKit wave outputs to explicit stored R peaks.

    A missing output key or a per-key length mismatch invalidates that field for
    every beat because NeuroKit has not retained a trustworthy positional map.
    Other failures are recorded per beat. No value is inferred or imputed.
    """
    if signal_length <= 0:
        raise ValueError(f"signal_length must be positive, got {signal_length}")
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    normalized_polarity = _validated_polarity(polarity)
    if not candidate:
        raise ValueError("candidate must be non-empty")
    method = DelineationMethod(method)
    expected_correction = method is DelineationMethod.PROMINENCE and normalized_polarity == -1
    if polarity_corrected != expected_correction:
        raise ValueError("polarity_corrected disagrees with the method-specific polarity policy")
    peaks = _stored_r_peaks(r_peak_samples, signal_length=signal_length)
    n_beats = peaks.size

    values_by_field: dict[str, Any | None] = {}
    sequence_reasons: list[DelineationFailureReason] = []
    for field in _WAVE_FIELDS:
        if field.neurokit_key not in waves:
            values_by_field[field.name] = None
            sequence_reasons.append(field.missing_output)
            continue
        values = waves[field.neurokit_key]
        try:
            length = len(values)
        except TypeError:
            length = -1
        if length != n_beats:
            values_by_field[field.name] = None
            sequence_reasons.append(field.length_mismatch)
            continue
        values_by_field[field.name] = values

    beats: list[BeatDelineation] = []
    for beat_idx, r_peak in enumerate(peaks):
        reasons = list(sequence_reasons)
        samples: dict[str, int] = {}
        for field in _WAVE_FIELDS:
            values = values_by_field[field.name]
            if values is None:
                continue
            sample, reason = _sample_value(
                values[beat_idx], field=field, signal_length=signal_length
            )
            if reason is not None:
                reasons.append(reason)
            elif sample is not None:
                samples[field.name] = sample

        landmarks = MorphologyLandmarks(
            p_onset_sample=samples.get("p_onset"),
            qrs_onset_sample=samples.get("qrs_onset"),
            qrs_offset_sample=samples.get("qrs_offset"),
            t_onset_sample=samples.get("t_onset"),
            t_offset_sample=samples.get("t_offset"),
        )
        reasons.extend(ordering_violations(landmarks, r_peak_sample=int(r_peak)))
        beats.append(
            BeatDelineation(
                r_peak_sample=int(r_peak),
                landmarks=landmarks,
                reasons=tuple(reasons),
            )
        )

    return LeadDelineation(
        method=method,
        candidate=candidate,
        sampling_rate=sampling_rate,
        polarity=normalized_polarity,
        polarity_corrected=polarity_corrected,
        r_peak_samples=tuple(int(value) for value in peaks),
        beats=tuple(beats),
        warning_messages=warning_messages,
    )


def prepare_delineation_signal(
    signal: np.ndarray,
    *,
    method: DelineationMethod,
    polarity: int,
) -> tuple[np.ndarray, bool]:
    """Return the method-specific signal copy and whether polarity was corrected."""
    method = DelineationMethod(method)
    normalized_polarity = _validated_polarity(polarity)
    values = np.asarray(signal)
    if values.ndim != 1:
        raise ValueError(f"signal must be one-dimensional, got {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("signal must be numeric")
    if not np.isfinite(values).all():
        raise ValueError("signal must be finite")
    polarity_corrected = method is DelineationMethod.PROMINENCE and normalized_polarity == -1
    prepared = np.negative(values) if polarity_corrected else values.copy()
    return prepared, polarity_corrected


def delineate_lead(
    signal: np.ndarray,
    *,
    r_peak_samples: np.ndarray,
    sampling_rate: int,
    method: DelineationMethod,
    candidate: str,
    polarity: int,
    prominence_kwargs: Mapping[str, int] | None = None,
) -> LeadDelineation:
    """Run one NeuroKit adapter using only the supplied stored R peaks."""
    import neurokit2 as nk

    method = DelineationMethod(method)
    normalized_polarity = _validated_polarity(polarity)
    prepared, polarity_corrected = prepare_delineation_signal(
        signal, method=method, polarity=normalized_polarity
    )
    peaks = _stored_r_peaks(r_peak_samples, signal_length=prepared.size)
    kwargs = dict(prominence_kwargs or {})
    if method is DelineationMethod.DWT and kwargs:
        raise ValueError("prominence_kwargs are only valid for the prominence method")
    unknown = kwargs.keys() - _PROMINENCE_KWARGS
    if unknown:
        raise ValueError(f"unsupported prominence kwargs: {sorted(unknown)}")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in kwargs.values()
    ):
        raise ValueError("prominence kwargs must be positive integer milliseconds")

    warning_messages: tuple[str, ...] = ()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _signals, waves = nk.ecg_delineate(
                prepared,
                rpeaks=peaks,
                sampling_rate=sampling_rate,
                method=method.value,
                check=False,
                **kwargs,
            )
        warning_messages = tuple(
            dict.fromkeys(f"{warning.category.__name__}: {warning.message}" for warning in caught)
        )
    except Exception as exc:
        beats = tuple(
            BeatDelineation(
                r_peak_sample=int(r_peak),
                landmarks=MorphologyLandmarks(),
                reasons=(DelineationFailureReason.DELINEATOR_ERROR,),
            )
            for r_peak in peaks
        )
        return LeadDelineation(
            method=method,
            candidate=candidate,
            sampling_rate=sampling_rate,
            polarity=normalized_polarity,
            polarity_corrected=polarity_corrected,
            r_peak_samples=tuple(int(value) for value in peaks),
            beats=beats,
            warning_messages=warning_messages,
            error_detail=f"{type(exc).__name__}: {exc}",
        )

    return adapt_neurokit_waves(
        waves,
        r_peak_samples=peaks,
        signal_length=prepared.size,
        sampling_rate=sampling_rate,
        method=method,
        candidate=candidate,
        polarity=normalized_polarity,
        polarity_corrected=polarity_corrected,
        warning_messages=warning_messages,
    )
