"""Quality-control rules and the ``SkipEvent`` data class.

Pure functions with no I/O and no logging. Each ``validate_*`` function returns
either ``None`` (the input passes QC) or a string identifying the failure
reason. Reasons match the canonical names used in ``reports/<base>/skipped.jsonl``
and ``report.json["skip_breakdown"]``.

Reason vocabulary:

R-peak / lead level:
- ``rpeak_detection_exception``
- ``rpeak_no_valid_polarity``
- ``rpeak_too_few_peaks``
- ``rpeak_non_monotonic``
- ``rpeak_out_of_bounds``
- ``rpeak_hr_out_of_range``
- ``too_few_valid_beats``
- ``lead_index_out_of_bounds``
- ``signal_too_short_for_filter``

Beat level:
- ``beat_nan_or_inf``
- ``beat_duration_out_of_range``
- ``beat_too_few_samples``
- ``beat_flat_amplitude``
- ``beat_anchor_out_of_bounds``
- ``pip_invalid_output``

Exam level:
- ``amplitude_norm_zero_scale``
- ``exam_no_valid_leads``
- ``exam_unexpected_error``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dataset.beats import BoundaryType


@dataclass(frozen=True)
class SkipEvent:
    """A single recoverable failure / drop event (logged to skipped.jsonl)."""

    level: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    exam_id: str | None = None
    lead: int | None = None
    beat_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"level": self.level, "reason": self.reason}
        if self.exam_id is not None:
            out["exam_id"] = self.exam_id
        if self.lead is not None:
            out["lead"] = int(self.lead)
        if self.beat_index is not None:
            out["beat_index"] = int(self.beat_index)
        out["details"] = dict(self.details)
        return out


def validate_rpeaks(
    peaks: np.ndarray,
    signal_length: int,
    sampling_rate: int,
    min_count: int,
    hr_range_bpm: tuple[float, float],
) -> str | None:
    """Validate detected R-peaks. Returns a reason string on failure or None."""
    if peaks.size < min_count:
        return "rpeak_too_few_peaks"
    if not np.all(np.diff(peaks) > 0):
        return "rpeak_non_monotonic"
    if int(peaks.min()) < 0 or int(peaks.max()) >= signal_length:
        return "rpeak_out_of_bounds"
    rr = np.diff(peaks).astype(np.float64) / float(sampling_rate)
    rr_median = float(np.median(rr))
    if rr_median <= 0:
        return "rpeak_hr_out_of_range"
    hr_median = 60.0 / rr_median
    lo, hi = hr_range_bpm
    if hr_median < lo or hr_median > hi:
        return "rpeak_hr_out_of_range"
    return None


def validate_beat_signal(
    y: np.ndarray,
    duration_seconds: float,
    rr_range_seconds: tuple[float, float],
    min_amplitude: float,
    min_samples: int,
    boundary_type: int = BoundaryType.MIDDLE,
    skip_duration_check_for_boundary: bool = False,
) -> str | None:
    """Validate a candidate beat segment. Returns a reason string on failure or None.

    When ``skip_duration_check_for_boundary`` is True and the span is a boundary
    beat (``boundary_type != BoundaryType.MIDDLE``), the RR-duration range check
    is bypassed: those spans run from the recording edge to the nearest R-peak,
    so their duration is a recording-boundary artifact rather than a true
    cardiac cycle.
    """
    if y.size < min_samples:
        return "beat_too_few_samples"
    if not np.all(np.isfinite(y)):
        return "beat_nan_or_inf"
    is_boundary = boundary_type != BoundaryType.MIDDLE
    skip_duration = is_boundary and skip_duration_check_for_boundary
    if not skip_duration and (
        duration_seconds < rr_range_seconds[0] or duration_seconds > rr_range_seconds[1]
    ):
        return "beat_duration_out_of_range"
    if float(y.max()) - float(y.min()) <= min_amplitude:
        return "beat_flat_amplitude"
    return None


def validate_beat_anchor(
    start_sample: int,
    r_peak_sample: int,
    end_sample: int,
) -> str | None:
    """Validate that ``start <= r_peak <= end`` (closed interval)."""
    if not (start_sample <= r_peak_sample <= end_sample):
        return "beat_anchor_out_of_bounds"
    return None


@dataclass(frozen=True)
class QcContract:
    """The output contract a node reducer promises for a single beat.

    ``expected_count`` is the exact number of selected nodes required, already
    resolved against the beat length (``None`` for variable-size reducers such
    as turning_points, which impose no exact count). ``anchors`` are local
    indices that must survive reduction; out-of-range anchors are ignored by the
    validator, so a reducer may list canonical anchors unconditionally.
    """

    expected_count: int | None
    anchors: tuple[int, ...] = ()


def validate_selected_output(
    selected: np.ndarray,
    *,
    n_beat_samples: int,
    contract: QcContract,
) -> str | None:
    """Validate a reducer's selected local indices against its ``QcContract``.

    Universal invariants (every reducer): ``int64`` dtype, at least 2 nodes,
    strictly increasing, all within ``[0, n_beat_samples)``. Then the
    per-reducer ``contract``: the exact node count (when fixed) and the required
    anchors. Returns ``None`` on success or ``"pip_invalid_output"`` on any
    violation.
    """
    if selected.dtype != np.int64:
        return "pip_invalid_output"
    if selected.size < 2:
        return "pip_invalid_output"
    if not np.all(np.diff(selected) > 0):
        return "pip_invalid_output"
    if int(selected.min()) < 0 or int(selected.max()) >= n_beat_samples:
        return "pip_invalid_output"
    if contract.expected_count is not None and selected.size != contract.expected_count:
        return "pip_invalid_output"
    sel_set = {int(s) for s in selected}
    for anchor in contract.anchors:
        if 0 <= anchor < n_beat_samples and anchor not in sel_set:
            return "pip_invalid_output"
    return None
