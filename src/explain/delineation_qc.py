"""Validated QC contracts for model-output-blind ECG delineation.

This module turns per-lead NeuroKit results into auditable lead-beat records,
method/candidate summaries, deterministic review samples, and common delineation
support. Duration bounds are flags only: they expose technically suspicious
candidates without rejecting long PR or QRS durations that may be pathological.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import pairwise
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from explain.delineation import (
    DelineationFailureReason,
    DelineationMethod,
    MorphologyLandmarks,
    region_availability,
)
from explain.ecg_taxonomy import REGION_ORDER, DurationName, ECGRegion, Polarity

PositiveInt = Annotated[int, Field(gt=0)]


class ProminenceParameters(BaseModel):
    """Explicit NeuroKit prominence search limits, in milliseconds."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_qrs_interval: PositiveInt
    max_pr_interval: PositiveInt
    max_r_rise_time: PositiveInt
    typical_st_segment: PositiveInt
    max_p_basepoint_interval: PositiveInt
    max_r_basepoint_interval: PositiveInt
    max_t_basepoint_interval: PositiveInt


class DurationFlagRange(BaseModel):
    """Broad technical flag range; values outside remain valid delineations."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    minimum: Annotated[float, Field(ge=0)]
    maximum: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def _ordered(self) -> DurationFlagRange:
        if self.minimum >= self.maximum:
            raise ValueError("duration flag range requires minimum < maximum")
        return self


class ReviewSampleConfig(BaseModel):
    """Deterministic labeled-overlay sample size."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    exams_per_class: PositiveInt
    beats_per_exam: PositiveInt


class DelineationQCConfig(BaseModel):
    """Strict, reproducible configuration for the standalone delineation QC."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    rpeak_source: Literal["graph_r_peak_sample"]
    signal_source: Literal["preprocessed_graph_build_pipeline"]
    dwt_signal_orientation: Literal["original"]
    prominence_negative_polarity_policy: Literal["invert"]
    prominence_candidates: dict[str, ProminenceParameters]
    selected_prominence_candidate: str
    selection_rationale: Annotated[str, Field(min_length=1)]
    duration_flag_ranges_ms: dict[DurationName, DurationFlagRange]
    common_support_exam_summary_threshold: Annotated[float, Field(ge=0, le=1)]
    review_sample: ReviewSampleConfig

    @model_validator(mode="after")
    def _complete_contract(self) -> DelineationQCConfig:
        if not self.prominence_candidates:
            raise ValueError("prominence_candidates must not be empty")
        if any(not name for name in self.prominence_candidates):
            raise ValueError("prominence candidate names must be non-empty")
        if self.selected_prominence_candidate not in self.prominence_candidates:
            raise ValueError(
                "selected_prominence_candidate must name one prominence_candidates entry"
            )
        if set(self.duration_flag_ranges_ms) != set(REGION_ORDER):
            raise ValueError("duration_flag_ranges_ms must define pr, qrs, and t")
        return self

    @property
    def selected_prominence_kwargs(self) -> ProminenceParameters:
        return self.prominence_candidates[self.selected_prominence_candidate]


class FiducialSamples(BaseModel):
    """Per-landmark absolute sample indices; ``None`` marks an unreturned landmark."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    p_onset_sample: int | None
    qrs_onset_sample: int | None
    qrs_offset_sample: int | None
    t_onset_sample: int | None
    t_offset_sample: int | None

    def complete_triplet(self) -> tuple[int, int, int, int, int]:
        """Return the five landmark samples, requiring every one to be present."""
        p_onset = self.p_onset_sample
        qrs_onset = self.qrs_onset_sample
        qrs_offset = self.qrs_offset_sample
        t_onset = self.t_onset_sample
        t_offset = self.t_offset_sample
        if (
            p_onset is None
            or qrs_onset is None
            or qrs_offset is None
            or t_onset is None
            or t_offset is None
        ):
            raise ValueError("complete triplet requires all five landmarks")
        return (p_onset, qrs_onset, qrs_offset, t_onset, t_offset)

    def landmarks(self) -> MorphologyLandmarks:
        return MorphologyLandmarks(
            p_onset_sample=self.p_onset_sample,
            qrs_onset_sample=self.qrs_onset_sample,
            qrs_offset_sample=self.qrs_offset_sample,
            t_onset_sample=self.t_onset_sample,
            t_offset_sample=self.t_offset_sample,
        )


class DelineationRecord(BaseModel):
    """Auditable result for one method/candidate and one lead-beat graph unit."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    graph_idx: Annotated[int, Field(ge=0)]
    exam_idx: Annotated[int, Field(ge=0)]
    exam_id: Annotated[str, Field(min_length=1)]
    positive_classes: list[str]
    lead: Annotated[int, Field(ge=0)]
    beat_index: Annotated[int, Field(ge=0)]
    start_sample: Annotated[int, Field(ge=0)]
    end_sample: Annotated[int, Field(ge=0)]
    boundary_type: int
    r_peak_sample: Annotated[int, Field(ge=0)]
    polarity: Polarity
    method: DelineationMethod
    candidate: Annotated[str, Field(min_length=1)]
    signal_orientation: Literal["original", "polarity_corrected"]
    polarity_corrected: bool
    complete: bool
    available_regions: list[ECGRegion]
    reasons: list[DelineationFailureReason]
    fiducials: FiducialSamples
    durations_ms: dict[DurationName, float]
    suspicious_duration_flags: list[DurationName]
    warning_messages: list[str]
    error_detail: str | None

    @model_validator(mode="after")
    def _coherent(self) -> DelineationRecord:
        expected_correction = self.method is DelineationMethod.PROMINENCE and self.polarity == -1
        if self.polarity_corrected != expected_correction:
            raise ValueError("polarity correction metadata violates the method policy")
        expected_orientation = "polarity_corrected" if expected_correction else "original"
        if self.signal_orientation != expected_orientation:
            raise ValueError("signal_orientation violates the method polarity policy")
        if self.end_sample < self.start_sample:
            raise ValueError("end_sample must be >= start_sample")
        if len(set(self.positive_classes)) != len(self.positive_classes):
            raise ValueError("positive_classes must not contain duplicates")
        if len(set(self.suspicious_duration_flags)) != len(self.suspicious_duration_flags):
            raise ValueError("suspicious_duration_flags must not contain duplicates")
        expected_available = region_availability(
            self.fiducials.landmarks(), r_peak_sample=self.r_peak_sample
        )
        if self.available_regions != [region for region in ECGRegion if expected_available[region]]:
            raise ValueError("available_regions must match the fiducials in canonical order")
        if self.complete != all(expected_available.values()):
            raise ValueError("complete must equal all three regions being available")
        if set(self.durations_ms) != set(self.available_regions):
            raise ValueError("durations_ms must cover exactly the available regions")
        if not set(self.suspicious_duration_flags) <= set(self.available_regions):
            raise ValueError("suspicious duration flags require the region to be available")
        if self.complete:
            if self.reasons:
                raise ValueError("complete records cannot carry failure reasons")
            if self.error_detail is not None:
                raise ValueError("complete records cannot carry delineator error details")
            ordered_samples = (
                self.fiducials.p_onset_sample,
                self.fiducials.qrs_onset_sample,
                self.r_peak_sample,
                self.fiducials.qrs_offset_sample,
                self.fiducials.t_onset_sample,
                self.fiducials.t_offset_sample,
            )
            if any(
                left is None or right is None or left >= right
                for left, right in pairwise(ordered_samples)
            ):
                raise ValueError("complete fiducials must be strictly ordered around the stored R")
        elif not self.reasons:
            raise ValueError("incomplete records require at least one failure reason")
        elif (DelineationFailureReason.DELINEATOR_ERROR in self.reasons) != (
            self.error_detail is not None
        ):
            raise ValueError("error_detail must appear exactly with delineator_error")
        return self


class DurationSummary(BaseModel):
    """Distribution of one duration over complete units."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    n: Annotated[int, Field(ge=0)]
    median: float | None
    q25: float | None
    q75: float | None
    minimum: float | None
    maximum: float | None


class QCGroupSummary(BaseModel):
    """Completeness, reasons, warnings, flags, and duration distributions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    n_units: Annotated[int, Field(ge=0)]
    n_complete: Annotated[int, Field(ge=0)]
    completeness_rate: Annotated[float, Field(ge=0, le=1)]
    failure_reason_counts: dict[str, int]
    warning_counts: dict[str, int]
    suspicious_duration_counts: dict[str, int]
    n_suspicious_duration_units: Annotated[int, Field(ge=0)]
    durations_ms: dict[str, DurationSummary]

    @model_validator(mode="after")
    def _bounded_counts(self) -> QCGroupSummary:
        if not self.n_suspicious_duration_units <= self.n_complete <= self.n_units:
            raise ValueError(
                "QC group counts require suspicious-duration units <= complete <= units"
            )
        total_flags = sum(self.suspicious_duration_counts.values())
        largest_flag_count = max(self.suspicious_duration_counts.values(), default=0)
        if not largest_flag_count <= self.n_suspicious_duration_units <= total_flags:
            raise ValueError(
                "suspicious-duration unit count must lie between the largest "
                "per-duration count and the total flag count"
            )
        return self


class MethodQCSummary(BaseModel):
    """QC dimensions for one delineation method/candidate."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    method: DelineationMethod
    candidate: str
    overall: QCGroupSummary
    by_class: dict[str, QCGroupSummary]
    by_exam: dict[str, QCGroupSummary]
    by_lead: dict[str, QCGroupSummary]
    by_beat_index: dict[str, QCGroupSummary]
    by_polarity: dict[str, QCGroupSummary]


class SupportGroupSummary(BaseModel):
    """Common-support coverage for one grouping."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    n_units: Annotated[int, Field(ge=0)]
    n_common: Annotated[int, Field(ge=0)]
    common_rate: Annotated[float, Field(ge=0, le=1)]


class CommonSupportUnit(BaseModel):
    """DWT/prominence completeness membership for one lead-beat unit."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    graph_idx: int
    exam_idx: int
    exam_id: str
    positive_classes: list[str]
    lead: int
    beat_index: int
    polarity: Polarity
    dwt_complete: bool
    prominence_complete: bool
    common: bool


class CommonSupportSection(BaseModel):
    """Materialized common support plus secondary exam-coverage summary."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    selected_prominence_candidate: str
    exam_summary_threshold: float
    membership: list[CommonSupportUnit]
    common_graph_indices: list[int]
    overall: SupportGroupSummary
    by_class: dict[str, SupportGroupSummary]
    by_exam: dict[str, SupportGroupSummary]
    by_lead: dict[str, SupportGroupSummary]
    by_beat_index: dict[str, SupportGroupSummary]
    by_polarity: dict[str, SupportGroupSummary]
    exams_at_or_above_threshold: list[int]


class ReviewSampleItem(BaseModel):
    """One labeled exam with the stored beat indices used in overlay pages."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: str
    exam_idx: int
    exam_id: str
    beat_indices: list[int]


def _duration_summary(values: list[float]) -> DurationSummary:
    if not values:
        return DurationSummary(
            n=0,
            median=None,
            q25=None,
            q75=None,
            minimum=None,
            maximum=None,
        )
    array = np.asarray(values, dtype=np.float64)
    return DurationSummary(
        n=int(array.size),
        median=float(np.median(array)),
        q25=float(np.quantile(array, 0.25)),
        q75=float(np.quantile(array, 0.75)),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


def _group_summary(records: list[DelineationRecord]) -> QCGroupSummary:
    if not records:
        raise ValueError("cannot summarize an empty delineation group")
    complete = [record for record in records if record.complete]
    reasons = Counter(reason.value for record in records for reason in record.reasons)
    warning_counts = Counter(message for record in records for message in record.warning_messages)
    suspicious = Counter(
        duration for record in complete for duration in record.suspicious_duration_flags
    )
    duration_values: dict[str, list[float]] = {"pr": [], "qrs": [], "t": []}
    for record in complete:
        for name, value in record.durations_ms.items():
            duration_values[name].append(value)
    return QCGroupSummary(
        n_units=len(records),
        n_complete=len(complete),
        completeness_rate=len(complete) / len(records),
        failure_reason_counts=dict(sorted(reasons.items())),
        warning_counts=dict(sorted(warning_counts.items())),
        suspicious_duration_counts=dict(sorted(suspicious.items())),
        n_suspicious_duration_units=sum(
            bool(record.suspicious_duration_flags) for record in complete
        ),
        durations_ms={name: _duration_summary(values) for name, values in duration_values.items()},
    )


def _summaries_by(
    records: list[DelineationRecord],
    keys: list[list[str]],
) -> dict[str, QCGroupSummary]:
    grouped: dict[str, list[DelineationRecord]] = defaultdict(list)
    for record, record_keys in zip(records, keys, strict=True):
        for key in record_keys:
            grouped[key].append(record)
    return {key: _group_summary(grouped[key]) for key in sorted(grouped)}


def build_method_summaries(records: list[DelineationRecord]) -> list[MethodQCSummary]:
    """Summarize every method/candidate over all required audit dimensions."""
    grouped: dict[tuple[DelineationMethod, str], list[DelineationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.candidate)].append(record)

    summaries: list[MethodQCSummary] = []
    for (method, candidate), group in sorted(
        grouped.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        summaries.append(
            MethodQCSummary(
                method=method,
                candidate=candidate,
                overall=_group_summary(group),
                by_class=_summaries_by(group, [record.positive_classes for record in group]),
                by_exam=_summaries_by(group, [[str(record.exam_idx)] for record in group]),
                by_lead=_summaries_by(group, [[str(record.lead)] for record in group]),
                by_beat_index=_summaries_by(group, [[str(record.beat_index)] for record in group]),
                by_polarity=_summaries_by(group, [[str(record.polarity)] for record in group]),
            )
        )
    return summaries


def _support_summary(units: list[CommonSupportUnit]) -> SupportGroupSummary:
    if not units:
        raise ValueError("cannot summarize an empty common-support group")
    n_common = sum(unit.common for unit in units)
    return SupportGroupSummary(
        n_units=len(units),
        n_common=n_common,
        common_rate=n_common / len(units),
    )


def _support_by(
    units: list[CommonSupportUnit],
    keys: list[list[str]],
) -> dict[str, SupportGroupSummary]:
    grouped: dict[str, list[CommonSupportUnit]] = defaultdict(list)
    for unit, unit_keys in zip(units, keys, strict=True):
        for key in unit_keys:
            grouped[key].append(unit)
    return {key: _support_summary(grouped[key]) for key in sorted(grouped)}


def _records_by_graph(
    records: list[DelineationRecord],
    *,
    method: DelineationMethod,
    candidate: str,
) -> dict[int, DelineationRecord]:
    selected = [
        record for record in records if record.method is method and record.candidate == candidate
    ]
    by_graph = {record.graph_idx: record for record in selected}
    if len(by_graph) != len(selected):
        raise ValueError(f"duplicate {method.value}/{candidate} records for one graph_idx")
    return by_graph


def build_common_support(
    records: list[DelineationRecord],
    *,
    selected_prominence_candidate: str,
    exam_summary_threshold: float,
) -> CommonSupportSection:
    """Materialize exact DWT/selected-prominence complete-unit intersection."""
    if not 0 <= exam_summary_threshold <= 1:
        raise ValueError("exam_summary_threshold must lie in [0, 1]")
    dwt = _records_by_graph(records, method=DelineationMethod.DWT, candidate="dwt")
    prominence = _records_by_graph(
        records,
        method=DelineationMethod.PROMINENCE,
        candidate=selected_prominence_candidate,
    )
    if not dwt:
        raise ValueError("no DWT records available for common support")
    if dwt.keys() != prominence.keys():
        missing_dwt = sorted(prominence.keys() - dwt.keys())
        missing_prominence = sorted(dwt.keys() - prominence.keys())
        raise ValueError(
            "DWT and selected prominence graph units differ: "
            f"missing DWT={missing_dwt[:8]}, missing prominence={missing_prominence[:8]}"
        )

    membership: list[CommonSupportUnit] = []
    identity_fields = (
        "exam_idx",
        "exam_id",
        "positive_classes",
        "lead",
        "beat_index",
        "polarity",
    )
    for graph_idx in sorted(dwt):
        dwt_record = dwt[graph_idx]
        prominence_record = prominence[graph_idx]
        if any(
            getattr(dwt_record, field) != getattr(prominence_record, field)
            for field in identity_fields
        ):
            raise ValueError(f"delineation identity differs for graph_idx {graph_idx}")
        common = dwt_record.complete and prominence_record.complete
        membership.append(
            CommonSupportUnit(
                graph_idx=graph_idx,
                exam_idx=dwt_record.exam_idx,
                exam_id=dwt_record.exam_id,
                positive_classes=dwt_record.positive_classes,
                lead=dwt_record.lead,
                beat_index=dwt_record.beat_index,
                polarity=dwt_record.polarity,
                dwt_complete=dwt_record.complete,
                prominence_complete=prominence_record.complete,
                common=common,
            )
        )

    by_exam = _support_by(membership, [[str(unit.exam_idx)] for unit in membership])
    return CommonSupportSection(
        selected_prominence_candidate=selected_prominence_candidate,
        exam_summary_threshold=exam_summary_threshold,
        membership=membership,
        common_graph_indices=[unit.graph_idx for unit in membership if unit.common],
        overall=_support_summary(membership),
        by_class=_support_by(membership, [unit.positive_classes for unit in membership]),
        by_exam=by_exam,
        by_lead=_support_by(membership, [[str(unit.lead)] for unit in membership]),
        by_beat_index=_support_by(membership, [[str(unit.beat_index)] for unit in membership]),
        by_polarity=_support_by(membership, [[str(unit.polarity)] for unit in membership]),
        exams_at_or_above_threshold=sorted(
            int(exam_idx)
            for exam_idx, summary in by_exam.items()
            if summary.common_rate >= exam_summary_threshold
        ),
    )


def _spread_selection(values: list[int], count: int) -> list[int]:
    if len(values) < count:
        raise ValueError(f"need {count} values, only {len(values)} are available")
    if count == 1:
        return [values[len(values) // 2]]
    positions = np.linspace(0, len(values) - 1, num=count).round().astype(int)
    return [values[int(position)] for position in positions]


def select_review_sample(
    records: list[DelineationRecord],
    *,
    class_order: tuple[str, ...],
    required_leads: tuple[int, ...],
    exams_per_class: int,
    beats_per_exam: int,
) -> list[ReviewSampleItem]:
    """Select deterministic unique exams and lead-complete stored beats.

    Selection uses only labels and graph availability from DWT audit records;
    delineation success, model outputs, and later Shapley results are irrelevant.
    Multi-label exams are assigned to the first class that selects them and are
    never duplicated in a later class.
    """
    if exams_per_class <= 0 or beats_per_exam <= 0:
        raise ValueError("review sample sizes must be positive")
    required = set(required_leads)
    if not required:
        raise ValueError("required_leads must not be empty")
    dwt_records = [
        record
        for record in records
        if record.method is DelineationMethod.DWT and record.candidate == "dwt"
    ]
    by_exam: dict[int, list[DelineationRecord]] = defaultdict(list)
    for record in dwt_records:
        by_exam[record.exam_idx].append(record)

    eligible_beats: dict[int, list[int]] = {}
    for exam_idx, exam_records in by_exam.items():
        leads_by_beat: dict[int, set[int]] = defaultdict(set)
        for record in exam_records:
            if record.boundary_type == 0:
                leads_by_beat[record.beat_index].add(record.lead)
        eligible_beats[exam_idx] = sorted(
            beat_index for beat_index, leads in leads_by_beat.items() if required.issubset(leads)
        )

    selected: list[ReviewSampleItem] = []
    used_exams: set[int] = set()
    for class_name in class_order:
        candidates = sorted(
            exam_idx
            for exam_idx, exam_records in by_exam.items()
            if class_name in exam_records[0].positive_classes
            and exam_idx not in used_exams
            and len(eligible_beats[exam_idx]) >= beats_per_exam
        )
        if len(candidates) < exams_per_class:
            raise ValueError(
                f"class {class_name} needs {exams_per_class} unique review exams, "
                f"only {len(candidates)} are eligible"
            )
        for exam_idx in candidates[:exams_per_class]:
            exam_records = by_exam[exam_idx]
            selected.append(
                ReviewSampleItem(
                    class_name=class_name,
                    exam_idx=exam_idx,
                    exam_id=exam_records[0].exam_id,
                    beat_indices=_spread_selection(eligible_beats[exam_idx], beats_per_exam),
                )
            )
            used_exams.add(exam_idx)
    return selected
