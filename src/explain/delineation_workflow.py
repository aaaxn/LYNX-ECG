"""Standalone CODE-Test delineation QC workflow and report contract."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from pathlib import Path
from typing import Any, Literal, Self

import h5py
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dataset.loaders import CODE_6CLASS_LABEL_NAMES
from dataset.raw_signal import load_preprocessed_exam_signal
from dataset.schema import decode_vlen_str, require_dataset_contract
from dataset.sources import resolve_source_path
from ecg_artifact_io import staged_artifact
from explain.delineation import (
    BeatDelineation,
    DelineationMethod,
    LeadDelineation,
    delineate_lead,
)
from explain.delineation_qc import (
    CommonSupportSection,
    DelineationQCConfig,
    DelineationRecord,
    FiducialSamples,
    MethodQCSummary,
    ReviewSampleItem,
    SupportGroupSummary,
    build_common_support,
    build_method_summaries,
    select_review_sample,
)
from explain.ecg_taxonomy import MORPHOLOGY_CLASSES, DurationName, ECGRegion
from explain.report import DatasetFingerprintModel
from explain.report_contracts import ReportProducer
from training.hdf5_meta import read_hdf5_meta

REPORT_PRODUCER = ReportProducer.REGION_DELINEATION_QC
ALL_EXAM_COHORT_KEYS = (
    "1dAVb/positive",
    "1dAVb/target_negative",
    "RBBB/positive",
    "RBBB/target_negative",
    "LBBB/positive",
    "LBBB/target_negative",
    "six_class_negative",
)
ExamScope = Literal["morphology_positive", "label_positive", "all_exams"]


class LabelDefinedCohorts(BaseModel):
    """Full-dataset label-only cohorts needed by target-control analyses."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    positive_by_target: dict[str, list[int]]
    target_negative_by_target: dict[str, list[int]]
    six_class_negative: list[int]

    @model_validator(mode="after")
    def _canonical_membership(self) -> Self:
        expected_targets = set(MORPHOLOGY_CLASSES)
        if (
            set(self.positive_by_target) != expected_targets
            or set(self.target_negative_by_target) != expected_targets
        ):
            raise ValueError("label-defined cohorts require all morphology targets")
        for indices in (
            *self.positive_by_target.values(),
            *self.target_negative_by_target.values(),
            self.six_class_negative,
        ):
            if not indices:
                raise ValueError("label-defined control cohorts must not be empty")
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError("label-defined cohort indices must be sorted and unique")
            if any(index < 0 for index in indices):
                raise ValueError("label-defined cohort indices must be non-negative")
        six_negative = set(self.six_class_negative)
        for target in MORPHOLOGY_CLASSES:
            positive = set(self.positive_by_target[target])
            negative = set(self.target_negative_by_target[target])
            if positive & negative:
                raise ValueError("target positive and negative cohorts must be disjoint")
            if not six_negative <= negative:
                raise ValueError("six-class-negative must be a subset of every target-negative")
        return self


class DelineationDatasetBinding(BaseModel):
    """Dataset bytes, semantic identity, and build preprocessing provenance."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str
    raw_source_path: str
    fingerprint: DatasetFingerprintModel
    dataset_source: Literal["codetest", "ptbxl"]
    source_variant: str
    sampling_rate: int
    graph_leads: list[int]
    total_exams: int
    total_graphs: int
    morphology_positive_exam_indices: list[int]
    analyzed_exam_indices: list[int]
    label_defined_cohorts: LabelDefinedCohorts | None

    @field_validator("path", "raw_source_path")
    @classmethod
    def _canonical_path(cls, value: str) -> str:
        if not value or not Path(value).is_absolute() or str(Path(value).resolve()) != value:
            raise ValueError("dataset and raw-source paths must be absolute and canonical")
        return value

    @model_validator(mode="after")
    def _consistent_indices(self) -> Self:
        if self.total_exams <= 0 or self.total_graphs <= 0:
            raise ValueError("delineation dataset must contain exams and graphs")
        for name in ("morphology_positive_exam_indices", "analyzed_exam_indices"):
            indices = getattr(self, name)
            if indices != sorted(indices) or len(indices) != len(set(indices)):
                raise ValueError(f"{name} must be sorted and unique")
            if any(index < 0 or index >= self.total_exams for index in indices):
                raise ValueError(f"{name} contains an exam outside the dataset")
        cohorts = self.label_defined_cohorts
        if cohorts is not None:
            expected = set(range(self.total_exams))
            for target in MORPHOLOGY_CLASSES:
                if (
                    set(cohorts.positive_by_target[target])
                    | set(cohorts.target_negative_by_target[target])
                    != expected
                ):
                    raise ValueError("target label cohorts must partition every dataset exam")
            if any(index >= self.total_exams for index in cohorts.six_class_negative):
                raise ValueError("six-class-negative contains an exam outside the dataset")
        return self


class TechnicalAssessment(BaseModel):
    """Model-output-blind interpretation of technical delineation availability."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    selected_prominence_candidate: str
    selection_rationale: str
    model_outputs_consulted: Literal[False]
    clinical_fiducial_ground_truth: Literal["unavailable"]
    statements: list[str]


class DelineationQCReport(BaseModel):
    """Strict report consumed by later common-support robustness analysis."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    producer: Literal[ReportProducer.REGION_DELINEATION_QC]
    exam_scope: ExamScope
    dataset: DelineationDatasetBinding
    config: DelineationQCConfig
    dependency_versions: dict[str, str]
    records: list[DelineationRecord]
    method_summaries: list[MethodQCSummary]
    common_support: CommonSupportSection
    cohort_method_summaries: dict[str, list[MethodQCSummary]]
    cohort_common_support: dict[str, SupportGroupSummary]
    review_sample: list[ReviewSampleItem]
    technical_assessment: TechnicalAssessment

    @model_validator(mode="after")
    def _consistent_scope(self) -> Self:
        if self.exam_scope == "morphology_positive":
            if self.dataset.analyzed_exam_indices != self.dataset.morphology_positive_exam_indices:
                raise ValueError("morphology-positive scope must analyze exactly that cohort")
            if self.dataset.label_defined_cohorts is not None:
                raise ValueError("morphology-positive scope must not bind control cohorts")
            if self.cohort_method_summaries or self.cohort_common_support:
                raise ValueError("label-defined control summaries require all-exam scope")
        elif self.exam_scope == "label_positive":
            analyzed = set(self.dataset.analyzed_exam_indices)
            if not set(self.dataset.morphology_positive_exam_indices) <= analyzed:
                raise ValueError("label-positive scope must include every morphology positive")
            if self.dataset.label_defined_cohorts is not None:
                raise ValueError("label-positive scope must not bind control cohorts")
            if self.cohort_method_summaries or self.cohort_common_support:
                raise ValueError("label-defined control summaries require all-exam scope")
        else:
            if self.dataset.analyzed_exam_indices != list(range(self.dataset.total_exams)):
                raise ValueError("all-exam scope must analyze every dataset exam")
            if self.dataset.label_defined_cohorts is None:
                raise ValueError("all-exam scope requires label-defined cohorts")
            expected = list(ALL_EXAM_COHORT_KEYS)
            if (
                list(self.cohort_method_summaries) != expected
                or list(self.cohort_common_support) != expected
            ):
                raise ValueError("all-exam cohort summaries require canonical key order")
        return self


@dataclasses.dataclass(frozen=True)
class _DelineationCollection:
    records: list[DelineationRecord]
    dataset: DelineationDatasetBinding


def load_delineation_qc_config(path: Path) -> DelineationQCConfig:
    """Load a strict delineation-QC JSON configuration."""
    try:
        return DelineationQCConfig.model_validate_json(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: invalid delineation QC config: {exc}") from exc


def validate_delineation_report_against_hdf5(report: DelineationQCReport) -> None:
    """Fail closed when report identities or coverage differ from its bound HDF5."""
    dataset_path = Path(report.dataset.path)
    meta = read_hdf5_meta(dataset_path)
    if (
        DatasetFingerprintModel.model_validate(dataclasses.asdict(meta.fingerprint))
        != report.dataset.fingerprint
        or meta.total_exams != report.dataset.total_exams
        or meta.total_graphs != report.dataset.total_graphs
    ):
        raise ValueError("bound HDF5 metadata differs from delineation report")

    with h5py.File(dataset_path, "r") as h5:
        class_names = [str(name) for name in h5.attrs["exam_label_names"]]
        labels = np.asarray(h5["exam_labels"][:], dtype=np.int64)
        positive_exams, classes_by_exam = _positive_exam_indices(labels, class_names)
        if positive_exams != report.dataset.morphology_positive_exam_indices:
            raise ValueError("morphology-positive cohort differs from HDF5")
        if (
            report.exam_scope == "label_positive"
            and _label_positive_exam_indices(labels, class_names)
            != report.dataset.analyzed_exam_indices
        ):
            raise ValueError("label-positive cohort differs from HDF5")
        if report.exam_scope == "all_exams":
            cohorts = _label_defined_cohorts(labels, class_names)
            if cohorts != report.dataset.label_defined_cohorts:
                raise ValueError("label-defined cohorts differ from HDF5")
        if report.dataset.source_variant != str(h5.attrs["source_variant"]):
            raise ValueError("source variant differs from HDF5")
        if report.dataset.sampling_rate != int(h5.attrs["sampling_rate"]):
            raise ValueError("sampling rate differs from HDF5")
        if report.dataset.graph_leads != [int(value) for value in h5.attrs["graph_leads"]]:
            raise ValueError("graph leads differ from HDF5")

        graph_fields = {
            "exam_idx": np.asarray(h5["graph_exam_index"][:], dtype=np.int64),
            "lead": np.asarray(h5["graph_lead"][:], dtype=np.int64),
            "beat_index": np.asarray(h5["graph_beat_index"][:], dtype=np.int64),
            "start_sample": np.asarray(h5["graph_start_sample"][:], dtype=np.int64),
            "end_sample": np.asarray(h5["graph_end_sample"][:], dtype=np.int64),
            "boundary_type": np.asarray(h5["graph_boundary_type"][:], dtype=np.int64),
            "r_peak_sample": np.asarray(h5["graph_r_peak_sample"][:], dtype=np.int64),
            "polarity": np.asarray(h5["graph_rpeak_polarity"][:], dtype=np.int64),
        }
        graph_exam = graph_fields["exam_idx"]
        analyzed_exams = set(report.dataset.analyzed_exam_indices)
        expected_graphs = {
            graph_idx
            for graph_idx, exam_idx in enumerate(graph_exam)
            if int(exam_idx) in analyzed_exams
        }
        expected_pairs = {(DelineationMethod.DWT, "dwt")} | {
            (DelineationMethod.PROMINENCE, candidate)
            for candidate in report.config.prominence_candidates
        }
        for pair in expected_pairs:
            graph_indices = [
                record.graph_idx
                for record in report.records
                if (record.method, record.candidate) == pair
            ]
            unique_graph_indices = set(graph_indices)
            if (
                len(graph_indices) != len(unique_graph_indices)
                or unique_graph_indices != expected_graphs
            ):
                raise ValueError(
                    f"{pair[0].value}/{pair[1]} records differ from HDF5 graph coverage"
                )

        exam_ids = h5["exam_id"][:]
        for record in report.records:
            if not 0 <= record.graph_idx < graph_exam.size:
                raise ValueError("delineation graph_idx is outside the HDF5")
            for field, values in graph_fields.items():
                if getattr(record, field) != int(values[record.graph_idx]):
                    raise ValueError(
                        f"delineation {field} differs from HDF5 for graph_idx {record.graph_idx}"
                    )
            exam_idx = int(graph_exam[record.graph_idx])
            if record.exam_id != decode_vlen_str(exam_ids[exam_idx]):
                raise ValueError(
                    f"delineation exam_id differs from HDF5 for graph_idx {record.graph_idx}"
                )
            if record.positive_classes != classes_by_exam.get(exam_idx, []):
                raise ValueError(
                    f"delineation positive classes differ from HDF5 for graph_idx {record.graph_idx}"
                )


def write_delineation_qc_report(
    path: Path,
    report: DelineationQCReport,
    *,
    overwrite: bool = False,
) -> None:
    """Validate against the bound HDF5 and atomically publish one report."""
    validated = _validate_delineation_qc_report(DelineationQCReport.model_validate(report))
    validate_delineation_report_against_hdf5(validated)
    with staged_artifact(path, overwrite=overwrite) as staged:
        staged.write_text(validated.model_dump_json(indent=2) + "\n")
        _validate_delineation_qc_report(DelineationQCReport.model_validate_json(staged.read_text()))


def load_delineation_qc_report(path: Path) -> DelineationQCReport:
    """Load and fail closed on a wrong, malformed, or source-stale report."""
    try:
        report = _validate_delineation_qc_report(
            DelineationQCReport.model_validate_json(path.read_text())
        )
        validate_delineation_report_against_hdf5(report)
        return report
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {REPORT_PRODUCER} report: {exc}") from exc


def _positive_exam_indices(
    labels: np.ndarray,
    class_names: list[str],
) -> tuple[list[int], dict[int, list[str]]]:
    missing = [name for name in MORPHOLOGY_CLASSES if name not in class_names]
    if missing:
        raise ValueError(f"dataset is missing morphology classes {missing}")
    class_indices = {name: class_names.index(name) for name in MORPHOLOGY_CLASSES}
    positive_classes: dict[int, list[str]] = {}
    for exam_idx in range(labels.shape[0]):
        names = [name for name, index in class_indices.items() if labels[exam_idx, index] == 1]
        if names:
            positive_classes[exam_idx] = names
    return sorted(positive_classes), positive_classes


def _label_positive_exam_indices(labels: np.ndarray, class_names: list[str]) -> list[int]:
    """Exams positive for at least one of the six CODE labels, rhythm labels included."""
    if tuple(class_names) != CODE_6CLASS_LABEL_NAMES:
        raise ValueError("label-positive delineation requires the canonical CODE six-class order")
    return np.flatnonzero(labels.any(axis=1)).tolist()


def _label_defined_cohorts(
    labels: np.ndarray,
    class_names: list[str],
) -> LabelDefinedCohorts:
    if tuple(class_names) != CODE_6CLASS_LABEL_NAMES:
        raise ValueError("all-exam delineation requires the canonical CODE six-class order")
    if (
        labels.ndim != 2
        or labels.shape[1] != len(CODE_6CLASS_LABEL_NAMES)
        or not np.isin(labels, (0, 1)).all()
    ):
        raise ValueError("all-exam delineation requires binary canonical CODE labels")
    return LabelDefinedCohorts(
        positive_by_target={
            target: np.flatnonzero(labels[:, class_names.index(target)] == 1).tolist()
            for target in MORPHOLOGY_CLASSES
        },
        target_negative_by_target={
            target: np.flatnonzero(labels[:, class_names.index(target)] == 0).tolist()
            for target in MORPHOLOGY_CLASSES
        },
        six_class_negative=np.flatnonzero(np.all(labels == 0, axis=1)).tolist(),
    )


def _duration_flags(
    durations_ms: dict[DurationName, float],
    config: DelineationQCConfig,
) -> list[DurationName]:
    flags: list[DurationName] = []
    for name in ECGRegion:
        if name not in durations_ms:
            continue
        bounds = config.duration_flag_ranges_ms[name]
        value = durations_ms[name]
        if value < bounds.minimum or value > bounds.maximum:
            flags.append(name)
    return flags


def _available_region_durations_ms(
    beat: BeatDelineation, *, sampling_rate: int
) -> dict[DurationName, float]:
    """Durations for each available region; availability guarantees the bounds."""
    scale = 1000.0 / sampling_rate
    landmarks = beat.landmarks
    bounds_by_region: dict[ECGRegion, tuple[int | None, int | None]] = {
        ECGRegion.PR: (landmarks.p_onset_sample, landmarks.qrs_onset_sample),
        ECGRegion.QRS: (landmarks.qrs_onset_sample, landmarks.qrs_offset_sample),
        ECGRegion.T: (landmarks.t_onset_sample, landmarks.t_offset_sample),
    }
    durations: dict[DurationName, float] = {}
    for region, available in beat.available_regions.items():
        if not available:
            continue
        start, end = bounds_by_region[region]
        assert start is not None and end is not None
        durations[region] = (end - start) * scale
    return durations


def _records_for_lead(
    result: LeadDelineation,
    *,
    graph_indices: np.ndarray,
    exam_idx: int,
    exam_id: str,
    positive_classes: list[str],
    graph_beat_index: np.ndarray,
    graph_start_sample: np.ndarray,
    graph_end_sample: np.ndarray,
    graph_boundary_type: np.ndarray,
    graph_lead: np.ndarray,
    config: DelineationQCConfig,
) -> list[DelineationRecord]:
    if len(graph_indices) != len(result.beats):
        raise ValueError("graph indices and lead delineation beats are not aligned")
    records: list[DelineationRecord] = []
    for graph_idx_raw, beat in zip(graph_indices, result.beats, strict=True):
        graph_idx = int(graph_idx_raw)
        fiducials = FiducialSamples.model_validate(dataclasses.asdict(beat.landmarks))
        durations = _available_region_durations_ms(beat, sampling_rate=result.sampling_rate)
        flags = _duration_flags(durations, config)
        available = beat.available_regions
        records.append(
            DelineationRecord(
                graph_idx=graph_idx,
                exam_idx=exam_idx,
                exam_id=exam_id,
                positive_classes=list(positive_classes),
                lead=int(graph_lead[graph_idx]),
                beat_index=int(graph_beat_index[graph_idx]),
                start_sample=int(graph_start_sample[graph_idx]),
                end_sample=int(graph_end_sample[graph_idx]),
                boundary_type=int(graph_boundary_type[graph_idx]),
                r_peak_sample=beat.r_peak_sample,
                polarity=result.polarity,
                method=result.method,
                candidate=result.candidate,
                signal_orientation=(
                    "polarity_corrected" if result.polarity_corrected else "original"
                ),
                polarity_corrected=result.polarity_corrected,
                complete=beat.complete,
                available_regions=[region for region in ECGRegion if available[region]],
                reasons=list(beat.reasons),
                fiducials=fiducials,
                durations_ms=durations,
                suspicious_duration_flags=flags,
                warning_messages=list(result.warning_messages),
                error_detail=result.error_detail,
            )
        )
    return records


def collect_delineation_records(
    dataset_path: Path,
    config: DelineationQCConfig,
    *,
    exam_scope: ExamScope = "morphology_positive",
    show_progress: bool = False,
) -> _DelineationCollection:
    """Run DWT and every prominence candidate over the selected exam scope.

    Signals come only from :func:`load_preprocessed_exam_signal`; R anchors come
    only from ``graph_r_peak_sample``. The function reads labels but no model,
    checkpoint, prediction, probability, or Shapley artifact.
    """
    dataset_path = dataset_path.resolve()
    with h5py.File(dataset_path, "r") as h5:
        dataset_config = require_dataset_contract(h5, dataset_path)
        if dataset_config.dataset_source not in ("codetest", "ptbxl"):
            raise ValueError(
                "delineation QC is defined only for CODE-Test and PTB-XL, got "
                f"{dataset_config.dataset_source!r}"
            )
        raw_source_path = resolve_source_path(
            dataset_config.dataset_source, dataset_config.source_variant
        ).resolve()
        class_names = [str(name) for name in h5.attrs["exam_label_names"]]
        labels = np.asarray(h5["exam_labels"][:], dtype=np.uint8)
        positive_exams, classes_by_exam = _positive_exam_indices(labels, class_names)
        if exam_scope == "all_exams":
            analyzed_exams = list(range(labels.shape[0]))
            label_defined_cohorts = _label_defined_cohorts(labels, class_names)
        elif exam_scope == "label_positive":
            analyzed_exams = _label_positive_exam_indices(labels, class_names)
            label_defined_cohorts = None
        elif exam_scope == "morphology_positive":
            analyzed_exams = positive_exams
            label_defined_cohorts = None
        else:
            raise ValueError(f"unsupported exam_scope {exam_scope!r}")

        graph_exam = np.asarray(h5["graph_exam_index"][:], dtype=np.int64)
        graph_lead = np.asarray(h5["graph_lead"][:], dtype=np.int64)
        graph_beat = np.asarray(h5["graph_beat_index"][:], dtype=np.int64)
        graph_start = np.asarray(h5["graph_start_sample"][:], dtype=np.int64)
        graph_end = np.asarray(h5["graph_end_sample"][:], dtype=np.int64)
        graph_rpeak = np.asarray(h5["graph_r_peak_sample"][:], dtype=np.int64)
        graph_boundary = np.asarray(h5["graph_boundary_type"][:], dtype=np.int64)
        graph_polarity = np.asarray(h5["graph_rpeak_polarity"][:], dtype=np.int64)
        graph_leads = [int(lead) for lead in h5.attrs["graph_leads"]]
        sampling_rate = int(h5.attrs["sampling_rate"])

        iterable: Any = analyzed_exams
        if show_progress:
            from tqdm import tqdm

            iterable = tqdm(analyzed_exams, desc="delineation QC", unit="exam")

        records: list[DelineationRecord] = []
        for exam_idx in iterable:
            exam_signal = load_preprocessed_exam_signal(h5, exam_idx)
            if exam_signal.sampling_rate != sampling_rate:
                raise ValueError(
                    f"exam {exam_idx}: reconstructed signal rate "
                    f"{exam_signal.sampling_rate} != graph rate {sampling_rate}"
                )
            graph_lo = int(np.searchsorted(graph_exam, exam_idx, side="left"))
            graph_hi = int(np.searchsorted(graph_exam, exam_idx, side="right"))
            exam_graph_indices = np.arange(graph_lo, graph_hi, dtype=np.int64)
            exam_id = str(exam_signal.exam_id)

            for lead in sorted(np.unique(graph_lead[exam_graph_indices])):
                lead_graph_indices = exam_graph_indices[graph_lead[exam_graph_indices] == lead]
                order = np.argsort(graph_rpeak[lead_graph_indices], kind="stable")
                lead_graph_indices = lead_graph_indices[order]
                r_peaks = graph_rpeak[lead_graph_indices]
                polarities = np.unique(graph_polarity[lead_graph_indices])
                if polarities.size != 1 or int(polarities[0]) not in (-1, 1):
                    raise ValueError(
                        f"exam {exam_idx} lead {int(lead)} has inconsistent stored "
                        f"QRS polarities {polarities.tolist()}"
                    )
                polarity = int(polarities[0])
                signal = exam_signal.signal[int(lead)]

                dwt = delineate_lead(
                    signal,
                    r_peak_samples=r_peaks,
                    sampling_rate=sampling_rate,
                    method=DelineationMethod.DWT,
                    candidate="dwt",
                    polarity=polarity,
                )
                records.extend(
                    _records_for_lead(
                        dwt,
                        graph_indices=lead_graph_indices,
                        exam_idx=exam_idx,
                        exam_id=exam_id,
                        positive_classes=classes_by_exam.get(exam_idx, []),
                        graph_beat_index=graph_beat,
                        graph_start_sample=graph_start,
                        graph_end_sample=graph_end,
                        graph_boundary_type=graph_boundary,
                        graph_lead=graph_lead,
                        config=config,
                    )
                )

                for candidate, parameters in config.prominence_candidates.items():
                    prominence = delineate_lead(
                        signal,
                        r_peak_samples=r_peaks,
                        sampling_rate=sampling_rate,
                        method=DelineationMethod.PROMINENCE,
                        candidate=candidate,
                        polarity=polarity,
                        prominence_kwargs=parameters.model_dump(),
                    )
                    records.extend(
                        _records_for_lead(
                            prominence,
                            graph_indices=lead_graph_indices,
                            exam_idx=exam_idx,
                            exam_id=exam_id,
                            positive_classes=classes_by_exam.get(exam_idx, []),
                            graph_beat_index=graph_beat,
                            graph_start_sample=graph_start,
                            graph_end_sample=graph_end,
                            graph_boundary_type=graph_boundary,
                            graph_lead=graph_lead,
                            config=config,
                        )
                    )

        meta = read_hdf5_meta(dataset_path)
        dataset_binding = DelineationDatasetBinding(
            path=str(dataset_path),
            raw_source_path=str(raw_source_path),
            fingerprint=DatasetFingerprintModel.model_validate(
                dataclasses.asdict(meta.fingerprint)
            ),
            dataset_source=dataset_config.dataset_source,
            source_variant=str(h5.attrs["source_variant"]),
            sampling_rate=int(h5.attrs["sampling_rate"]),
            graph_leads=graph_leads,
            total_exams=meta.total_exams,
            total_graphs=meta.total_graphs,
            morphology_positive_exam_indices=positive_exams,
            analyzed_exam_indices=analyzed_exams,
            label_defined_cohorts=label_defined_cohorts,
        )
        return _DelineationCollection(records=records, dataset=dataset_binding)


def _dependency_versions() -> dict[str, str]:
    packages = ("lynx-ecg", "neurokit2", "numpy", "scipy")
    return {name: importlib.metadata.version(name) for name in packages}


def _technical_assessment(
    summaries: list[MethodQCSummary],
    common_support: CommonSupportSection,
    config: DelineationQCConfig,
) -> TechnicalAssessment:
    statements = [
        (
            f"{summary.method.value}/{summary.candidate}: "
            f"{summary.overall.n_complete}/{summary.overall.n_units} complete "
            f"({summary.overall.completeness_rate:.3%})."
        )
        for summary in summaries
    ]
    statements.append(
        f"DWT/selected-prominence common support: "
        f"{common_support.overall.n_common}/{common_support.overall.n_units} "
        f"({common_support.overall.common_rate:.3%})."
    )
    statements.extend(
        [
            (
                "Completeness and duration plausibility are technical availability checks, "
                "not accuracy against clinical fiducial annotations."
            ),
            (
                "The labeled overlays are a secondary, non-blocking review aid; no clinical "
                "fiducial ground truth was available."
            ),
        ]
    )
    return TechnicalAssessment(
        selected_prominence_candidate=config.selected_prominence_candidate,
        selection_rationale=config.selection_rationale,
        model_outputs_consulted=False,
        clinical_fiducial_ground_truth="unavailable",
        statements=statements,
    )


def _cohort_exam_indices(cohorts: LabelDefinedCohorts) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for target in MORPHOLOGY_CLASSES:
        result[f"{target}/positive"] = set(cohorts.positive_by_target[target])
        result[f"{target}/target_negative"] = set(cohorts.target_negative_by_target[target])
    result["six_class_negative"] = set(cohorts.six_class_negative)
    return result


def _cohort_method_summaries(
    records: list[DelineationRecord],
    cohorts: LabelDefinedCohorts | None,
) -> dict[str, list[MethodQCSummary]]:
    if cohorts is None:
        return {}
    return {
        key: build_method_summaries(
            [record for record in records if record.exam_idx in exam_indices]
        )
        for key, exam_indices in _cohort_exam_indices(cohorts).items()
    }


def _cohort_support_summaries(
    support: CommonSupportSection,
    cohorts: LabelDefinedCohorts | None,
) -> dict[str, SupportGroupSummary]:
    if cohorts is None:
        return {}
    summaries: dict[str, SupportGroupSummary] = {}
    for key, exam_indices in _cohort_exam_indices(cohorts).items():
        units = [unit for unit in support.membership if unit.exam_idx in exam_indices]
        if not units:
            raise ValueError(f"cannot summarize empty delineation cohort {key!r}")
        n_common = sum(unit.common for unit in units)
        summaries[key] = SupportGroupSummary(
            n_units=len(units),
            n_common=n_common,
            common_rate=n_common / len(units),
        )
    return summaries


def _record_unit_identity(record: DelineationRecord) -> tuple[Any, ...]:
    return (
        record.exam_idx,
        record.exam_id,
        tuple(record.positive_classes),
        record.lead,
        record.beat_index,
        record.start_sample,
        record.end_sample,
        record.boundary_type,
        record.r_peak_sample,
        record.polarity,
    )


def _validate_delineation_records(report: DelineationQCReport) -> None:
    expected_pairs = {(DelineationMethod.DWT, "dwt")} | {
        (DelineationMethod.PROMINENCE, candidate)
        for candidate in report.config.prominence_candidates
    }
    records_by_pair: dict[tuple[DelineationMethod, str], list[DelineationRecord]] = {}
    for record in report.records:
        records_by_pair.setdefault((record.method, record.candidate), []).append(record)
    if set(records_by_pair) != expected_pairs:
        raise ValueError("method/candidate coverage is inconsistent with delineation config")

    reference_records = records_by_pair[(DelineationMethod.DWT, "dwt")]
    reference = {record.graph_idx: _record_unit_identity(record) for record in reference_records}
    if len(reference) != len(reference_records):
        raise ValueError("method/candidate coverage contains duplicate graph units")
    if {record.exam_idx for record in reference_records} != set(
        report.dataset.analyzed_exam_indices
    ):
        raise ValueError("DWT exam coverage disagrees with the declared analysis scope")
    for pair, records in records_by_pair.items():
        units = {record.graph_idx: _record_unit_identity(record) for record in records}
        if len(units) != len(records) or units != reference:
            raise ValueError(f"method/candidate coverage differs for {pair[0].value}/{pair[1]}")

    sampling_rate = report.dataset.sampling_rate
    if sampling_rate <= 0 or sampling_rate != report.dataset.fingerprint.sampling_rate:
        raise ValueError("delineation dataset sampling rate is inconsistent")
    scale = 1000.0 / sampling_rate
    for record in report.records:
        if not record.complete:
            continue
        p_onset, qrs_onset, qrs_offset, t_onset, t_offset = record.fiducials.complete_triplet()
        expected_durations: dict[DurationName, float] = {
            ECGRegion.PR: (qrs_onset - p_onset) * scale,
            ECGRegion.QRS: (qrs_offset - qrs_onset) * scale,
            ECGRegion.T: (t_offset - t_onset) * scale,
        }
        if any(
            not np.isclose(
                record.durations_ms[name],
                expected_durations[name],
                rtol=1e-9,
                atol=1e-9,
            )
            for name in ECGRegion
        ):
            raise ValueError(f"record {record.graph_idx} durations disagree with fiducials")
        if record.suspicious_duration_flags != _duration_flags(expected_durations, report.config):
            raise ValueError(f"record {record.graph_idx} duration flags disagree with config")


def _validate_delineation_qc_report(report: DelineationQCReport) -> DelineationQCReport:
    """Require every stored section to match its records, dataset, and config."""
    _validate_delineation_records(report)
    summaries = build_method_summaries(report.records)
    if report.method_summaries != summaries:
        raise ValueError("method summaries are inconsistent with delineation records")

    support = build_common_support(
        report.records,
        selected_prominence_candidate=report.config.selected_prominence_candidate,
        exam_summary_threshold=report.config.common_support_exam_summary_threshold,
    )
    if report.common_support != support:
        raise ValueError("common support is inconsistent with delineation records")

    expected_cohort_methods = _cohort_method_summaries(
        report.records,
        report.dataset.label_defined_cohorts,
    )
    if report.cohort_method_summaries != expected_cohort_methods:
        raise ValueError("cohort method summaries are inconsistent with delineation records")
    expected_cohort_support = _cohort_support_summaries(
        support,
        report.dataset.label_defined_cohorts,
    )
    if report.cohort_common_support != expected_cohort_support:
        raise ValueError("cohort common support is inconsistent with delineation records")

    review = select_review_sample(
        report.records,
        class_order=MORPHOLOGY_CLASSES,
        required_leads=tuple(report.dataset.graph_leads),
        exams_per_class=report.config.review_sample.exams_per_class,
        beats_per_exam=report.config.review_sample.beats_per_exam,
    )
    if report.review_sample != review:
        raise ValueError("review sample is inconsistent with delineation records and config")

    assessment = _technical_assessment(summaries, support, report.config)
    if report.technical_assessment != assessment:
        raise ValueError("technical assessment is inconsistent with delineation results")
    return report


def _assemble_delineation_qc_report(
    config: DelineationQCConfig,
    collection: _DelineationCollection,
    *,
    exam_scope: ExamScope = "morphology_positive",
) -> DelineationQCReport:
    records = collection.records
    summaries = build_method_summaries(records)
    common_support = build_common_support(
        records,
        selected_prominence_candidate=config.selected_prominence_candidate,
        exam_summary_threshold=config.common_support_exam_summary_threshold,
    )
    review_sample = select_review_sample(
        records,
        class_order=MORPHOLOGY_CLASSES,
        required_leads=tuple(collection.dataset.graph_leads),
        exams_per_class=config.review_sample.exams_per_class,
        beats_per_exam=config.review_sample.beats_per_exam,
    )
    return DelineationQCReport(
        producer=REPORT_PRODUCER,
        exam_scope=exam_scope,
        dataset=collection.dataset,
        config=config,
        dependency_versions=_dependency_versions(),
        records=records,
        method_summaries=summaries,
        common_support=common_support,
        cohort_method_summaries=_cohort_method_summaries(
            records,
            collection.dataset.label_defined_cohorts,
        ),
        cohort_common_support=_cohort_support_summaries(
            common_support,
            collection.dataset.label_defined_cohorts,
        ),
        review_sample=review_sample,
        technical_assessment=_technical_assessment(summaries, common_support, config),
    )


def build_delineation_qc_report(
    dataset_path: Path,
    config: DelineationQCConfig,
    *,
    exam_scope: ExamScope = "morphology_positive",
    show_progress: bool = False,
) -> DelineationQCReport:
    """Execute every delineator candidate and assemble a strict full-cohort report."""
    collection = collect_delineation_records(
        dataset_path.resolve(),
        config,
        exam_scope=exam_scope,
        show_progress=show_progress,
    )
    return _assemble_delineation_qc_report(config, collection, exam_scope=exam_scope)
