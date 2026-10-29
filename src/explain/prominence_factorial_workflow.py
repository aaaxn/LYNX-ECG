"""Exact PR-QRS-T 2^3 factorial analysis on selected-Prominence support.

A lead-beat graph unit enters the analysis exactly when the selected Prominence
candidate has a complete, ordered PR/QRS/T triplet. Fixed legacy, fixed
standardized, and Prominence are evaluated on that same mask. DWT does not
participate in support selection or interventions.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import explain.factorial_aggregation as factorial_aggregation
from config.train_models import load_train_preset
from dataset.schema import decode_vlen_str
from ecg_artifact_io import sha256_file, staged_artifact
from explain.coalition import (
    EIGHT_STATE_ORDER,
    FACTORIAL_TERM_ORDER,
    evaluate_eight_state_coalitions,
)
from explain.coalition_workflow import build_coalition_state_models
from explain.common_support import (
    PROMINENCE_SUPPORT_METHOD_ORDER,
    RegionDefinitionMethod,
    prominence_support_intervals_for_exam,
)
from explain.delineation import DelineationMethod
from explain.delineation_qc import DelineationRecord, ProminenceParameters
from explain.delineation_workflow import DelineationQCReport, load_delineation_qc_report
from explain.ecg_taxonomy import (
    MORPHOLOGY_CLASSES,
    REGION_PAIRS,
    DurationName,
    ECGRegion,
    OutputScale,
)
from explain.ecg_timing import exam_timing
from explain.region_intervention import graph_local_region_masks
from explain.report import (
    ArtifactPathBinding,
    DatasetBinding,
    DatasetFingerprintModel,
    RepresentationBinding,
)
from explain.report_contracts import ReportProducer
from models.hier_model import HierGNNModel
from training.checkpoint import resolve_subgraph_representation, resolve_symmetrize_edges
from training.dataset import ExamGraphDataset
from training.hdf5_meta import HDF5Meta
from training.inference import single_exam_inputs

REPORT_PRODUCER = ReportProducer.PROMINENCE_SUPPORT_FACTORIAL
SCALE_ORDER = factorial_aggregation.SCALE_ORDER
SUPPORT_VIEW_ORDER = (
    "nonzero_prominence_support",
    "prominence_support_rate_at_least_0_5",
)
SupportView = Literal[
    "nonzero_prominence_support",
    "prominence_support_rate_at_least_0_5",
]


class ProminenceFactorialExamResult(BaseModel):
    """One positive exam paired across the three region definitions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    exam_index: Annotated[int, Field(ge=0)]
    exam_id: Annotated[str, Field(min_length=1)]
    graph_indices: list[int]
    prominence_graph_indices: list[int]
    prominence_support_rate: Annotated[float, Field(gt=0, le=1)]
    methods: list[factorial_aggregation.FactorialMethodResult]

    @model_validator(mode="after")
    def _paired_support(self) -> Self:
        if self.graph_indices != sorted(set(self.graph_indices)) or not self.graph_indices:
            raise ValueError("exam graph indices must be non-empty, sorted, and unique")
        if self.prominence_graph_indices != sorted(set(self.prominence_graph_indices)):
            raise ValueError("Prominence graph indices must be sorted and unique")
        if not self.prominence_graph_indices or not set(self.prominence_graph_indices) <= set(
            self.graph_indices
        ):
            raise ValueError("analyzed exams require nonzero Prominence support")
        expected_rate = len(self.prominence_graph_indices) / len(self.graph_indices)
        if self.prominence_support_rate != expected_rate:
            raise ValueError("Prominence support rate disagrees with graph membership")
        if [result.method for result in self.methods] != list(PROMINENCE_SUPPORT_METHOD_ORDER):
            raise ValueError("exam requires the canonical fixed/fixed/Prominence method order")
        if any(result.logit is None or result.probability is None for result in self.methods):
            raise ValueError("Prominence factorial exams require finite eight-state games")
        return self


class ProminenceFactorialClassResult(BaseModel):
    """Factorial exam results for one morphology-positive cohort."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: Annotated[str, Field(min_length=1)]
    class_index: Annotated[int, Field(ge=0)]
    n_positive_exams: Annotated[int, Field(ge=0)]
    n_zero_prominence_support_exams: Annotated[int, Field(ge=0)]
    n_not_evaluated_exams: Annotated[int, Field(ge=0)]
    exams: list[ProminenceFactorialExamResult]

    @model_validator(mode="after")
    def _coherent_counts(self) -> Self:
        if self.n_positive_exams != (
            len(self.exams) + self.n_zero_prominence_support_exams + self.n_not_evaluated_exams
        ):
            raise ValueError("Prominence factorial class counts disagree with exam records")
        indices = [exam.exam_index for exam in self.exams]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("Prominence factorial exams must be sorted and unique")
        return self


class ProminenceFactorialSeedRun(BaseModel):
    """One checkpoint's full-factorial results on Prominence-only support."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    seed: Annotated[int, Field(ge=0)]
    delineation_qc_report: ArtifactPathBinding
    delineation_qc_report_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    training_preset_path: str
    training_preset_sha256: str
    dataset: DatasetBinding
    dataset_hdf5_sha256: str
    representation: RepresentationBinding
    selected_prominence_candidate: Annotated[str, Field(min_length=1)]
    selected_prominence_parameters: ProminenceParameters
    candidate_selection_rationale: Annotated[str, Field(min_length=1)]
    classes: list[ProminenceFactorialClassResult]

    @field_validator(
        "delineation_qc_report_sha256",
        "checkpoint_sha256",
        "training_preset_sha256",
        "dataset_hdf5_sha256",
    )
    @classmethod
    def _sha256_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("source digests must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _canonical_run(self) -> Self:
        for name in ("checkpoint_path", "training_preset_path"):
            value = getattr(self, name)
            if not value or str(Path(value).resolve()) != value:
                raise ValueError(f"{name} must be absolute and canonical")
        names = [result.class_name for result in self.classes]
        if not names or len(names) != len(set(names)):
            raise ValueError("Prominence factorial classes must be non-empty and unique")
        return self


class SegmentMainEffectContrastSummary(BaseModel):
    """Paired contrast and data-derived equivalence-margin bound for two segments.

    The margin is not a post-hoc equivalence claim. It is the smallest symmetric
    margin around zero containing the pointwise 95% bootstrap interval. A
    scientifically chosen margin must be declared independently before using
    this bound in an equivalence decision.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: str
    support_view: SupportView
    method: RegionDefinitionMethod
    scale: OutputScale
    segment_a: DurationName
    segment_b: DurationName
    n_exams: Annotated[int, Field(gt=0)]
    mean_difference_a_minus_b: float
    bootstrap_mean_difference_ci_low: float
    bootstrap_mean_difference_ci_high: float
    minimum_symmetric_equivalence_margin: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def _valid_segment_contrast(self) -> Self:
        if (self.segment_a, self.segment_b) not in REGION_PAIRS:
            raise ValueError("segment contrast requires a canonical ordered pair")
        values = (
            self.mean_difference_a_minus_b,
            self.bootstrap_mean_difference_ci_low,
            self.bootstrap_mean_difference_ci_high,
            self.minimum_symmetric_equivalence_margin,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("segment contrast values must be finite")
        if self.bootstrap_mean_difference_ci_low > self.bootstrap_mean_difference_ci_high:
            raise ValueError("segment contrast confidence interval is not ordered")
        expected_margin = max(
            abs(self.bootstrap_mean_difference_ci_low),
            abs(self.bootstrap_mean_difference_ci_high),
        )
        if self.minimum_symmetric_equivalence_margin != expected_margin:
            raise ValueError("equivalence-margin bound disagrees with the confidence interval")
        return self


class ProminenceSupportFactorialReport(BaseModel):
    """Strict three-seed full-factorial report on Prominence-only support."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    producer: Literal[ReportProducer.PROMINENCE_SUPPORT_FACTORIAL]
    seed_order: list[int]
    runs: list[ProminenceFactorialSeedRun]
    class_order: list[str]
    method_order: list[RegionDefinitionMethod]
    support_view_order: list[SupportView]
    state_order: list[str]
    term_order: list[str]
    scale_order: list[OutputScale]
    support_definition: Literal["selected_prominence_complete_ordered_pr_qrs_t_triplet"]
    no_imputation_policy: Literal["units_outside_prominence_support_remain_intact_for_all_methods"]
    dwt_policy: Literal["not_used_for_support_interventions_or_summaries"]
    candidate_selection_rationale: Annotated[str, Field(min_length=1)]
    factor_coding: Literal["plus_one_region_present_minus_one_region_deleted"]
    effect_definition: Literal["mean_response_at_term_sign_plus_one_minus_term_sign_minus_one"]
    primary_scale: Literal["logit"]
    seed_aggregation: Literal["mean_across_seeds_within_exam_before_across_exam_summary"]
    uncertainty: Literal["deterministic_exam_level_nonparametric_bootstrap_of_the_mean"]
    interval_scope: Literal["pointwise_exploratory_no_multiplicity_adjustment"]
    bootstrap: factorial_aggregation.BootstrapConfig
    summaries: list[factorial_aggregation.EffectDistributionSummary]
    paired_method_contrasts: list[factorial_aggregation.PairedMethodContrastSummary]
    segment_main_effect_contrasts: list[SegmentMainEffectContrastSummary]
    equivalence_interpretation: Literal[
        "minimum_margin_is_descriptive_formal_equivalence_requires_an_external_predeclared_margin"
    ]
    causal_interpretation: Literal["model_dependence_not_physiological_causality"]

    @model_validator(mode="after")
    def _coherent_report(self) -> Self:
        if self.seed_order != [run.seed for run in self.runs]:
            raise ValueError("seed_order disagrees with runs")
        if len(self.seed_order) != 3 or len(set(self.seed_order)) != 3:
            raise ValueError("Prominence factorial report requires exactly three seeds")
        if self.method_order != list(PROMINENCE_SUPPORT_METHOD_ORDER):
            raise ValueError("Prominence factorial report requires canonical method order")
        if any(
            run.candidate_selection_rationale != self.candidate_selection_rationale
            for run in self.runs
        ):
            raise ValueError("candidate selection rationale disagrees with runs")
        if self.support_view_order != list(SUPPORT_VIEW_ORDER):
            raise ValueError("Prominence factorial report requires canonical support views")
        if self.state_order != list(EIGHT_STATE_ORDER):
            raise ValueError("Prominence factorial report requires all eight states")
        if self.term_order != list(FACTORIAL_TERM_ORDER):
            raise ValueError("Prominence factorial report requires canonical factorial terms")
        if self.scale_order != list(SCALE_ORDER):
            raise ValueError("Prominence factorial report requires logit then probability")
        if not self.class_order or any(
            [result.class_name for result in run.classes] != self.class_order for run in self.runs
        ):
            raise ValueError("all runs must share the declared class order")
        expected_summary_keys = [
            (class_name, support_view, method, scale, term)
            for class_name in self.class_order
            for support_view in SUPPORT_VIEW_ORDER
            for method in PROMINENCE_SUPPORT_METHOD_ORDER
            for scale in SCALE_ORDER
            for term in FACTORIAL_TERM_ORDER
        ]
        actual_summary_keys = [
            (summary.class_name, summary.support_view, summary.method, summary.scale, summary.term)
            for summary in self.summaries
        ]
        if actual_summary_keys != expected_summary_keys:
            raise ValueError("factorial summaries require canonical identities and order")
        expected_method_keys = [
            (class_name, support_view, method_a, method_b, scale, term)
            for class_name in self.class_order
            for support_view in SUPPORT_VIEW_ORDER
            for method_a, method_b in combinations(PROMINENCE_SUPPORT_METHOD_ORDER, 2)
            for scale in SCALE_ORDER
            for term in FACTORIAL_TERM_ORDER
        ]
        actual_method_keys = [
            (
                contrast.class_name,
                contrast.support_view,
                contrast.method_a,
                contrast.method_b,
                contrast.scale,
                contrast.term,
            )
            for contrast in self.paired_method_contrasts
        ]
        if actual_method_keys != expected_method_keys:
            raise ValueError("paired method contrasts require canonical identities and order")
        expected_segment_keys = [
            (class_name, support_view, method, scale, segment_a, segment_b)
            for class_name in self.class_order
            for support_view in SUPPORT_VIEW_ORDER
            for method in PROMINENCE_SUPPORT_METHOD_ORDER
            for scale in SCALE_ORDER
            for segment_a, segment_b in REGION_PAIRS
        ]
        actual_segment_keys = [
            (
                contrast.class_name,
                contrast.support_view,
                contrast.method,
                contrast.scale,
                contrast.segment_a,
                contrast.segment_b,
            )
            for contrast in self.segment_main_effect_contrasts
        ]
        if actual_segment_keys != expected_segment_keys:
            raise ValueError("segment contrasts require canonical identities and order")
        return self


def _selected_prominence_records_by_exam(
    h5: h5py.File,
    report: DelineationQCReport,
) -> tuple[np.ndarray, dict[int, list[DelineationRecord]]]:
    """Validate and index only the selected Prominence records.

    DWT records in the QC artifact are deliberately not consulted here.
    """
    if report.exam_scope != "morphology_positive":
        raise ValueError("Prominence factorial analysis requires morphology-positive QC")
    hdf5_path = str(Path(h5.filename).resolve())
    if hdf5_path != report.dataset.path:
        raise ValueError("open HDF5 differs from the delineation QC dataset")
    candidate = report.config.selected_prominence_candidate
    records = [
        record
        for record in report.records
        if record.method is DelineationMethod.PROMINENCE and record.candidate == candidate
    ]
    graph_exam = np.asarray(h5["graph_exam_index"][:], dtype=np.int64)
    labels = np.asarray(h5["exam_labels"][:], dtype=np.int64)
    class_names = [str(name) for name in h5.attrs["exam_label_names"]]
    missing = [name for name in MORPHOLOGY_CLASSES if name not in class_names]
    if missing:
        raise ValueError(f"HDF5 is missing morphology classes {missing}")
    morphology_columns = [class_names.index(name) for name in MORPHOLOGY_CLASSES]
    positive_exams = {
        exam_index
        for exam_index in range(labels.shape[0])
        if any(labels[exam_index, column] == 1 for column in morphology_columns)
    }
    expected_graphs = {
        graph_index
        for graph_index, exam_index in enumerate(graph_exam)
        if int(exam_index) in positive_exams
    }
    graph_indices = [record.graph_idx for record in records]
    if len(graph_indices) != len(set(graph_indices)) or set(graph_indices) != expected_graphs:
        raise ValueError(
            "selected Prominence records must cover every morphology-positive graph exactly once"
        )
    records_by_exam: dict[int, list[DelineationRecord]] = {}
    for record in records:
        records_by_exam.setdefault(record.exam_idx, []).append(record)
    return graph_exam, records_by_exam


def analyze_prominence_factorial_seed(
    model: HierGNNModel,
    dataset: ExamGraphDataset,
    h5: h5py.File,
    delineation_report: DelineationQCReport,
    *,
    delineation_qc_path: Path,
    delineation_qc_report_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_state: dict[str, Any],
    dataset_meta: HDF5Meta,
    dataset_hdf5_sha256: str,
    seed: int,
    device: torch.device,
) -> ProminenceFactorialSeedRun:
    """Evaluate all eight states for three methods on Prominence-only support."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    delineation_qc_path = delineation_qc_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    dataset_path = Path(delineation_report.dataset.path)
    training_preset_path = checkpoint_path.parent / "preset.json"
    expected_sources = {
        delineation_qc_path: delineation_qc_report_sha256,
        checkpoint_path: checkpoint_sha256,
        dataset_path: dataset_hdf5_sha256,
    }
    for path, expected_digest in expected_sources.items():
        if sha256_file(path) != expected_digest:
            raise ValueError(f"source bytes changed before Prominence factorial analysis: {path}")
    training_preset_sha256 = sha256_file(training_preset_path)
    training_preset = load_train_preset(training_preset_path)
    if training_preset.optim.seed != seed:
        raise ValueError(
            f"declared seed {seed} differs from training preset seed {training_preset.optim.seed}"
        )

    fingerprint = DatasetFingerprintModel(**asdict(dataset_meta.fingerprint))
    if (
        delineation_report.dataset.fingerprint != fingerprint
        or list(dataset_meta.class_names) != fingerprint.class_names
    ):
        raise ValueError("checkpoint-compatible dataset differs from delineation QC")
    if list(dataset_meta.class_names) != [str(name) for name in h5.attrs["exam_label_names"]]:
        raise ValueError("HDF5 class order differs from validated dataset metadata")

    graph_exam, records_by_exam = _selected_prominence_records_by_exam(
        h5,
        delineation_report,
    )
    if graph_exam.size == 0 or np.any(graph_exam[1:] < graph_exam[:-1]):
        raise ValueError("graph_exam_index must be non-empty and sorted")
    labels = np.asarray(h5["exam_labels"][:], dtype=np.int64)
    exam_ids = h5["exam_id"][:]
    sampling_rate = int(h5.attrs["sampling_rate"])
    class_names = list(dataset_meta.class_names)
    candidate = delineation_report.config.selected_prominence_candidate

    classes: list[ProminenceFactorialClassResult] = []
    for class_name in MORPHOLOGY_CLASSES:
        class_index = class_names.index(class_name)
        positive_exams = np.flatnonzero(labels[:, class_index] == 1)
        exams: list[ProminenceFactorialExamResult] = []
        n_zero_support = 0
        for raw_exam_index in positive_exams:
            exam_index = int(raw_exam_index)
            graph_start = int(np.searchsorted(graph_exam, exam_index, side="left"))
            graph_end = int(np.searchsorted(graph_exam, exam_index, side="right"))
            graph_indices = tuple(range(graph_start, graph_end))
            timing = exam_timing(h5, exam_index, graph_exam_index=graph_exam)
            interval_set = prominence_support_intervals_for_exam(
                records_by_exam.get(exam_index, []),
                graph_indices=graph_indices,
                beat_r_peak_t=timing.beat_r_peak_t,
                class_name=class_name,
                sampling_rate=sampling_rate,
                selected_prominence_candidate=candidate,
            )
            prominence_graph_indices = [
                graph_index
                for graph_index, available in zip(
                    graph_indices,
                    interval_set.common_mask.tolist(),
                    strict=True,
                )
                if available
            ]
            if not prominence_graph_indices:
                n_zero_support += 1
                continue
            inputs = single_exam_inputs(dataset, exam_index, device)
            method_results: list[factorial_aggregation.FactorialMethodResult] = []
            for method in PROMINENCE_SUPPORT_METHOD_ORDER:
                masks = graph_local_region_masks(
                    timing.node_t,
                    timing.node_beat,
                    interval_set.intervals[method],
                )
                evaluation = evaluate_eight_state_coalitions(
                    model,
                    inputs,
                    masks,
                    target_class=class_index,
                )
                method_results.append(
                    factorial_aggregation.build_factorial_method_result(
                        method,
                        build_coalition_state_models(evaluation),
                    )
                )
            exams.append(
                ProminenceFactorialExamResult(
                    exam_index=exam_index,
                    exam_id=decode_vlen_str(exam_ids[exam_index]),
                    graph_indices=list(graph_indices),
                    prominence_graph_indices=prominence_graph_indices,
                    prominence_support_rate=(len(prominence_graph_indices) / len(graph_indices)),
                    methods=method_results,
                )
            )
        classes.append(
            ProminenceFactorialClassResult(
                class_name=class_name,
                class_index=class_index,
                n_positive_exams=int(positive_exams.size),
                n_zero_prominence_support_exams=n_zero_support,
                n_not_evaluated_exams=0,
                exams=exams,
            )
        )

    for path, expected_digest in expected_sources.items():
        if sha256_file(path) != expected_digest:
            raise ValueError(f"source bytes changed during Prominence factorial analysis: {path}")
    if sha256_file(training_preset_path) != training_preset_sha256:
        raise ValueError("training preset bytes changed during Prominence factorial analysis")
    return ProminenceFactorialSeedRun(
        seed=seed,
        delineation_qc_report=ArtifactPathBinding(path=str(delineation_qc_path)),
        delineation_qc_report_sha256=delineation_qc_report_sha256,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,
        training_preset_path=str(training_preset_path.resolve()),
        training_preset_sha256=training_preset_sha256,
        dataset=DatasetBinding(path=str(dataset_path), fingerprint=fingerprint),
        dataset_hdf5_sha256=dataset_hdf5_sha256,
        representation=RepresentationBinding(
            subgraph_representation=resolve_subgraph_representation(checkpoint_state),
            symmetrize_edges=resolve_symmetrize_edges(checkpoint_state),
        ),
        selected_prominence_candidate=candidate,
        selected_prominence_parameters=(delineation_report.config.selected_prominence_kwargs),
        candidate_selection_rationale=delineation_report.config.selection_rationale,
        classes=classes,
    )


def _validate_aligned_runs(runs: list[ProminenceFactorialSeedRun]) -> None:
    if len(runs) != 3 or len({run.seed for run in runs}) != 3:
        raise ValueError("Prominence factorial aggregation requires exactly three distinct seeds")
    for field in ("checkpoint_sha256", "training_preset_sha256"):
        if len({getattr(run, field) for run in runs}) != len(runs):
            raise ValueError(f"Prominence factorial runs require distinct {field} values")
    reference = runs[0]
    for run in runs:
        if any(
            [method.method for method in exam.methods] != list(PROMINENCE_SUPPORT_METHOD_ORDER)
            for class_result in run.classes
            for exam in class_result.exams
        ):
            raise ValueError("Prominence factorial runs require canonical three-method order")
    for run in runs[1:]:
        if (
            run.delineation_qc_report != reference.delineation_qc_report
            or run.delineation_qc_report_sha256 != reference.delineation_qc_report_sha256
            or run.dataset != reference.dataset
            or run.dataset_hdf5_sha256 != reference.dataset_hdf5_sha256
            or run.representation != reference.representation
            or run.selected_prominence_candidate != reference.selected_prominence_candidate
            or run.selected_prominence_parameters != reference.selected_prominence_parameters
            or run.candidate_selection_rationale != reference.candidate_selection_rationale
        ):
            raise ValueError("Prominence factorial runs bind incompatible analysis inputs")
        if len(run.classes) != len(reference.classes):
            raise ValueError("Prominence factorial runs have different class counts")
        for candidate_class, reference_class in zip(run.classes, reference.classes, strict=True):
            candidate_identity = (
                candidate_class.class_name,
                candidate_class.class_index,
                candidate_class.n_positive_exams,
                candidate_class.n_zero_prominence_support_exams,
                candidate_class.n_not_evaluated_exams,
            )
            reference_identity = (
                reference_class.class_name,
                reference_class.class_index,
                reference_class.n_positive_exams,
                reference_class.n_zero_prominence_support_exams,
                reference_class.n_not_evaluated_exams,
            )
            if candidate_identity != reference_identity or len(candidate_class.exams) != len(
                reference_class.exams
            ):
                raise ValueError("Prominence factorial runs have incompatible class cohorts")
            for candidate_exam, reference_exam in zip(
                candidate_class.exams,
                reference_class.exams,
                strict=True,
            ):
                if (
                    candidate_exam.exam_index != reference_exam.exam_index
                    or candidate_exam.exam_id != reference_exam.exam_id
                    or candidate_exam.graph_indices != reference_exam.graph_indices
                    or candidate_exam.prominence_graph_indices
                    != reference_exam.prominence_graph_indices
                    or candidate_exam.prominence_support_rate
                    != reference_exam.prominence_support_rate
                ):
                    raise ValueError("Prominence factorial runs have incompatible exam support")


_PROMINENCE_SUPPORT_VIEWS = (
    factorial_aggregation.SupportViewSpec(name="nonzero_prominence_support"),
    factorial_aggregation.SupportViewSpec(
        name="prominence_support_rate_at_least_0_5",
        minimum_rate=0.5,
    ),
)


def _aggregation_inputs(
    runs: list[ProminenceFactorialSeedRun],
) -> list[factorial_aggregation.FactorialRunInput]:
    return [
        factorial_aggregation.FactorialRunInput(
            classes=tuple(
                factorial_aggregation.FactorialClassInput(
                    class_name=class_result.class_name,
                    exams=tuple(
                        factorial_aggregation.FactorialExamInput(
                            exam_index=exam.exam_index,
                            support_rate=exam.prominence_support_rate,
                            methods=tuple(exam.methods),
                        )
                        for exam in class_result.exams
                    ),
                )
                for class_result in run.classes
            )
        )
        for run in runs
    ]


def _aggregate_prominence_factorial_runs(
    runs: list[ProminenceFactorialSeedRun],
    bootstrap: factorial_aggregation.BootstrapConfig,
) -> factorial_aggregation.FactorialAggregation:
    _validate_aligned_runs(runs)
    return factorial_aggregation.aggregate_factorial_effects(
        _aggregation_inputs(runs),
        bootstrap,
        support_views=_PROMINENCE_SUPPORT_VIEWS,
        method_order=PROMINENCE_SUPPORT_METHOD_ORDER,
        method_contrast_identity_namespace="method_contrast",
        term_contrast_pairs=REGION_PAIRS,
        term_contrast_identity_namespace="segment_contrast",
    )


def _segment_contrasts(
    aggregated: factorial_aggregation.FactorialAggregation,
) -> list[SegmentMainEffectContrastSummary]:
    return [
        SegmentMainEffectContrastSummary(
            class_name=contrast.class_name,
            support_view=cast(SupportView, contrast.support_view),
            method=contrast.method,
            scale=contrast.scale,
            segment_a=ECGRegion(contrast.term_a),
            segment_b=ECGRegion(contrast.term_b),
            n_exams=contrast.n_exams,
            mean_difference_a_minus_b=contrast.mean_difference_a_minus_b,
            bootstrap_mean_difference_ci_low=contrast.bootstrap_mean_difference_ci_low,
            bootstrap_mean_difference_ci_high=contrast.bootstrap_mean_difference_ci_high,
            minimum_symmetric_equivalence_margin=max(
                abs(contrast.bootstrap_mean_difference_ci_low),
                abs(contrast.bootstrap_mean_difference_ci_high),
            ),
        )
        for contrast in aggregated.term_contrasts
    ]


def _check_prominence_factorial_report_statistics(
    report: ProminenceSupportFactorialReport,
    expected: factorial_aggregation.FactorialAggregation,
) -> None:
    if report.summaries != expected.summaries:
        raise ValueError("factorial summaries disagree with per-exam run values")
    if report.paired_method_contrasts != expected.method_contrasts:
        raise ValueError("paired method contrasts disagree with per-exam run values")
    if report.segment_main_effect_contrasts != _segment_contrasts(expected):
        raise ValueError("segment main-effect contrasts disagree with per-exam run values")


def validate_prominence_factorial_report_integrity(
    report: ProminenceSupportFactorialReport,
) -> None:
    """Explicitly verify every per-game effect and aggregate derived statistic."""
    _check_prominence_factorial_report_statistics(
        report,
        _aggregate_prominence_factorial_runs(report.runs, report.bootstrap),
    )


def build_prominence_factorial_report(
    runs: list[ProminenceFactorialSeedRun],
    *,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 100_000,
) -> ProminenceSupportFactorialReport:
    """Build a strict aggregate report from three aligned checkpoint runs."""
    bootstrap = factorial_aggregation.BootstrapConfig(
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
        confidence_level=0.95,
        resampling_unit="exam",
    )
    aggregated = _aggregate_prominence_factorial_runs(runs, bootstrap)
    segment_contrasts = _segment_contrasts(aggregated)
    report = ProminenceSupportFactorialReport(
        producer=REPORT_PRODUCER,
        seed_order=[run.seed for run in runs],
        runs=runs,
        class_order=[result.class_name for result in runs[0].classes],
        method_order=list(PROMINENCE_SUPPORT_METHOD_ORDER),
        support_view_order=list(SUPPORT_VIEW_ORDER),
        state_order=list(EIGHT_STATE_ORDER),
        term_order=list(FACTORIAL_TERM_ORDER),
        scale_order=list(SCALE_ORDER),
        support_definition="selected_prominence_complete_ordered_pr_qrs_t_triplet",
        no_imputation_policy=("units_outside_prominence_support_remain_intact_for_all_methods"),
        dwt_policy="not_used_for_support_interventions_or_summaries",
        candidate_selection_rationale=runs[0].candidate_selection_rationale,
        factor_coding="plus_one_region_present_minus_one_region_deleted",
        effect_definition="mean_response_at_term_sign_plus_one_minus_term_sign_minus_one",
        primary_scale="logit",
        seed_aggregation="mean_across_seeds_within_exam_before_across_exam_summary",
        uncertainty="deterministic_exam_level_nonparametric_bootstrap_of_the_mean",
        interval_scope="pointwise_exploratory_no_multiplicity_adjustment",
        bootstrap=bootstrap,
        summaries=aggregated.summaries,
        paired_method_contrasts=aggregated.method_contrasts,
        segment_main_effect_contrasts=segment_contrasts,
        equivalence_interpretation=(
            "minimum_margin_is_descriptive_formal_equivalence_requires_an_external_predeclared_margin"
        ),
        causal_interpretation="model_dependence_not_physiological_causality",
    )
    _check_prominence_factorial_report_statistics(report, aggregated)
    return report


def _validate_bound_sources(report: ProminenceSupportFactorialReport) -> None:
    """Revalidate source bytes and Prominence-only support membership."""
    _validate_aligned_runs(report.runs)
    reference = report.runs[0]
    qc_path = Path(reference.delineation_qc_report.path)
    dataset_path = Path(reference.dataset.path)
    if sha256_file(qc_path) != reference.delineation_qc_report_sha256:
        raise ValueError("bound delineation QC bytes changed")
    if sha256_file(dataset_path) != reference.dataset_hdf5_sha256:
        raise ValueError("bound dataset bytes changed")
    delineation = load_delineation_qc_report(qc_path)
    if (
        delineation.dataset.path != reference.dataset.path
        or delineation.dataset.fingerprint != reference.dataset.fingerprint
        or delineation.config.selected_prominence_candidate
        != reference.selected_prominence_candidate
        or delineation.config.selected_prominence_kwargs != reference.selected_prominence_parameters
        or delineation.config.selection_rationale != reference.candidate_selection_rationale
    ):
        raise ValueError("factorial provenance differs from the bound delineation QC")

    for run in report.runs:
        for path, expected_digest in (
            (Path(run.checkpoint_path), run.checkpoint_sha256),
            (Path(run.training_preset_path), run.training_preset_sha256),
        ):
            if sha256_file(path) != expected_digest:
                raise ValueError(f"bound Prominence factorial source bytes changed: {path}")
        preset = load_train_preset(Path(run.training_preset_path))
        if preset.optim.seed != run.seed:
            raise ValueError("factorial seed differs from its bound training preset")

    with h5py.File(dataset_path, "r") as h5:
        graph_exam, records_by_exam = _selected_prominence_records_by_exam(h5, delineation)
        if graph_exam.size == 0 or np.any(graph_exam[1:] < graph_exam[:-1]):
            raise ValueError("graph_exam_index must be non-empty and sorted")
        labels = np.asarray(h5["exam_labels"][:], dtype=np.int64)
        exam_ids = h5["exam_id"][:]
        class_names = [str(name) for name in h5.attrs["exam_label_names"]]
        selected_records_by_graph = {
            record.graph_idx: record for records in records_by_exam.values() for record in records
        }
        for class_result in reference.classes:
            class_index = class_names.index(class_result.class_name)
            positive_exams = [int(index) for index in np.flatnonzero(labels[:, class_index] == 1)]
            expected_exams: list[tuple[int, str, list[int], list[int], float]] = []
            expected_zero = 0
            for exam_index in positive_exams:
                graph_start = int(np.searchsorted(graph_exam, exam_index, side="left"))
                graph_end = int(np.searchsorted(graph_exam, exam_index, side="right"))
                graph_indices = list(range(graph_start, graph_end))
                prominence_indices = [
                    graph_index
                    for graph_index in graph_indices
                    if selected_records_by_graph[graph_index].complete
                ]
                if not prominence_indices:
                    expected_zero += 1
                    continue
                expected_exams.append(
                    (
                        exam_index,
                        decode_vlen_str(exam_ids[exam_index]),
                        graph_indices,
                        prominence_indices,
                        len(prominence_indices) / len(graph_indices),
                    )
                )
            actual_exams = [
                (
                    exam.exam_index,
                    exam.exam_id,
                    exam.graph_indices,
                    exam.prominence_graph_indices,
                    exam.prominence_support_rate,
                )
                for exam in class_result.exams
            ]
            if (
                class_result.class_index != class_index
                or class_result.n_positive_exams != len(positive_exams)
                or class_result.n_zero_prominence_support_exams != expected_zero
                or class_result.n_not_evaluated_exams != 0
                or actual_exams != expected_exams
            ):
                raise ValueError("factorial exam support differs from selected Prominence QC")

    if sha256_file(qc_path) != reference.delineation_qc_report_sha256:
        raise ValueError("bound delineation QC changed during validation")
    if sha256_file(dataset_path) != reference.dataset_hdf5_sha256:
        raise ValueError("bound dataset changed during validation")


def write_prominence_factorial_report(
    path: Path,
    report: ProminenceSupportFactorialReport,
    *,
    overwrite: bool = False,
) -> None:
    """Validate source bindings and atomically publish one JSON report."""
    validated = ProminenceSupportFactorialReport.model_validate(report.model_dump(mode="python"))
    validate_prominence_factorial_report_integrity(validated)
    _validate_bound_sources(validated)
    with staged_artifact(path, overwrite=overwrite) as staged:
        staged.write_text(validated.model_dump_json(indent=2) + "\n")
        staged_report = ProminenceSupportFactorialReport.model_validate_json(staged.read_text())
        if staged_report != validated:
            raise ValueError("serialized factorial report differs from the validated report")
