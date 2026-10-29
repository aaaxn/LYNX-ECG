"""All-exam Prominence PR-QRS-T Shapley under two availability rules.

For every CODE-Test exam, the eight structural-deletion states are evaluated
once per availability rule, and the three morphology-class outputs are read
from the same forward pass because Prominence intervals are class-independent.
``beat_complete`` is all-or-nothing support (a beat needs all three regions);
``per_region`` lets a beat missing only its T contribute its PR and its QRS.
The two rules attenuate in opposite directions, so paired reports bracket the
region effects from both sides.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, Self

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from config.train_models import load_train_preset
from dataset.schema import decode_vlen_str
from ecg_artifact_io import sha256_file, staged_artifact
from explain.coalition import (
    EIGHT_STATE_ORDER,
    evaluate_eight_state_coalitions_multiclass,
    exact_regional_shapley,
)
from explain.coalition_workflow import CoalitionStateModel, build_coalition_state_models
from explain.common_support import AvailabilityRule, prominence_intervals_for_exam
from explain.delineation import DelineationMethod
from explain.delineation_qc import DelineationRecord, ProminenceParameters
from explain.delineation_workflow import DelineationQCReport, LabelDefinedCohorts
from explain.ecg_taxonomy import MORPHOLOGY_CLASSES, REGION_ORDER, REGION_PAIR_STATE_ORDER
from explain.ecg_timing import exam_timing
from explain.region_intervention import graph_local_region_masks
from explain.report import (
    ArtifactPathBinding,
    DatasetBinding,
    DatasetFingerprintModel,
    RepresentationBinding,
)
from explain.report_contracts import ReportProducer, SharedReportValue
from models.hier_model import HierGNNModel
from training.checkpoint import resolve_subgraph_representation, resolve_symmetrize_edges
from training.dataset import ExamGraphDataset
from training.hdf5_meta import HDF5Meta
from training.inference import single_exam_inputs

AVAILABILITY_RULE_ORDER: tuple[AvailabilityRule, ...] = (
    AvailabilityRule.BEAT_COMPLETE,
    AvailabilityRule.PER_REGION,
)


class ControlShapleyScaleModel(BaseModel):
    """Exact three-region Shapley values on one model-output scale."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    contributions: dict[str, float]
    joint_drop: float
    efficiency_error: float
    pair_interactions: dict[str, float]

    @model_validator(mode="after")
    def _consistent_allocation(self) -> Self:
        if set(self.contributions) != set(REGION_ORDER):
            raise ValueError("Shapley contributions require exactly PR, QRS, and T")
        if set(self.pair_interactions) != set(REGION_PAIR_STATE_ORDER):
            raise ValueError("pair interactions require exactly the three canonical pairs")
        values = [
            *self.contributions.values(),
            self.joint_drop,
            self.efficiency_error,
            *self.pair_interactions.values(),
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Shapley outputs must be finite")
        if not math.isclose(
            sum(self.contributions.values()),
            self.joint_drop,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("Shapley efficiency check failed")
        expected_error = sum(self.contributions.values()) - self.joint_drop
        if not math.isclose(self.efficiency_error, expected_error, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("reported Shapley efficiency error is inconsistent")
        return self


class ClassShapleyModel(BaseModel):
    """Exact Shapley allocations for one class, or the reason they are absent."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: str
    probability: ControlShapleyScaleModel | None
    logit: ControlShapleyScaleModel | None

    @model_validator(mode="after")
    def _paired_scales(self) -> Self:
        if self.class_name not in MORPHOLOGY_CLASSES:
            raise ValueError(f"unknown morphology class {self.class_name!r}")
        if (self.probability is None) != (self.logit is None):
            raise ValueError("probability and logit Shapley must be present together")
        return self


class ClassStateScoreModel(BaseModel):
    """One class's output for one coalition state."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: str
    probability: float | None
    logit: float | None

    @model_validator(mode="after")
    def _paired(self) -> Self:
        if self.class_name not in MORPHOLOGY_CLASSES:
            raise ValueError(f"unknown morphology class {self.class_name!r}")
        if (self.probability is None) != (self.logit is None):
            raise ValueError("probability and logit must be present together")
        return self


class MulticlassStateModel(BaseModel):
    """One deletion state's structural audit and all class outputs.

    ``structural`` reuses :class:`CoalitionStateModel`: its removed-node and
    cascade counts are class-independent, but its ``probability``/``logit``
    mirror the first morphology class only. Read per-class scores from
    ``class_scores``.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    structural: CoalitionStateModel
    class_scores: list[ClassStateScoreModel]

    @model_validator(mode="after")
    def _canonical_classes(self) -> Self:
        if [score.class_name for score in self.class_scores] != list(MORPHOLOGY_CLASSES):
            raise ValueError("class scores must follow the canonical morphology order")
        return self


class AvailabilityRuleResult(BaseModel):
    """One exam's eight-state game under one availability rule."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    availability_rule: AvailabilityRule
    region_available_units: dict[str, int]
    beat_complete_units: Annotated[int, Field(ge=0)]
    states: list[MulticlassStateModel]
    shapley: list[ClassShapleyModel]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.region_available_units) != set(REGION_ORDER):
            raise ValueError("region availability requires exactly PR, QRS, and T counts")
        if any(count < 0 for count in self.region_available_units.values()):
            raise ValueError("region availability counts must be non-negative")
        if [state.structural.state for state in self.states] != list(EIGHT_STATE_ORDER):
            raise ValueError("states must follow the canonical eight-state order")
        if [value.class_name for value in self.shapley] != list(MORPHOLOGY_CLASSES):
            raise ValueError("Shapley entries must follow the canonical morphology order")
        for shapley in self.shapley:
            class_defined = all(
                score.probability is not None
                for state in self.states
                for score in state.class_scores
                if score.class_name == shapley.class_name
            )
            if class_defined != (shapley.probability is not None):
                raise ValueError(
                    "Shapley must be present exactly when all eight states are defined"
                )
        if self.availability_rule is AvailabilityRule.BEAT_COMPLETE:
            if any(
                count != self.beat_complete_units for count in self.region_available_units.values()
            ):
                raise ValueError("beat-complete availability must be equal for all regions")
        elif any(
            count < self.beat_complete_units for count in self.region_available_units.values()
        ):
            raise ValueError("per-region availability cannot fall below beat completeness")
        return self


DECISION_THRESHOLD = 0.5


class ClassVerdict(NamedTuple):
    """One class's verdict under one availability rule.

    ``contributions`` and ``winner`` are ``None`` when the eight-state game is
    undefined for the class. ``correct`` conditions the intact probability on
    :data:`DECISION_THRESHOLD`.
    """

    intact_probability: float | None
    correct: bool
    contributions: dict[str, float] | None
    winner: str | None


def class_verdict(rule_result: AvailabilityRuleResult, class_name: str) -> ClassVerdict:
    """Read one class's winner and correctness off one availability-rule result."""
    shapley = next(entry for entry in rule_result.shapley if entry.class_name == class_name)
    intact_state = next(state for state in rule_result.states if state.structural.state == "intact")
    intact = next(score for score in intact_state.class_scores if score.class_name == class_name)
    correct = intact.probability is not None and intact.probability >= DECISION_THRESHOLD
    if shapley.logit is None:
        return ClassVerdict(intact.probability, correct, None, None)
    contributions = dict(shapley.logit.contributions)
    winner = max(contributions, key=contributions.__getitem__)
    return ClassVerdict(intact.probability, correct, contributions, winner)


class ProminenceShapleyExamResult(BaseModel):
    """All availability rules for one exam."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    exam_index: Annotated[int, Field(ge=0)]
    exam_id: Annotated[str, Field(min_length=1)]
    positive_classes: list[str]
    n_units: Annotated[int, Field(gt=0)]
    rules: list[AvailabilityRuleResult]

    @model_validator(mode="after")
    def _canonical_rules(self) -> Self:
        if len(set(self.positive_classes)) != len(self.positive_classes):
            raise ValueError("positive_classes must not contain duplicates")
        if [rule.availability_rule for rule in self.rules] != list(AVAILABILITY_RULE_ORDER):
            raise ValueError("rules must follow the canonical availability order")
        for rule in self.rules:
            if any(count > self.n_units for count in rule.region_available_units.values()):
                raise ValueError("region availability cannot exceed the exam's unit count")
        return self

    def rule_result(self, rule: AvailabilityRule) -> AvailabilityRuleResult:
        """Return this exam's result under one availability rule."""
        return next(result for result in self.rules if result.availability_rule is rule)


class ProminenceAvailabilityShapleyReport(BaseModel):
    """One seed's all-exam Prominence Shapley report under both rules."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    producer: Literal[ReportProducer.PROMINENCE_AVAILABILITY_SHAPLEY]
    seed: Annotated[int, Field(ge=0)]
    checkpoint_path: str
    checkpoint_sha256: str
    training_preset_path: str
    training_preset_sha256: str
    dataset: DatasetBinding
    dataset_hdf5_sha256: str
    delineation_qc_report: ArtifactPathBinding
    delineation_qc_report_sha256: str
    representation: RepresentationBinding
    class_order: list[str]
    target_classes: list[str]
    availability_rule_order: list[AvailabilityRule]
    state_order: list[str]
    game: Literal[SharedReportValue.EXACT_EIGHT_STATE_DELETION]
    effect_definition: Literal[SharedReportValue.SCORE_DROP]
    primary_scale: Literal["logit"]
    selected_prominence_candidate: Annotated[str, Field(min_length=1)]
    selected_prominence_parameters: ProminenceParameters
    support_definition: Literal["selected_prominence_landmarks_per_availability_rule"]
    no_imputation_policy: Literal["units_outside_each_rule_support_remain_intact"]
    availability_bias_note: Literal[
        "beat_complete_dilutes_all_regions_of_unmarked_beats_per_region_deletes_t_from_fewer_beats"
    ]
    n_dataset_exams: Annotated[int, Field(gt=0)]
    label_defined_cohorts: LabelDefinedCohorts
    exams: list[ProminenceShapleyExamResult]

    @field_validator(
        "checkpoint_sha256",
        "training_preset_sha256",
        "dataset_hdf5_sha256",
        "delineation_qc_report_sha256",
    )
    @classmethod
    def _sha256_digest(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("source digests must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _consistent_report(self) -> Self:
        if self.class_order != self.dataset.fingerprint.class_names:
            raise ValueError("class_order must match the dataset fingerprint")
        if self.target_classes != list(MORPHOLOGY_CLASSES):
            raise ValueError("target classes must be the canonical morphology classes")
        if self.availability_rule_order != list(AVAILABILITY_RULE_ORDER):
            raise ValueError("availability rules must follow the canonical order")
        if self.state_order != list(EIGHT_STATE_ORDER):
            raise ValueError("state_order must be the canonical eight states")
        indices = [exam.exam_index for exam in self.exams]
        if indices != list(range(self.n_dataset_exams)):
            raise ValueError("exams must cover every dataset exam exactly once, in order")
        return self


def _prominence_records_by_exam(
    h5: h5py.File,
    report: DelineationQCReport,
) -> tuple[np.ndarray, dict[int, list[DelineationRecord]]]:
    """Validate and index the selected Prominence records over every exam."""
    if report.exam_scope != "all_exams":
        raise ValueError("availability Shapley requires the all-exam delineation QC")
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
    graph_indices = [record.graph_idx for record in records]
    if len(graph_indices) != len(set(graph_indices)) or set(graph_indices) != set(
        range(graph_exam.size)
    ):
        raise ValueError("selected Prominence records must cover every graph unit exactly once")
    records_by_exam: dict[int, list[DelineationRecord]] = {}
    for record in records:
        records_by_exam.setdefault(record.exam_idx, []).append(record)
    return graph_exam, records_by_exam


def analyze_prominence_availability_seed(
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
    show_progress: bool = False,
) -> ProminenceAvailabilityShapleyReport:
    """Evaluate both availability rules over every exam for one checkpoint."""
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
            raise ValueError(f"source bytes changed before availability analysis: {path}")
    training_preset_sha256 = sha256_file(training_preset_path)
    training_preset = load_train_preset(training_preset_path)
    if training_preset.optim.seed != seed:
        raise ValueError(
            f"declared seed {seed} differs from training preset seed {training_preset.optim.seed}"
        )
    fingerprint = DatasetFingerprintModel(**asdict(dataset_meta.fingerprint))
    if delineation_report.dataset.fingerprint != fingerprint:
        raise ValueError("checkpoint-compatible dataset differs from delineation QC")
    class_names = list(dataset_meta.class_names)
    if class_names != [str(name) for name in h5.attrs["exam_label_names"]]:
        raise ValueError("HDF5 class order differs from validated dataset metadata")
    cohorts = delineation_report.dataset.label_defined_cohorts
    if cohorts is None:
        raise ValueError("all-exam delineation QC must bind label-defined cohorts")

    graph_exam, records_by_exam = _prominence_records_by_exam(h5, delineation_report)
    labels = np.asarray(h5["exam_labels"][:], dtype=np.int64)
    exam_ids = h5["exam_id"][:]
    sampling_rate = int(h5.attrs["sampling_rate"])
    candidate = delineation_report.config.selected_prominence_candidate
    class_indices = tuple(class_names.index(name) for name in MORPHOLOGY_CLASSES)

    total_exams = int(labels.shape[0])
    exam_iterable: Any = range(total_exams)
    if show_progress:
        from tqdm import tqdm

        exam_iterable = tqdm(exam_iterable, desc=f"seed {seed}", unit="exam")

    exams: list[ProminenceShapleyExamResult] = []
    for exam_index in exam_iterable:
        graph_start = int(np.searchsorted(graph_exam, exam_index, side="left"))
        graph_end = int(np.searchsorted(graph_exam, exam_index, side="right"))
        graph_indices = tuple(range(graph_start, graph_end))
        timing = exam_timing(h5, exam_index, graph_exam_index=graph_exam)
        inputs = single_exam_inputs(dataset, exam_index, device)
        rule_results: list[AvailabilityRuleResult] = []
        for rule in AVAILABILITY_RULE_ORDER:
            intervals = prominence_intervals_for_exam(
                records_by_exam.get(exam_index, []),
                graph_indices=graph_indices,
                beat_r_peak_t=timing.beat_r_peak_t,
                sampling_rate=sampling_rate,
                selected_prominence_candidate=candidate,
                availability_rule=rule,
            )
            masks = graph_local_region_masks(timing.node_t, timing.node_beat, intervals)
            evaluations = evaluate_eight_state_coalitions_multiclass(
                model,
                inputs,
                masks,
                target_classes=class_indices,
            )
            structural_states = build_coalition_state_models(evaluations[class_indices[0]])
            states: list[MulticlassStateModel] = []
            for state_position, structural in enumerate(structural_states):
                class_scores: list[ClassStateScoreModel] = []
                for class_name, class_index in zip(MORPHOLOGY_CLASSES, class_indices, strict=True):
                    result = evaluations[class_index].states[state_position]
                    class_scores.append(
                        ClassStateScoreModel(
                            class_name=class_name,
                            probability=None if result.score is None else result.score.probability,
                            logit=None if result.score is None else result.score.logit,
                        )
                    )
                states.append(
                    MulticlassStateModel(structural=structural, class_scores=class_scores)
                )
            shapley_entries: list[ClassShapleyModel] = []
            for class_name, class_index in zip(MORPHOLOGY_CLASSES, class_indices, strict=True):
                evaluation = evaluations[class_index]
                if any(state.score is None for state in evaluation.states):
                    shapley_entries.append(
                        ClassShapleyModel(class_name=class_name, probability=None, logit=None)
                    )
                    continue
                scale_models: dict[str, ControlShapleyScaleModel] = {}
                for scale in ("probability", "logit"):
                    allocation = exact_regional_shapley(evaluation.score_values(scale))
                    scale_models[scale] = ControlShapleyScaleModel(
                        contributions=allocation.contributions,
                        joint_drop=allocation.joint_drop,
                        efficiency_error=allocation.efficiency_error,
                        pair_interactions=allocation.pair_interactions,
                    )
                shapley_entries.append(
                    ClassShapleyModel(
                        class_name=class_name,
                        probability=scale_models["probability"],
                        logit=scale_models["logit"],
                    )
                )
            available = intervals.available
            rule_results.append(
                AvailabilityRuleResult(
                    availability_rule=rule,
                    region_available_units={
                        region: int(available[:, region_index].sum().item())
                        for region_index, region in enumerate(REGION_ORDER)
                    },
                    beat_complete_units=int(available.all(dim=1).sum().item()),
                    states=states,
                    shapley=shapley_entries,
                )
            )
        exams.append(
            ProminenceShapleyExamResult(
                exam_index=exam_index,
                exam_id=decode_vlen_str(exam_ids[exam_index]),
                positive_classes=[
                    name
                    for name, class_index in zip(MORPHOLOGY_CLASSES, class_indices, strict=True)
                    if labels[exam_index, class_index] == 1
                ],
                n_units=len(graph_indices),
                rules=rule_results,
            )
        )

    representation = RepresentationBinding(
        subgraph_representation=resolve_subgraph_representation(checkpoint_state),
        symmetrize_edges=resolve_symmetrize_edges(checkpoint_state),
    )
    return ProminenceAvailabilityShapleyReport(
        producer=ReportProducer.PROMINENCE_AVAILABILITY_SHAPLEY,
        seed=seed,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,
        training_preset_path=str(training_preset_path.resolve()),
        training_preset_sha256=training_preset_sha256,
        dataset=DatasetBinding(path=str(dataset_path), fingerprint=fingerprint),
        dataset_hdf5_sha256=dataset_hdf5_sha256,
        delineation_qc_report=ArtifactPathBinding(path=str(delineation_qc_path)),
        delineation_qc_report_sha256=delineation_qc_report_sha256,
        representation=representation,
        class_order=class_names,
        target_classes=list(MORPHOLOGY_CLASSES),
        availability_rule_order=list(AVAILABILITY_RULE_ORDER),
        state_order=list(EIGHT_STATE_ORDER),
        game=SharedReportValue.EXACT_EIGHT_STATE_DELETION,
        effect_definition=SharedReportValue.SCORE_DROP,
        primary_scale="logit",
        selected_prominence_candidate=candidate,
        selected_prominence_parameters=delineation_report.config.selected_prominence_kwargs,
        support_definition="selected_prominence_landmarks_per_availability_rule",
        no_imputation_policy="units_outside_each_rule_support_remain_intact",
        availability_bias_note=(
            "beat_complete_dilutes_all_regions_of_unmarked_beats_"
            "per_region_deletes_t_from_fewer_beats"
        ),
        n_dataset_exams=total_exams,
        label_defined_cohorts=cohorts,
        exams=exams,
    )


def write_prominence_availability_report(
    path: Path,
    report: ProminenceAvailabilityShapleyReport,
    *,
    overwrite: bool = False,
) -> None:
    with staged_artifact(path, overwrite=overwrite) as staged:
        staged.write_text(report.model_dump_json())


def load_prominence_availability_report(path: Path) -> ProminenceAvailabilityShapleyReport:
    return ProminenceAvailabilityShapleyReport.model_validate_json(path.read_text())
