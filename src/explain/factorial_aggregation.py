"""Shared exact-factorial result models and deterministic aggregation.

The two factorial workflows differ in provenance and support construction, but
not in how complete games are represented or how aligned checkpoint effects
are summarized.  This module owns that common computation.  Callers adapt
workflow records to the small immutable aggregation input and retain
responsibility for workflow-specific provenance validation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from explain.coalition import EIGHT_STATE_ORDER, FACTORIAL_TERM_ORDER, factorial_effects
from explain.coalition_workflow import CoalitionStateModel
from explain.common_support import RegionDefinitionMethod
from explain.ecg_taxonomy import OutputScale

SCALE_ORDER: tuple[OutputScale, ...] = ("logit", "probability")
FactorialSupportView = Literal[
    "nonzero_common_support",
    "common_support_rate_at_least_0_5",
    "nonzero_prominence_support",
    "prominence_support_rate_at_least_0_5",
]
_SUPPORT_VIEW_NAMES = (
    "nonzero_common_support",
    "common_support_rate_at_least_0_5",
    "nonzero_prominence_support",
    "prominence_support_rate_at_least_0_5",
)


class FactorialScaleEffectsModel(BaseModel):
    """Grand mean and seven orthogonal effects on one output scale."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    grand_mean: float
    effects: dict[str, float]

    @model_validator(mode="after")
    def _finite_canonical_terms(self) -> Self:
        if list(self.effects) != list(FACTORIAL_TERM_ORDER):
            raise ValueError("factorial effects require canonical term order")
        if not math.isfinite(self.grand_mean) or any(
            not math.isfinite(value) for value in self.effects.values()
        ):
            raise ValueError("factorial effects must be finite")
        return self


class FactorialMethodResult(BaseModel):
    """Eight raw states and exact factorial effects for one method/exam."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    method: RegionDefinitionMethod
    states: list[CoalitionStateModel]
    probability: FactorialScaleEffectsModel | None
    logit: FactorialScaleEffectsModel | None

    @model_validator(mode="after")
    def _coherent_result(self) -> Self:
        validate_factorial_method_result_integrity(self)
        return self


class BootstrapConfig(BaseModel):
    """Deterministic exam-level bootstrap configuration."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    seed: Annotated[int, Field(ge=0)]
    resamples: Annotated[int, Field(ge=100, le=100_000)]
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)]
    resampling_unit: Literal["exam"]


class EffectDistributionSummary(BaseModel):
    """Across-exam distribution after averaging checkpoints within exam."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: str
    support_view: FactorialSupportView
    method: RegionDefinitionMethod
    scale: OutputScale
    term: str
    n_exams: Annotated[int, Field(gt=0)]
    mean: float
    median: float
    q1: float
    q3: float
    bootstrap_mean_ci_low: float
    bootstrap_mean_ci_high: float

    @model_validator(mode="after")
    def _canonical_finite_summary(self) -> Self:
        if self.term not in FACTORIAL_TERM_ORDER:
            raise ValueError("unknown factorial term")
        values = (
            self.mean,
            self.median,
            self.q1,
            self.q3,
            self.bootstrap_mean_ci_low,
            self.bootstrap_mean_ci_high,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("effect summary values must be finite")
        if not self.q1 <= self.median <= self.q3:
            raise ValueError("effect quartiles are not ordered")
        if self.bootstrap_mean_ci_low > self.bootstrap_mean_ci_high:
            raise ValueError("bootstrap confidence interval is not ordered")
        return self


class PairedMethodContrastSummary(BaseModel):
    """Paired across-exam contrast between two region definitions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: str
    support_view: FactorialSupportView
    method_a: RegionDefinitionMethod
    method_b: RegionDefinitionMethod
    scale: OutputScale
    term: str
    n_exams: Annotated[int, Field(gt=0)]
    mean_difference_a_minus_b: float
    bootstrap_mean_difference_ci_low: float
    bootstrap_mean_difference_ci_high: float

    @model_validator(mode="after")
    def _valid_contrast(self) -> Self:
        if self.method_a == self.method_b or self.term not in FACTORIAL_TERM_ORDER:
            raise ValueError("paired contrast requires distinct methods and a canonical term")
        values = (
            self.mean_difference_a_minus_b,
            self.bootstrap_mean_difference_ci_low,
            self.bootstrap_mean_difference_ci_high,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("paired contrast values must be finite")
        if self.bootstrap_mean_difference_ci_low > self.bootstrap_mean_difference_ci_high:
            raise ValueError("paired contrast confidence interval is not ordered")
        return self


class PairedTermContrastSummary(BaseModel):
    """Paired across-exam contrast between two factorial terms."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    class_name: str
    support_view: FactorialSupportView
    method: RegionDefinitionMethod
    scale: OutputScale
    term_a: str
    term_b: str
    n_exams: Annotated[int, Field(gt=0)]
    mean_difference_a_minus_b: float
    bootstrap_mean_difference_ci_low: float
    bootstrap_mean_difference_ci_high: float

    @model_validator(mode="after")
    def _valid_contrast(self) -> Self:
        if (
            self.term_a == self.term_b
            or self.term_a not in FACTORIAL_TERM_ORDER
            or self.term_b not in FACTORIAL_TERM_ORDER
        ):
            raise ValueError("term contrast requires distinct canonical factorial terms")
        values = (
            self.mean_difference_a_minus_b,
            self.bootstrap_mean_difference_ci_low,
            self.bootstrap_mean_difference_ci_high,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("term contrast values must be finite")
        if self.bootstrap_mean_difference_ci_low > self.bootstrap_mean_difference_ci_high:
            raise ValueError("term contrast confidence interval is not ordered")
        return self


def _validate_method_structure(result: FactorialMethodResult) -> None:
    if [state.state for state in result.states] != list(EIGHT_STATE_ORDER):
        raise ValueError("factorial method result requires all eight states in canonical order")
    complete = all(state.failure is None for state in result.states)
    if complete != (result.probability is not None and result.logit is not None):
        raise ValueError("factorial effects exist exactly when all eight states are finite")


def _scale_model(
    states: list[CoalitionStateModel], scale: OutputScale
) -> FactorialScaleEffectsModel | None:
    if any(state.failure is not None for state in states):
        return None
    values = {
        state.state: getattr(state, scale) for state in states if getattr(state, scale) is not None
    }
    result = factorial_effects(values)
    return FactorialScaleEffectsModel(
        grand_mean=result.grand_mean,
        effects=dict(result.effects),
    )


def build_factorial_method_result(
    method: RegionDefinitionMethod,
    states: list[CoalitionStateModel],
) -> FactorialMethodResult:
    """Build a strict method result from exactly eight raw states."""
    return FactorialMethodResult(
        method=method,
        states=states,
        probability=_scale_model(states, "probability"),
        logit=_scale_model(states, "logit"),
    )


def validate_factorial_method_result_integrity(result: FactorialMethodResult) -> None:
    """Verify stored factorial effects against raw states without model-parse side effects."""
    _validate_method_structure(result)
    if result.probability is None or result.logit is None:
        return
    for scale in SCALE_ORDER:
        expected = _scale_model(result.states, scale)
        if getattr(result, scale) != expected:
            raise ValueError(f"{scale} factorial effects disagree with raw states")


@dataclass(frozen=True)
class SupportViewSpec:
    """One ordered support view and its inclusive minimum support rate."""

    name: FactorialSupportView
    minimum_rate: float | None = None

    def __post_init__(self) -> None:
        if self.name not in _SUPPORT_VIEW_NAMES:
            raise ValueError(f"unknown factorial support view {self.name!r}")
        if self.minimum_rate is not None and (
            not math.isfinite(self.minimum_rate) or not 0.0 < self.minimum_rate <= 1.0
        ):
            raise ValueError("minimum support rate must lie in (0, 1]")

    def includes(self, support_rate: float) -> bool:
        """Return whether an already-supported exam belongs to this view."""
        return self.minimum_rate is None or support_rate >= self.minimum_rate


@dataclass(frozen=True)
class FactorialExamInput:
    """Workflow-neutral effects and support rate for one aligned exam."""

    exam_index: int
    support_rate: float
    methods: tuple[FactorialMethodResult, ...]


@dataclass(frozen=True)
class FactorialClassInput:
    """Ordered exams for one class in one checkpoint run."""

    class_name: str
    exams: tuple[FactorialExamInput, ...]


@dataclass(frozen=True)
class FactorialRunInput:
    """Ordered class/exam effects for one checkpoint run."""

    classes: tuple[FactorialClassInput, ...]


@dataclass(frozen=True)
class FactorialAggregation:
    """All deterministic summaries requested by one aggregation call."""

    summaries: list[EffectDistributionSummary]
    method_contrasts: list[PairedMethodContrastSummary]
    term_contrasts: list[PairedTermContrastSummary]


def _identity_seed(base_seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _bootstrap_mean_ci(
    values: np.ndarray,
    config: BootstrapConfig,
    *,
    identity: str,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires a non-empty finite one-dimensional sample")
    rng = np.random.default_rng(_identity_seed(config.seed, identity))
    indices = rng.integers(0, values.size, size=(config.resamples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - config.confidence_level) / 2.0
    low, high = np.quantile(means, (alpha, 1.0 - alpha))
    return float(low), float(high)


def _validate_inputs(
    runs: list[FactorialRunInput],
    support_views: tuple[SupportViewSpec, ...],
    method_order: tuple[RegionDefinitionMethod, ...],
    term_contrast_pairs: tuple[tuple[str, str], ...],
) -> None:
    if not runs:
        raise ValueError("factorial aggregation requires at least one run")
    if not support_views or len({view.name for view in support_views}) != len(support_views):
        raise ValueError("factorial aggregation requires distinct support views")
    if not method_order or len(set(method_order)) != len(method_order):
        raise ValueError("factorial aggregation requires distinct methods")
    if len(set(term_contrast_pairs)) != len(term_contrast_pairs) or any(
        first == second or first not in FACTORIAL_TERM_ORDER or second not in FACTORIAL_TERM_ORDER
        for first, second in term_contrast_pairs
    ):
        raise ValueError("factorial aggregation requires distinct canonical term pairs")

    reference_classes = runs[0].classes
    class_order = [class_result.class_name for class_result in reference_classes]
    if (
        not class_order
        or any(not class_name for class_name in class_order)
        or len(class_order) != len(set(class_order))
    ):
        raise ValueError("factorial aggregation requires non-empty unique classes")
    for run in runs:
        if [class_result.class_name for class_result in run.classes] != class_order:
            raise ValueError("factorial aggregation runs have different class order")
        for class_result, reference_class in zip(
            run.classes,
            reference_classes,
            strict=True,
        ):
            identities = [(exam.exam_index, exam.support_rate) for exam in class_result.exams]
            reference_identities = [
                (exam.exam_index, exam.support_rate) for exam in reference_class.exams
            ]
            if identities != reference_identities:
                raise ValueError("factorial aggregation runs have incompatible exam support")
            if [exam.exam_index for exam in class_result.exams] != sorted(
                {exam.exam_index for exam in class_result.exams}
            ) or any(exam.exam_index < 0 for exam in class_result.exams):
                raise ValueError(
                    "factorial aggregation exam indices must be non-negative, sorted, and unique"
                )
            for exam in class_result.exams:
                if not math.isfinite(exam.support_rate) or not 0.0 < exam.support_rate <= 1.0:
                    raise ValueError("factorial aggregation support rates must lie in (0, 1]")
                if tuple(method.method for method in exam.methods) != method_order:
                    raise ValueError("factorial aggregation requires the declared method order")
                for method in exam.methods:
                    validate_factorial_method_result_integrity(method)


def aggregate_factorial_effects(
    runs: list[FactorialRunInput],
    bootstrap: BootstrapConfig,
    *,
    support_views: tuple[SupportViewSpec, ...],
    method_order: tuple[RegionDefinitionMethod, ...],
    method_contrast_identity_namespace: str,
    term_contrast_pairs: tuple[tuple[str, str], ...] = (),
    term_contrast_identity_namespace: str | None = None,
) -> FactorialAggregation:
    """Aggregate aligned runs with stable ordering, identities, and float64 arithmetic.

    The workflow adapters validate provenance before crossing this seam.  This
    module validates the normalized alignment and every stored per-game effect,
    averages checkpoints within exam, then performs the deterministic
    exam-level bootstrap exactly once per requested statistic.
    """
    if not method_contrast_identity_namespace or "/" in method_contrast_identity_namespace:
        raise ValueError("method-contrast identity namespace must be one path segment")
    if term_contrast_pairs and (
        not term_contrast_identity_namespace or "/" in term_contrast_identity_namespace
    ):
        raise ValueError("term-contrast identity namespace must be one path segment")
    if not term_contrast_pairs and term_contrast_identity_namespace is not None:
        raise ValueError("term-contrast namespace requires term contrast pairs")
    _validate_inputs(runs, support_views, method_order, term_contrast_pairs)

    class_order = [class_result.class_name for class_result in runs[0].classes]
    values_by_key: dict[
        tuple[str, FactorialSupportView, RegionDefinitionMethod, OutputScale, str],
        tuple[list[int], np.ndarray],
    ] = {}
    summaries: list[EffectDistributionSummary] = []
    for class_position, class_name in enumerate(class_order):
        run_classes = [run.classes[class_position] for run in runs]
        for support_view in support_views:
            for method_position, method in enumerate(method_order):
                for scale in SCALE_ORDER:
                    for term in FACTORIAL_TERM_ORDER:
                        indices: list[int] = []
                        values: list[float] = []
                        for exam_position, reference_exam in enumerate(run_classes[0].exams):
                            if not support_view.includes(reference_exam.support_rate):
                                continue
                            seed_values: list[float] = []
                            for class_result in run_classes:
                                scale_result = getattr(
                                    class_result.exams[exam_position].methods[method_position],
                                    scale,
                                )
                                if scale_result is None:
                                    break
                                seed_values.append(scale_result.effects[term])
                            if len(seed_values) == len(runs):
                                indices.append(reference_exam.exam_index)
                                values.append(
                                    float(np.mean(np.asarray(seed_values, dtype=np.float64)))
                                )
                        value_array = np.asarray(values, dtype=np.float64)
                        if value_array.size == 0:
                            raise ValueError(
                                "no complete factorial exams for "
                                f"{class_name}/{support_view.name}/{method.value}/{scale}"
                            )
                        key = (class_name, support_view.name, method, scale, term)
                        values_by_key[key] = (indices, value_array)
                        ci_low, ci_high = _bootstrap_mean_ci(
                            value_array,
                            bootstrap,
                            identity="/".join(
                                (
                                    "summary",
                                    class_name,
                                    support_view.name,
                                    method.value,
                                    scale,
                                    term,
                                )
                            ),
                        )
                        q1, median, q3 = np.quantile(value_array, (0.25, 0.5, 0.75))
                        summaries.append(
                            EffectDistributionSummary(
                                class_name=class_name,
                                support_view=support_view.name,
                                method=method,
                                scale=scale,
                                term=term,
                                n_exams=int(value_array.size),
                                mean=float(value_array.mean()),
                                median=float(median),
                                q1=float(q1),
                                q3=float(q3),
                                bootstrap_mean_ci_low=ci_low,
                                bootstrap_mean_ci_high=ci_high,
                            )
                        )

    method_contrasts: list[PairedMethodContrastSummary] = []
    for class_name in class_order:
        for support_view in support_views:
            for method_a, method_b in combinations(method_order, 2):
                for scale in SCALE_ORDER:
                    for term in FACTORIAL_TERM_ORDER:
                        indices_a, values_a = values_by_key[
                            (class_name, support_view.name, method_a, scale, term)
                        ]
                        indices_b, values_b = values_by_key[
                            (class_name, support_view.name, method_b, scale, term)
                        ]
                        if indices_a != indices_b:
                            raise ValueError("method contrast requires the same complete exams")
                        differences = values_a - values_b
                        ci_low, ci_high = _bootstrap_mean_ci(
                            differences,
                            bootstrap,
                            identity="/".join(
                                (
                                    method_contrast_identity_namespace,
                                    class_name,
                                    support_view.name,
                                    method_a.value,
                                    method_b.value,
                                    scale,
                                    term,
                                )
                            ),
                        )
                        method_contrasts.append(
                            PairedMethodContrastSummary(
                                class_name=class_name,
                                support_view=support_view.name,
                                method_a=method_a,
                                method_b=method_b,
                                scale=scale,
                                term=term,
                                n_exams=int(differences.size),
                                mean_difference_a_minus_b=float(differences.mean()),
                                bootstrap_mean_difference_ci_low=ci_low,
                                bootstrap_mean_difference_ci_high=ci_high,
                            )
                        )

    term_contrasts: list[PairedTermContrastSummary] = []
    for class_name in class_order:
        for support_view in support_views:
            for method in method_order:
                for scale in SCALE_ORDER:
                    for term_a, term_b in term_contrast_pairs:
                        indices_a, values_a = values_by_key[
                            (class_name, support_view.name, method, scale, term_a)
                        ]
                        indices_b, values_b = values_by_key[
                            (class_name, support_view.name, method, scale, term_b)
                        ]
                        if indices_a != indices_b:
                            raise ValueError("term contrast requires the same complete exams")
                        differences = values_a - values_b
                        assert term_contrast_identity_namespace is not None
                        ci_low, ci_high = _bootstrap_mean_ci(
                            differences,
                            bootstrap,
                            identity="/".join(
                                (
                                    term_contrast_identity_namespace,
                                    class_name,
                                    support_view.name,
                                    method.value,
                                    scale,
                                    term_a,
                                    term_b,
                                )
                            ),
                        )
                        term_contrasts.append(
                            PairedTermContrastSummary(
                                class_name=class_name,
                                support_view=support_view.name,
                                method=method,
                                scale=scale,
                                term_a=term_a,
                                term_b=term_b,
                                n_exams=int(differences.size),
                                mean_difference_a_minus_b=float(differences.mean()),
                                bootstrap_mean_difference_ci_low=ci_low,
                                bootstrap_mean_difference_ci_high=ci_high,
                            )
                        )
    return FactorialAggregation(
        summaries=summaries,
        method_contrasts=method_contrasts,
        term_contrasts=term_contrasts,
    )
