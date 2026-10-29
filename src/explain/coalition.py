"""Seven- and eight-state PR-QRS-T structural-deletion games.

A coalition names regions that have been deleted. Singleton/pair directional
effects and exact Shapley contributions use marginal score drops. The complete
eight-state game additionally supports an orthogonal 2^3 decomposition coded
with +1 for region presence and -1 for deletion; its pair/triple terms are
factorial contrasts, not single before/after marginals. The intentionally
incomplete seven-state game cannot define Shapley or factorial effects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from explain.ecg_taxonomy import (
    REGION_ORDER,
    REGION_PAIR_STATE_ORDER,
    SINGLE_PAIR_STATE_ORDER,
    ECGRegion,
)
from explain.occlusion import (
    EmptyExamError,
    TargetScore,
    delete_nodes_with_cascade,
    target_scores,
)
from explain.region_intervention import RegionNodeMasks
from models.hier_model import HierGNNModel

SINGLE_PAIR_STATE_REGIONS = {
    "intact": (),
    "pr": (ECGRegion.PR,),
    "qrs": (ECGRegion.QRS,),
    "t": (ECGRegion.T,),
    "pr+qrs": (ECGRegion.PR, ECGRegion.QRS),
    "pr+t": (ECGRegion.PR, ECGRegion.T),
    "qrs+t": (ECGRegion.QRS, ECGRegion.T),
}
EIGHT_STATE_ORDER = (*SINGLE_PAIR_STATE_ORDER, "pr+qrs+t")
FACTORIAL_TERM_ORDER = (
    "pr",
    "qrs",
    "t",
    "pr*qrs",
    "pr*t",
    "qrs*t",
    "pr*qrs*t",
)
EIGHT_STATE_REGIONS = {
    **SINGLE_PAIR_STATE_REGIONS,
    "pr+qrs+t": (ECGRegion.PR, ECGRegion.QRS, ECGRegion.T),
}
_STATE_BY_REGIONS = {
    frozenset(region.value for region in regions): state
    for state, regions in EIGHT_STATE_REGIONS.items()
}


@dataclass(frozen=True)
class ExactRegionalShapley:
    """Exact three-player allocation and secondary pair interactions."""

    contributions: dict[str, float]
    joint_drop: float
    efficiency_error: float
    pair_interactions: dict[str, float]


@dataclass(frozen=True)
class FactorialEffects:
    """Orthogonal effect-coded decomposition of one complete 2^3 game.

    ``effects`` are high-minus-low factorial effects, equal to twice the
    corresponding coefficient in a saturated model coded with +1 for a region
    being present and -1 for it being deleted. This common normalization makes
    main, pair, and triple terms directly comparable on the same score scale.
    """

    grand_mean: float
    effects: dict[str, float]


@dataclass(frozen=True)
class CoalitionStateScore:
    """One raw coalition score and structural-deletion audit counts."""

    state: str
    regions: tuple[ECGRegion, ...]
    score: TargetScore | None
    removed_nodes: int
    removed_subgraphs: int | None
    failure: Literal["empty_exam", "non_finite_score"] | None


@dataclass(frozen=True)
class CoalitionEvaluation:
    """Ordered raw coalition states for one exam and target class."""

    states: tuple[CoalitionStateScore, ...]

    @property
    def by_state(self) -> dict[str, CoalitionStateScore]:
        return {state.state: state for state in self.states}

    def score_values(self, scale: str) -> dict[str, float]:
        """Return one complete raw score scale, rejecting undefined states."""
        if scale not in {"probability", "logit"}:
            raise ValueError(f"scale must be 'probability' or 'logit', got {scale!r}")
        values: dict[str, float] = {}
        for state in self.states:
            if state.score is None:
                raise ValueError(f"coalition state {state.state!r} is undefined: {state.failure}")
            values[state.state] = float(getattr(state.score, scale))
        return values


def _finite_scores(
    model: HierGNNModel,
    inputs: dict[str, Any],
    *,
    target_classes: tuple[int, ...],
    target_exam: int,
) -> dict[int, TargetScore | None]:
    scores = target_scores(
        model,
        inputs,
        target_classes=target_classes,
        target_exam=target_exam,
    )
    return {
        target_class: (
            score if math.isfinite(score.logit) and math.isfinite(score.probability) else None
        )
        for target_class, score in scores.items()
    }


def _evaluate_coalitions_multiclass(
    model: HierGNNModel,
    inputs: dict[str, Any],
    masks: RegionNodeMasks,
    *,
    target_classes: tuple[int, ...],
    target_exam: int,
    state_order: tuple[str, ...],
    state_regions: Mapping[str, tuple[ECGRegion, ...]],
) -> dict[int, CoalitionEvaluation]:
    """Walk the deletion states once and score every requested class.

    The deletion states depend only on the region masks, so one structural
    forward pass per state serves every requested class. Score finiteness and
    failure codes are tracked per class.
    """
    if not target_classes or len(set(target_classes)) != len(target_classes):
        raise ValueError("target_classes must be non-empty and unique")

    states_by_class: dict[int, list[CoalitionStateScore]] = {
        target_class: [] for target_class in target_classes
    }

    def append_state(
        state: str,
        regions: tuple[ECGRegion, ...],
        scores: dict[int, TargetScore | None] | None,
        *,
        removed_nodes: int,
        removed_subgraphs: int | None,
        failure: Literal["empty_exam"] | None,
    ) -> None:
        for target_class in target_classes:
            score = None if scores is None else scores[target_class]
            if failure is not None:
                class_failure: Literal["empty_exam", "non_finite_score"] | None = failure
            elif score is None:
                class_failure = "non_finite_score"
            else:
                class_failure = None
            states_by_class[target_class].append(
                CoalitionStateScore(
                    state=state,
                    regions=regions,
                    score=score,
                    removed_nodes=removed_nodes,
                    removed_subgraphs=removed_subgraphs,
                    failure=class_failure,
                )
            )

    append_state(
        "intact",
        (),
        _finite_scores(model, inputs, target_classes=target_classes, target_exam=target_exam),
        removed_nodes=0,
        removed_subgraphs=0,
        failure=None,
    )
    for state in state_order[1:]:
        regions = state_regions[state]
        remove_mask = masks.for_coalition(set(regions))
        try:
            deletion = delete_nodes_with_cascade(model, inputs, remove_mask)
        except EmptyExamError:
            append_state(
                state,
                regions,
                None,
                removed_nodes=int(remove_mask.sum().item()),
                removed_subgraphs=None,
                failure="empty_exam",
            )
            continue
        append_state(
            state,
            regions,
            _finite_scores(
                model,
                deletion.inputs,
                target_classes=target_classes,
                target_exam=target_exam,
            ),
            removed_nodes=deletion.removed_nodes,
            removed_subgraphs=deletion.removed_subgraphs,
            failure=None,
        )
    return {
        target_class: CoalitionEvaluation(states=tuple(states))
        for target_class, states in states_by_class.items()
    }


def _evaluate_coalitions(
    model: HierGNNModel,
    inputs: dict[str, Any],
    masks: RegionNodeMasks,
    *,
    target_class: int,
    target_exam: int,
    state_order: tuple[str, ...],
    state_regions: Mapping[str, tuple[ECGRegion, ...]],
) -> CoalitionEvaluation:
    return _evaluate_coalitions_multiclass(
        model,
        inputs,
        masks,
        target_classes=(target_class,),
        target_exam=target_exam,
        state_order=state_order,
        state_regions=state_regions,
    )[target_class]


def evaluate_eight_state_coalitions(
    model: HierGNNModel,
    inputs: dict[str, Any],
    masks: RegionNodeMasks,
    *,
    target_class: int,
    target_exam: int = 0,
) -> CoalitionEvaluation:
    """Evaluate every PR-QRS-T deletion coalition with cascade semantics."""
    return _evaluate_coalitions(
        model,
        inputs,
        masks,
        target_class=target_class,
        target_exam=target_exam,
        state_order=EIGHT_STATE_ORDER,
        state_regions=EIGHT_STATE_REGIONS,
    )


def evaluate_eight_state_coalitions_multiclass(
    model: HierGNNModel,
    inputs: dict[str, Any],
    masks: RegionNodeMasks,
    *,
    target_classes: tuple[int, ...],
    target_exam: int = 0,
) -> dict[int, CoalitionEvaluation]:
    """Evaluate all eight coalitions once and read several class outputs."""
    return _evaluate_coalitions_multiclass(
        model,
        inputs,
        masks,
        target_classes=target_classes,
        target_exam=target_exam,
        state_order=EIGHT_STATE_ORDER,
        state_regions=EIGHT_STATE_REGIONS,
    )


def _coalition_state(regions: frozenset[str]) -> str:
    return _STATE_BY_REGIONS[regions]


def exact_regional_shapley(values: Mapping[str, float]) -> ExactRegionalShapley:
    """Allocate an exact eight-state PR-QRS-T deletion game.

    Each contribution is the average marginal drop across all six deletion
    orders. Pair interaction is the change in one region's marginal drop after
    its partner is deleted, averaged without and with the third region already
    deleted. Positive interaction therefore means the partner's absence makes
    the region's deletion more consequential; negative means less consequential.
    """
    expected = set(EIGHT_STATE_ORDER)
    missing = expected - values.keys()
    extra = values.keys() - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing states: {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"unexpected states: {', '.join(sorted(extra))}")
        raise ValueError("; ".join(parts))
    score = {state: float(values[state]) for state in EIGHT_STATE_ORDER}
    if any(not math.isfinite(value) for value in score.values()):
        raise ValueError("all eight coalition scores must be finite")

    def marginal(region: str, context: frozenset[str]) -> float:
        before = score[_coalition_state(context)]
        after = score[_coalition_state(context | {region})]
        return before - after

    contributions: dict[str, float] = {}
    for region in REGION_ORDER:
        other = [candidate for candidate in REGION_ORDER if candidate != region]
        contributions[region] = (
            2.0 * marginal(region, frozenset())
            + marginal(region, frozenset({other[0]}))
            + marginal(region, frozenset({other[1]}))
            + 2.0 * marginal(region, frozenset(other))
        ) / 6.0

    interactions: dict[str, float] = {}
    for pair in REGION_PAIR_STATE_ORDER:
        first, second = pair.split("+")
        third = next(region for region in REGION_ORDER if region not in {first, second})
        without_third = marginal(first, frozenset({second})) - marginal(first, frozenset())
        with_third = marginal(first, frozenset({second, third})) - marginal(
            first, frozenset({third})
        )
        interactions[pair] = (without_third + with_third) / 2.0

    joint_drop = score["intact"] - score["pr+qrs+t"]
    efficiency_error = sum(contributions.values()) - joint_drop
    if not math.isclose(
        sum(contributions.values()),
        joint_drop,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        raise ArithmeticError("regional Shapley efficiency check failed")
    return ExactRegionalShapley(
        contributions=contributions,
        joint_drop=joint_drop,
        efficiency_error=efficiency_error,
        pair_interactions=interactions,
    )


def factorial_effects(values: Mapping[str, float]) -> FactorialEffects:
    """Decompose eight PR-QRS-T states into seven orthogonal factorial effects.

    Every state has equal weight. Region presence is the high level (+1), and
    structural deletion is the low level (-1). For every non-empty term, the
    reported effect is ``mean(x_term=+1) - mean(x_term=-1)``.
    """
    expected = set(EIGHT_STATE_ORDER)
    missing = expected - values.keys()
    extra = values.keys() - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing states: {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"unexpected states: {', '.join(sorted(extra))}")
        raise ValueError("; ".join(parts))
    scores = {state: float(values[state]) for state in EIGHT_STATE_ORDER}
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("all eight coalition scores must be finite")

    effects: dict[str, float] = {}
    for term in FACTORIAL_TERM_ORDER:
        regions = term.split("*")
        contrast = 0.0
        for state, score in scores.items():
            deleted = frozenset() if state == "intact" else frozenset(state.split("+"))
            sign = math.prod(-1.0 if region in deleted else 1.0 for region in regions)
            contrast += sign * score
        effects[term] = contrast / 4.0
    return FactorialEffects(
        grand_mean=sum(scores.values()) / 8.0,
        effects=effects,
    )
