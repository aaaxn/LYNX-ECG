"""Shared PR-QRS-T coalition projections and fixed-window evaluation primitives."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from explain.coalition import (
    EIGHT_STATE_REGIONS,
    CoalitionEvaluation,
)
from explain.ecg_taxonomy import (
    ECGRegion,
)


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


class CoalitionStateModel(BaseModel):
    """Raw output and cascade audit for one deletion coalition."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    state: str
    deleted_regions: list[ECGRegion]
    probability: float | None
    logit: float | None
    removed_nodes: int
    removed_subgraphs: int | None
    failure: Literal["empty_exam", "non_finite_score"] | None

    @model_validator(mode="after")
    def _consistent_state(self) -> Self:
        if self.state not in EIGHT_STATE_REGIONS:
            raise ValueError(f"unknown coalition state {self.state!r}")
        if tuple(self.deleted_regions) != EIGHT_STATE_REGIONS[self.state]:
            raise ValueError("deleted regions do not match the coalition state")
        if self.removed_nodes < 0 or (
            self.removed_subgraphs is not None and self.removed_subgraphs < 0
        ):
            raise ValueError("removed-node/subgraph counts must be non-negative")
        if self.state == "intact" and (self.removed_nodes != 0 or self.removed_subgraphs != 0):
            raise ValueError("the intact state cannot remove nodes or subgraphs")
        has_scores = self.probability is not None and self.logit is not None
        has_no_scores = self.probability is None and self.logit is None
        if (self.failure is None and not has_scores) or (
            self.failure is not None and not has_no_scores
        ):
            raise ValueError("scores must be present exactly when the state has no failure")
        if self.failure == "empty_exam" and self.removed_subgraphs is not None:
            raise ValueError("an empty-exam failure has no completed cascade count")
        if self.failure != "empty_exam" and self.removed_subgraphs is None:
            raise ValueError("a completed deletion requires a removed-subgraph count")
        if has_scores and (
            not 0.0 <= self.probability <= 1.0
            or not math.isfinite(self.probability)
            or not math.isfinite(self.logit)
        ):
            raise ValueError("raw probability/logit scores must be finite and valid")
        if has_scores and not math.isclose(
            self.probability,
            _sigmoid(self.logit),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise ValueError("probability does not match the sigmoid of its logit")
        return self


def build_coalition_state_models(evaluation: CoalitionEvaluation) -> list[CoalitionStateModel]:
    """Project one raw evaluation's coalition states into their report models.

    Shared by common-support seven-state analyses and exact eight-state fixed
    analyses. Both persist the same :class:`CoalitionStateModel` rows because
    the projection reads only fields every coalition evaluation has.
    """
    return [
        CoalitionStateModel(
            state=result.state,
            deleted_regions=list(result.regions),
            probability=None if result.score is None else result.score.probability,
            logit=None if result.score is None else result.score.logit,
            removed_nodes=result.removed_nodes,
            removed_subgraphs=result.removed_subgraphs,
            failure=result.failure,
        )
        for result in evaluation.states
    ]
