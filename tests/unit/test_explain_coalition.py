"""Literal tests for signed PR-QRS-T singleton and pair effects."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from explain.coalition import (
    EIGHT_STATE_ORDER,
    FACTORIAL_TERM_ORDER,
    REGION_ORDER,
    SINGLE_PAIR_STATE_ORDER,
    evaluate_eight_state_coalitions,
    evaluate_eight_state_coalitions_multiclass,
    exact_regional_shapley,
    factorial_effects,
)
from explain.region_intervention import RegionNodeMasks
from models.hier_model import ModelOutput

pytestmark = pytest.mark.unit


class _SumNodeModel(nn.Module):
    """Analytic model whose target logit is the sum of retained node weights."""

    num_classes = 1

    def forward(self, **inputs) -> ModelOutput:
        logit = inputs["x"][:, 0].sum().reshape(1, 1)
        return ModelOutput(
            logits=logit,
        )


def _analytic_inputs() -> dict:
    return {
        "x": torch.tensor([[0.25], [0.50], [1.00], [-0.25]]),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "edge_attr": torch.empty((3, 0)),
        "subgraph_node_batch": torch.zeros(4, dtype=torch.long),
        "super_edge_index": torch.empty((2, 0), dtype=torch.long),
        "super_edge_step": torch.empty(0, dtype=torch.long),
        "subgraph_lead": torch.zeros(1, dtype=torch.long),
        "subgraph_exam": torch.zeros(1, dtype=torch.long),
        "subgraph_beat": torch.zeros(1, dtype=torch.long),
        "batch_size": 1,
    }


def test_exact_regional_shapley_allocates_an_additive_deletion_game():
    values = {
        "intact": 10.0,
        "pr": 8.0,
        "qrs": 7.0,
        "t": 9.0,
        "pr+qrs": 5.0,
        "pr+t": 7.0,
        "qrs+t": 6.0,
        "pr+qrs+t": 4.0,
    }

    result = exact_regional_shapley(values)

    assert (*SINGLE_PAIR_STATE_ORDER, "pr+qrs+t") == EIGHT_STATE_ORDER
    assert result.contributions == pytest.approx({"pr": 2.0, "qrs": 3.0, "t": 1.0})
    assert result.joint_drop == pytest.approx(6.0)
    assert result.efficiency_error == pytest.approx(0.0)
    assert result.pair_interactions == pytest.approx({"pr+qrs": 0.0, "pr+t": 0.0, "qrs+t": 0.0})


def test_exact_regional_shapley_splits_redundant_pair_evidence():
    result = exact_regional_shapley(
        {
            "intact": 1.0,
            "pr": 1.0,
            "qrs": 1.0,
            "t": 1.0,
            "pr+qrs": 0.0,
            "pr+t": 1.0,
            "qrs+t": 1.0,
            "pr+qrs+t": 0.0,
        }
    )

    assert result.contributions == pytest.approx({"pr": 0.5, "qrs": 0.5, "t": 0.0})
    assert result.pair_interactions == pytest.approx({"pr+qrs": 1.0, "pr+t": 0.0, "qrs+t": 0.0})


def test_exact_regional_shapley_preserves_a_suppressive_region():
    result = exact_regional_shapley(
        {
            "intact": 10.0,
            "pr": 8.0,
            "qrs": 9.0,
            "t": 10.5,
            "pr+qrs": 7.0,
            "pr+t": 8.5,
            "qrs+t": 9.5,
            "pr+qrs+t": 7.5,
        }
    )

    assert result.contributions == pytest.approx({"pr": 2.0, "qrs": 1.0, "t": -0.5})
    assert result.joint_drop == pytest.approx(2.5)


def test_exact_regional_shapley_includes_third_region_context():
    result = exact_regional_shapley(
        {
            "intact": 1.0,
            "pr": 1.0,
            "qrs": 1.0,
            "t": 1.0,
            "pr+qrs": 1.0,
            "pr+t": 1.0,
            "qrs+t": 1.0,
            "pr+qrs+t": 0.0,
        }
    )

    assert result.contributions == pytest.approx({"pr": 1 / 3, "qrs": 1 / 3, "t": 1 / 3})
    assert result.pair_interactions == pytest.approx({"pr+qrs": 0.5, "pr+t": 0.5, "qrs+t": 0.5})


def test_factorial_effects_recover_main_pair_and_triple_effect_coding_terms():
    coefficients = {
        "pr": 2.0,
        "qrs": 3.0,
        "t": 4.0,
        "pr*qrs": 5.0,
        "pr*t": -1.0,
        "qrs*t": 0.5,
        "pr*qrs*t": 7.0,
    }
    values: dict[str, float] = {}
    for state in EIGHT_STATE_ORDER:
        deleted = set() if state == "intact" else set(state.split("+"))
        signs = {region: (-1.0 if region in deleted else 1.0) for region in REGION_ORDER}
        values[state] = 10.0 + sum(
            coefficient * math.prod(signs[region] for region in term.split("*"))
            for term, coefficient in coefficients.items()
        )

    result = factorial_effects(values)

    assert FACTORIAL_TERM_ORDER == (
        "pr",
        "qrs",
        "t",
        "pr*qrs",
        "pr*t",
        "qrs*t",
        "pr*qrs*t",
    )
    assert result.grand_mean == pytest.approx(10.0)
    assert result.effects == pytest.approx(
        {term: 2.0 * coefficient for term, coefficient in coefficients.items()}
    )


def test_factorial_pair_effect_has_documented_relation_to_shapley_pair_interaction():
    values = {
        "intact": 1.0,
        "pr": 0.7,
        "qrs": 0.6,
        "t": 0.9,
        "pr+qrs": 0.2,
        "pr+t": 0.65,
        "qrs+t": 0.5,
        "pr+qrs+t": 0.1,
    }

    factorial = factorial_effects(values)
    shapley = exact_regional_shapley(values)

    assert shapley.pair_interactions == pytest.approx(
        {
            pair: -2.0 * factorial.effects[pair.replace("+", "*")]
            for pair in ("pr+qrs", "pr+t", "qrs+t")
        }
    )
    assert shapley.contributions == pytest.approx(
        {
            region: factorial.effects[region] + factorial.effects["pr*qrs*t"] / 3.0
            for region in REGION_ORDER
        }
    )


def test_factorial_main_effect_equals_uniform_average_marginal_presence_effect():
    values = {
        "intact": 10.0,
        "pr": 8.0,
        "qrs": 7.0,
        "t": 9.0,
        "pr+qrs": 4.0,
        "pr+t": 7.5,
        "qrs+t": 6.0,
        "pr+qrs+t": 2.0,
    }

    result = factorial_effects(values)

    expected_pr = ((10.0 - 8.0) + (7.0 - 4.0) + (9.0 - 7.5) + (6.0 - 2.0)) / 4
    assert result.effects["pr"] == pytest.approx(expected_pr)
    assert result.effects["pr*qrs"] == pytest.approx(
        ((10.0 - 8.0 - 7.0 + 4.0) + (9.0 - 7.5 - 6.0 + 2.0)) / 4
    )


def test_analytic_model_evaluates_triple_state_for_exact_shapley():
    masks = RegionNodeMasks(
        pr=torch.tensor([False, True, False, False]),
        qrs=torch.tensor([False, False, True, False]),
        t=torch.tensor([False, False, False, True]),
    )

    evaluation = evaluate_eight_state_coalitions(
        _SumNodeModel(), _analytic_inputs(), masks, target_class=0
    )

    assert tuple(evaluation.by_state) == EIGHT_STATE_ORDER
    triple = evaluation.by_state["pr+qrs+t"]
    assert triple.score is not None
    assert triple.score.logit == pytest.approx(0.25)
    assert triple.removed_nodes == 3
    shapley = exact_regional_shapley(evaluation.score_values("logit"))
    assert shapley.contributions == pytest.approx({"pr": 0.5, "qrs": 1.0, "t": -0.25})


def test_triple_coalition_cascades_an_emptied_subgraph_but_keeps_exam_valid():
    inputs = _analytic_inputs()
    inputs["subgraph_node_batch"] = torch.tensor([0, 0, 0, 1])
    inputs["subgraph_lead"] = torch.tensor([0, 1])
    inputs["subgraph_exam"] = torch.tensor([0, 0])
    inputs["subgraph_beat"] = torch.tensor([0, 0])
    masks = RegionNodeMasks(
        pr=torch.tensor([True, False, False, False]),
        qrs=torch.tensor([False, True, False, False]),
        t=torch.tensor([False, False, True, False]),
    )

    evaluation = evaluate_eight_state_coalitions(_SumNodeModel(), inputs, masks, target_class=0)

    triple = evaluation.by_state["pr+qrs+t"]
    assert triple.failure is None
    assert triple.removed_subgraphs == 1
    assert triple.score is not None
    assert triple.score.logit == pytest.approx(-0.25)
    assert exact_regional_shapley(
        evaluation.score_values("logit")
    ).efficiency_error == pytest.approx(0.0)


def test_triple_that_empties_the_whole_exam_is_invalid_for_shapley():
    inputs = _analytic_inputs()
    inputs["x"] = inputs["x"][:3]
    inputs["edge_index"] = inputs["edge_index"][:, :2]
    inputs["edge_attr"] = inputs["edge_attr"][:2]
    inputs["subgraph_node_batch"] = torch.zeros(3, dtype=torch.long)
    masks = RegionNodeMasks(
        pr=torch.tensor([True, False, False]),
        qrs=torch.tensor([False, True, False]),
        t=torch.tensor([False, False, True]),
    )

    evaluation = evaluate_eight_state_coalitions(_SumNodeModel(), inputs, masks, target_class=0)

    assert evaluation.by_state["pr+qrs+t"].failure == "empty_exam"
    with pytest.raises(ValueError, match="empty_exam"):
        evaluation.score_values("logit")


class _ThreeClassSumModel(nn.Module):
    """Analytic model with three class logits derived from one retained sum."""

    num_classes = 3

    def forward(self, **inputs) -> ModelOutput:
        retained = inputs["x"][:, 0].sum()
        logits = torch.stack([retained, 2.0 * retained, -retained]).reshape(1, 3)
        return ModelOutput(
            logits=logits,
        )


def test_multiclass_eight_state_evaluation_matches_per_class_evaluations():
    masks = RegionNodeMasks(
        pr=torch.tensor([False, True, False, False]),
        qrs=torch.tensor([False, False, True, False]),
        t=torch.tensor([False, False, False, True]),
    )
    model = _ThreeClassSumModel()

    shared = evaluate_eight_state_coalitions_multiclass(
        model, _analytic_inputs(), masks, target_classes=(0, 1, 2)
    )

    for target_class in (0, 1, 2):
        separate = evaluate_eight_state_coalitions(
            model, _analytic_inputs(), masks, target_class=target_class
        )
        for shared_state, separate_state in zip(
            shared[target_class].states, separate.states, strict=True
        ):
            assert shared_state.state == separate_state.state
            assert shared_state.removed_nodes == separate_state.removed_nodes
            assert shared_state.removed_subgraphs == separate_state.removed_subgraphs
            assert shared_state.failure == separate_state.failure
            assert shared_state.score is not None and separate_state.score is not None
            assert shared_state.score.logit == pytest.approx(separate_state.score.logit)
            assert shared_state.score.probability == pytest.approx(separate_state.score.probability)
    base = exact_regional_shapley(shared[0].score_values("logit"))
    doubled = exact_regional_shapley(shared[1].score_values("logit"))
    negated = exact_regional_shapley(shared[2].score_values("logit"))
    for region in REGION_ORDER:
        assert doubled.contributions[region] == pytest.approx(2.0 * base.contributions[region])
        assert negated.contributions[region] == pytest.approx(-base.contributions[region])
