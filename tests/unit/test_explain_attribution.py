"""Unit tests for Integrated Gradients node attribution (explain.attribution)."""

from __future__ import annotations

import pytest
import torch

from explain.attribution import NodeAttribution, integrated_gradients
from models.hier_model import HierGNNModel
from tests.fixtures.hier_graph import (
    ring_subgraph_inputs as _ring_subgraphs,
)
from tests.fixtures.hier_graph import (
    single_exam_ring_inputs as _single_exam_inputs,
)
from tests.fixtures.hier_graph import (
    tiny_hier_model as _model,
)

pytestmark = pytest.mark.unit


def _two_exam_inputs() -> dict:
    """Exam 0: subgraphs 0..3 (16 nodes). Exam 1: subgraph 4 (4 nodes)."""
    base = _ring_subgraphs(5)
    base.update(
        super_edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
        super_edge_step=torch.tensor([1, -1, 1, -1]),
        subgraph_lead=torch.tensor([0, 0, 1, 1, 0]),
        subgraph_exam=torch.tensor([0, 0, 0, 0, 1]),
        subgraph_beat=torch.tensor([0, 1, 0, 1, 0]),
        batch_size=2,
    )
    return base


def test_ig_shapes_and_metadata():
    torch.manual_seed(0)
    model = _model()
    inputs = _single_exam_inputs()
    attr = integrated_gradients(model, inputs, target_class=2, steps=32)

    assert isinstance(attr, NodeAttribution)
    n_nodes = inputs["x"].shape[0]
    assert attr.node_importance.shape == (n_nodes,)
    assert attr.feature_importance.shape == (n_nodes, 2)
    assert attr.target_class == 2
    assert attr.target_exam == 0
    assert torch.isfinite(attr.node_importance).all()


def test_ig_satisfies_completeness():
    """sum(IG) == f(x) - f(baseline) on a case with a NON-TRIVIAL gap.

    Completeness is trivially satisfied when the prediction gap is ~0 (flat path
    => zero gradients => both sides collapse to 0). To make the test
    self-certifying, we first find a model that yields a healthy gap (scaled
    inputs push the prediction away from the zero baseline), then assert the gap
    is reproduced to integration precision. It cannot pass by collapsing to zero.
    """
    torch.manual_seed(12345)
    inputs = _single_exam_inputs()
    inputs["x"] = inputs["x"] * 3.0

    attr = None
    for seed in range(20):
        torch.manual_seed(seed)
        model = HierGNNModel(
            num_classes=5,
            num_leads=2,
            hidden_channels=32,
            beat_encoder_layers=2,
            beat_encoder_mlp_hidden=64,
            gatv2_heads=4,
            gatv2_layers=2,
        )
        attr = integrated_gradients(model, inputs, target_class=3, steps=256)
        if abs(attr.prediction_gap) > 0.05:
            break
    else:
        pytest.fail("no seed produced a non-trivial prediction gap to test completeness")

    assert abs(attr.prediction_gap) > 0.05
    assert attr.completeness_gap < 0.01 * abs(attr.prediction_gap)
    assert attr.completeness_gap < 5e-3


def test_ig_zero_attribution_for_nontarget_exam_nodes():
    """Attributing exam 0's logit must leave exam 1's nodes at ~0 importance."""
    torch.manual_seed(2)
    model = _model()
    inputs = _two_exam_inputs()
    attr = integrated_gradients(model, inputs, target_class=1, target_exam=0, steps=64)

    exam1_node_mask = inputs["subgraph_node_batch"] == 4
    assert exam1_node_mask.any()
    assert torch.allclose(
        attr.node_importance[exam1_node_mask],
        torch.zeros(int(exam1_node_mask.sum())),
        atol=1e-6,
    )


def test_ig_rejects_bad_baseline_shape():
    torch.manual_seed(0)
    model = _model()
    inputs = _single_exam_inputs()
    with pytest.raises(ValueError, match="baseline shape"):
        integrated_gradients(model, inputs, target_class=0, baseline=torch.zeros(3, 2), steps=8)


def test_ig_rejects_out_of_range_targets():
    torch.manual_seed(0)
    model = _model()
    inputs = _single_exam_inputs()
    with pytest.raises(ValueError, match="target_class"):
        integrated_gradients(model, inputs, target_class=99, steps=8)
    with pytest.raises(ValueError, match="target_exam"):
        integrated_gradients(model, inputs, target_class=0, target_exam=5, steps=8)


def test_ig_restores_training_state_on_exception(monkeypatch):
    """The try/finally must restore train mode even if the forward raises."""
    torch.manual_seed(0)
    model = _model()
    model.train()
    inputs = _single_exam_inputs()

    def boom(*args, **kwargs):
        raise RuntimeError("forward exploded")

    monkeypatch.setattr(model, "forward", boom)
    with pytest.raises(RuntimeError, match="forward exploded"):
        integrated_gradients(model, inputs, target_class=0, steps=8)
    assert model.training is True


def _ig_reference(
    model: HierGNNModel,
    inputs: dict,
    *,
    target_class: int,
    target_exam: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Sequential, one-forward-per-step Integrated Gradients reference.

    Returns ``(node_importance, feature_importance, prediction_gap,
    completeness_gap)`` from a plain midpoint Riemann sum, to check that the
    replicate-batched ``integrated_gradients`` computes the same values. Forces
    eval mode (and restores it after), matching what ``integrated_gradients``
    itself does internally. Otherwise this reference would run with dropout
    active while the batched implementation does not.
    """
    was_training = model.training
    model.eval()
    try:
        x = inputs["x"]
        baseline = torch.zeros_like(x)
        delta = x - baseline
        grad_sum = torch.zeros_like(x)
        for k in range(steps):
            alpha = (k + 0.5) / steps
            x_interp = (baseline + alpha * delta).detach().requires_grad_(True)
            logits = model(**{**inputs, "x": x_interp}).logits
            target_logit = logits[target_exam, target_class]
            (grad,) = torch.autograd.grad(target_logit, x_interp)
            grad_sum += grad
        avg_grad = grad_sum / steps
        feature_importance = delta * avg_grad
        node_importance = feature_importance.sum(dim=1)
        with torch.no_grad():
            f_x = model(**{**inputs, "x": x}).logits[target_exam, target_class]
            f_b = model(**{**inputs, "x": baseline}).logits[target_exam, target_class]
        prediction_gap = float(f_x - f_b)
        completeness_gap = abs(float(feature_importance.sum()) - prediction_gap)
    finally:
        model.train(was_training)
    return node_importance, feature_importance, prediction_gap, completeness_gap


def test_ig_batched_matches_sequential_reference():
    """Parity: the replicate-batched IG must match the sequential reference loop.

    Uses steps=8, micro_batch=3 to force uneven chunking (3, 3, 2) and a
    two-exam input so the target_exam offset arithmetic is exercised too.
    """
    torch.manual_seed(7)
    model = _model()
    inputs = _two_exam_inputs()

    node_imp, feat_imp, pred_gap, comp_gap = _ig_reference(
        model, inputs, target_class=2, target_exam=1, steps=8
    )
    attr = integrated_gradients(
        model, inputs, target_class=2, target_exam=1, steps=8, micro_batch=3
    )

    assert torch.allclose(attr.node_importance, node_imp, rtol=1e-4, atol=1e-6)
    assert torch.allclose(attr.feature_importance, feat_imp, rtol=1e-4, atol=1e-6)
    assert attr.prediction_gap == pytest.approx(pred_gap, abs=1e-6)
    assert attr.completeness_gap == pytest.approx(comp_gap, abs=1e-6)
    assert attr.completeness_gap < 0.05


def _ig_edges_reference(
    model: HierGNNModel,
    inputs: dict,
    *,
    target_class: int,
    target_exam: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Sequential, one-forward-per-step reference for the edge-attribution loop."""
    was_training = model.training
    model.eval()
    try:
        edge_attr = inputs["edge_attr"]
        baseline = torch.zeros_like(edge_attr)
        delta = edge_attr - baseline
        grad_sum = torch.zeros_like(edge_attr)
        for k in range(steps):
            alpha = (k + 0.5) / steps
            ea_interp = (baseline + alpha * delta).detach().requires_grad_(True)
            logits = model(**{**inputs, "edge_attr": ea_interp}).logits
            target_logit = logits[target_exam, target_class]
            (grad,) = torch.autograd.grad(target_logit, ea_interp)
            grad_sum += grad
        avg_grad = grad_sum / steps
        feature_importance = delta * avg_grad
        edge_importance = feature_importance.sum(dim=1)
        with torch.no_grad():
            f_x = model(**{**inputs, "edge_attr": edge_attr}).logits[target_exam, target_class]
            f_b = model(**{**inputs, "edge_attr": baseline}).logits[target_exam, target_class]
        prediction_gap = float(f_x - f_b)
        completeness_gap = abs(float(feature_importance.sum()) - prediction_gap)
    finally:
        model.train(was_training)
    return edge_importance, feature_importance, prediction_gap, completeness_gap
