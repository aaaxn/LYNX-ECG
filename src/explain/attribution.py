"""Integrated Gradients node attribution for the HierGNN ECG model.

The attention surfaced in :class:`models.hier_model.AttentionMaps` is intrinsic
and *class-agnostic*: it reflects how the model pools its hierarchy, not which
nodes drive a *specific* diagnosis. The GINE subgraph encoder, where per-node
(turning-point) information enters, has no attention at all. This module fills
that gap with a *class-specific* post-hoc attribution.

We use Integrated Gradients (Sundararajan, Taly & Yan, 2017) over the node
feature tensor ``x`` (``[amplitude, t_norm_beat]`` per node), holding the graph
structure fixed. IG is chosen over GNNExplainer/Captum because it needs no extra
dependency, yields signed per-node importance, and satisfies the *completeness*
axiom: the attributions sum to the prediction gap ``f(x) - f(baseline)``, which
gives a built-in correctness check (see ``NodeAttribution.completeness_gap``).

The attribution is computed for one ``(target_exam, target_class)`` at a time.
Because the model pools per exam, gradients of one exam's logit w.r.t. another
exam's nodes are structurally zero, so completeness is exact (up to Riemann error)
for a single-exam input and node importances for other exams are ~0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from explain._inputs import validate_input_keys
from explain.replicate import replicate_inputs
from models.hier_model import HierGNNModel


@dataclass
class NodeAttribution:
    """Result of an Integrated Gradients attribution for one exam and class.

    - ``node_importance`` ``[N]``: signed IG attribution per subgraph node,
      summed over the input feature dimensions. Positive pushes the target
      logit up; negative pushes it down.
    - ``feature_importance`` ``[N, in_channels]``: per-feature IG, before the
      sum over features (e.g. amplitude vs. ``t_norm_beat`` contribution).
    - ``target_class`` / ``target_exam``: what was attributed.
    - ``prediction_gap``: ``f(x) - f(baseline)`` for the target logit.
    - ``completeness_gap``: ``|sum(IG) - prediction_gap|``. Small relative to
      ``|prediction_gap|`` confirms the integration converged; a large value
      means too few ``steps`` (or a degenerate baseline).
    """

    node_importance: torch.Tensor
    feature_importance: torch.Tensor
    target_class: int
    target_exam: int
    prediction_gap: float
    completeness_gap: float


def integrated_gradients(
    model: HierGNNModel,
    inputs: dict[str, Any],
    *,
    target_class: int,
    target_exam: int = 0,
    baseline: torch.Tensor | None = None,
    steps: int = 64,
    micro_batch: int = 16,
) -> NodeAttribution:
    """Attribute the ``target_class`` logit of ``target_exam`` to input nodes.

    ``inputs`` is the keyword-argument dict accepted by
    :meth:`HierGNNModel.forward` (``x``, ``edge_index``, ..., ``batch_size``).
    Unknown keys are rejected.
    ``baseline`` defaults to an all-zeros node-feature tensor (a flat,
    origin-time reference beat); pass an explicit tensor of the same shape as
    ``inputs['x']`` to use a different reference. Attribution runs in eval mode
    so the result is deterministic (no dropout) and does not change the model.

    Uses the midpoint Riemann rule with ``steps`` samples along the straight
    path from ``baseline`` to ``x``. The ``steps`` interpolants are scored in
    chunks of up to ``micro_batch`` via :func:`explain.replicate.replicate_inputs`
    by replicating the (structurally identical) graph ``chunk_n`` times and
    varying only ``x`` per copy, instead of one sequential forward per step;
    this changes only wall-clock, not the returned values (see the parity test
    in ``tests/unit/test_explain_attribution.py``). Lower ``micro_batch`` to
    trade speed for peak memory.
    """
    validate_input_keys(inputs)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if micro_batch < 1:
        raise ValueError(f"micro_batch must be >= 1, got {micro_batch}")

    x: torch.Tensor = inputs["x"]
    batch_size = int(inputs["batch_size"])
    if not 0 <= target_exam < batch_size:
        raise ValueError(f"target_exam {target_exam} out of range [0, {batch_size})")
    if not 0 <= target_class < model.num_classes:
        raise ValueError(f"target_class {target_class} out of range [0, {model.num_classes})")

    if baseline is None:
        baseline = torch.zeros_like(x)
    elif baseline.shape != x.shape:
        raise ValueError(f"baseline shape {tuple(baseline.shape)} != x shape {tuple(x.shape)}")
    else:
        baseline = baseline.to(x.device)

    was_training = model.training
    model.eval()
    try:
        delta = x - baseline
        n_nodes, n_features = x.shape

        alphas = (torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps
        all_interp = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * delta.unsqueeze(0)

        grad_sum = torch.zeros_like(x)
        for start in range(0, steps, micro_batch):
            chunk_n = min(micro_batch, steps - start)
            rep_inputs = replicate_inputs(inputs, chunk_n)
            chunk = (
                all_interp[start : start + chunk_n]
                .reshape(chunk_n * n_nodes, n_features)
                .detach()
                .requires_grad_(True)
            )
            rep_inputs["x"] = chunk
            logits = model(**rep_inputs).logits
            copy_offsets = torch.arange(chunk_n, device=x.device) * batch_size
            target_logits = logits[target_exam + copy_offsets, target_class]
            (grad,) = torch.autograd.grad(target_logits.sum(), chunk)
            grad_sum += grad.reshape(chunk_n, n_nodes, n_features).sum(dim=0)

        avg_grad = grad_sum / steps
        feature_importance = delta * avg_grad
        node_importance = feature_importance.sum(dim=1)

        with torch.no_grad():
            endpoints_inputs = replicate_inputs(inputs, 2)
            endpoints_inputs["x"] = torch.cat([x, baseline], dim=0)
            endpoint_logits = model(**endpoints_inputs).logits
            f_x = endpoint_logits[target_exam, target_class]
            f_b = endpoint_logits[target_exam + batch_size, target_class]
        prediction_gap = float(f_x - f_b)
        completeness_gap = abs(float(feature_importance.sum()) - prediction_gap)
    finally:
        model.train(was_training)

    return NodeAttribution(
        node_importance=node_importance.detach(),
        feature_importance=feature_importance.detach(),
        target_class=target_class,
        target_exam=target_exam,
        prediction_gap=prediction_gap,
        completeness_gap=completeness_gap,
    )
