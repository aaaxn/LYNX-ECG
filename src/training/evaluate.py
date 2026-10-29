"""Loader evaluation and final test evaluation.

:func:`evaluate_loader` is the single inference→metrics path shared by
the training loop's val/peek evals, the final test eval, and the ``evaluate``
CLI. :func:`evaluate_test` wraps it with checkpoint loading for the
post-training test pass.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training import ddp
from training.checkpoint import load_checkpoint, load_model_state
from training.inference import gather_predictions
from training.metrics import PRIMARY_METRIC_KEYS, compute_metrics


def require_final_metrics(
    metrics: dict[str, float], *, label: str = "final evaluation"
) -> dict[str, float]:
    """Require the macro metrics needed by a final report."""
    missing = [key for key in PRIMARY_METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"{label} is missing required metric keys: {missing}")
    for key in PRIMARY_METRIC_KEYS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{label} metric {key!r} must be a finite real number")
    return metrics


def require_checkpoint_epoch(state: dict, *, label: str = "checkpoint") -> int:
    """Require a non-negative integer save epoch in a final-eval checkpoint."""
    epoch = state.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"{label} is missing a valid non-negative integer 'epoch'")
    return epoch


@dataclass(frozen=True)
class EvalReport:
    """Result of :func:`evaluate_loader`."""

    metrics: dict[str, float]
    avg_loss: float


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    class_names: Sequence[str],
    threshold: float = 0.5,
    desc: str = "eval",
) -> EvalReport:
    """Gather predictions (DDP-aware) and score them at one threshold for every class.

    An empty loader yields an empty metric dict (matching the training loop's
    prior behaviour) while still returning a valid ``avg_loss``.
    """
    if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be finite and in [0, 1], got {threshold}")
    preds = gather_predictions(model, loader, device, num_classes=len(class_names), desc=desc)
    if preds.logits.shape[0] == 0:
        return EvalReport(metrics={}, avg_loss=preds.avg_loss)
    metrics = compute_metrics(
        preds.labels, preds.logits, class_names=class_names, threshold=threshold
    )
    return EvalReport(metrics=metrics, avg_loss=preds.avg_loss)


def evaluate_test(
    *,
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    ckpt_path: Path,
    class_names: Sequence[str],
    device: torch.device,
    threshold: float = 0.5,
    out_json: Path | None = None,
) -> dict[str, float]:
    """Load ``ckpt_path`` (typically ``best_f1_val.pt``) into ``model`` and evaluate the test split.

    ``threshold`` is applied to every class.
    """
    state = load_checkpoint(ckpt_path)
    epoch = require_checkpoint_epoch(state, label=f"checkpoint {ckpt_path}")
    base_model = load_model_state(model, state)
    base_model.to(device)

    report = evaluate_loader(
        base_model, test_loader, device, class_names=class_names, threshold=threshold, desc="test"
    )
    require_final_metrics(report.metrics)
    if out_json is not None and ddp.is_main_process():
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w") as f:
            json.dump(
                {
                    "ckpt_path": str(ckpt_path),
                    "threshold": threshold,
                    "metrics": report.metrics,
                    "epoch_at_save": epoch,
                },
                f,
                indent=2,
            )
    return report.metrics
