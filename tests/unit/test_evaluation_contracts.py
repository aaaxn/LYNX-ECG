from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from training import PRIMARY_METRIC_KEYS
from training.evaluate import evaluate_loader, require_checkpoint_epoch, require_final_metrics
from training.inference import GatheredPredictions

pytestmark = pytest.mark.unit


def test_require_final_metrics_rejects_each_missing_report_metric() -> None:
    complete = dict(zip(PRIMARY_METRIC_KEYS, (0.5, 0.6, 0.4), strict=True))

    for missing in complete:
        incomplete = dict(complete)
        incomplete.pop(missing)
        with pytest.raises(ValueError, match=missing):
            require_final_metrics(incomplete)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "0.5", None, True])
def test_require_final_metrics_rejects_invalid_required_values(invalid) -> None:
    metrics = {
        "f1/macro": invalid,
        "auroc/macro": 0.6,
        "pr_auc/macro": 0.4,
    }

    with pytest.raises(ValueError, match="f1/macro"):
        require_final_metrics(metrics)


def test_require_final_metrics_preserves_complete_report() -> None:
    metrics = {
        "f1/macro": 0.5,
        "auroc/macro": 0.6,
        "pr_auc/macro": 0.4,
        "f1/AF": 0.3,
    }

    assert require_final_metrics(metrics) is metrics


@pytest.mark.parametrize("epoch", [None, -1, 1.0, True, "1"])
def test_require_checkpoint_epoch_rejects_missing_or_invalid_epoch(epoch) -> None:
    state = {} if epoch is None else {"epoch": epoch}

    with pytest.raises(ValueError, match="epoch"):
        require_checkpoint_epoch(state)


def test_require_checkpoint_epoch_accepts_non_negative_integer() -> None:
    assert require_checkpoint_epoch({"epoch": 0}) == 0


class _Model(nn.Module):
    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -0.1, 1.1])
def test_evaluate_loader_rejects_invalid_threshold(monkeypatch, threshold) -> None:
    predictions = GatheredPredictions(
        logits=np.array([[0.0, 0.0]]),
        labels=np.array([[0, 1]]),
        avg_loss=0.0,
    )
    monkeypatch.setattr(
        "training.evaluate.gather_predictions",
        lambda *_args, **_kwargs: predictions,
    )

    with pytest.raises(ValueError, match="threshold"):
        evaluate_loader(
            _Model(),
            [],
            torch.device("cpu"),
            class_names=["A", "B"],
            threshold=threshold,
        )
