"""Efficiency-focused regression tests for training/evaluation helpers."""

from __future__ import annotations

import pytest
import torch

from tests.unit.fixtures import MinimalBatch, MinimalModel
from training.evaluate import evaluate_loader
from training.hdf5_meta import DatasetFingerprint

_FP_AB = DatasetFingerprint(
    num_classes=2,
    num_leads=2,
    class_names=["A", "B"],
    sampling_rate=500,
    node_feature_names=["amplitude", "t_norm_beat"],
    edge_attr_names=["delta_t", "delta_y"],
    amplitude_normalization=None,
    node_reduction={
        "method": "uniform",
        "target_nodes": 30,
    },
)

pytestmark = pytest.mark.unit


def test_evaluate_gathers_predictions_once_per_pass(monkeypatch):
    """DDP eval should not all-gather once per batch."""
    calls = {"gather": 0}

    def fake_all_gather_concat(tensor: torch.Tensor) -> torch.Tensor:
        calls["gather"] += 1
        return tensor.detach().cpu()

    monkeypatch.setattr("training.inference.ddp.all_gather_concat", fake_all_gather_concat)
    monkeypatch.setattr("training.inference.ddp.all_reduce_sum", lambda tensor: tensor)
    monkeypatch.setattr("training.inference.ddp.is_main_process", lambda: True)

    loader = [
        MinimalBatch(y=torch.tensor([1.0, 0.0])),
        MinimalBatch(y=torch.tensor([0.0, 1.0])),
    ]

    report = evaluate_loader(
        MinimalModel(),
        loader,
        torch.device("cpu"),
        class_names=["A", "B"],
        threshold=0.5,
    )

    assert calls["gather"] == 2
    assert report.avg_loss > 0.0
    assert "f1/macro" in report.metrics


def test_train_survives_empty_train_loader(tmp_path, monkeypatch):
    """Regression: an empty train_loader must not raise on the unbound
    loop variable `step`; the epoch should complete and evaluation still run."""
    from config.train_models import OptimConfig, SubgraphRepresentationConfig, ThresholdConfig
    from training.train import train

    monkeypatch.setattr("training.inference.ddp.all_gather_concat", lambda t: t.detach().cpu())
    monkeypatch.setattr("training.inference.ddp.all_reduce_sum", lambda t: t)
    monkeypatch.setattr("training.inference.ddp.is_main_process", lambda: True)
    monkeypatch.setattr("training.train.ddp.is_main_process", lambda: True)
    monkeypatch.setattr("training.train.ddp.all_reduce_sum", lambda t: t)
    monkeypatch.setattr("training.train.ddp.is_distributed", lambda: False)

    result = train(
        model=MinimalModel(),
        train_loader=[],
        val_loader=[MinimalBatch(y=torch.tensor([1.0, 0.0]))],
        optim=OptimConfig(epochs=1, eval_val_every_epochs=1),
        threshold=ThresholdConfig(),
        class_names=["A", "B"],
        dataset_fingerprint=_FP_AB,
        device=torch.device("cpu"),
        ckpt_dir=tmp_path,
        subgraph_representation=SubgraphRepresentationConfig(),
    )

    assert result.epochs_completed == 1
