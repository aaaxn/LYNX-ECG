"""Unit tests for ``training.train.build_optimizer``.

The benchmark script imports this builder so its step timings always measure
the real training-loop optimizer.
"""

from __future__ import annotations

import pytest
import torch

from config.train_models import OptimConfig
from training.train import build_optimizer

pytestmark = pytest.mark.unit


def test_build_uses_config_hyperparams():
    model = torch.nn.Linear(4, 2)
    optim_cfg = OptimConfig(lr=3e-4, weight_decay=1e-5)

    optimizer = build_optimizer(model, optim_cfg, torch.device("cpu"))

    assert isinstance(optimizer, torch.optim.AdamW)
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(3e-4)
    assert group["weight_decay"] == pytest.approx(1e-5)
