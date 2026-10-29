"""Tests for ModelConfig.build, config-driven model construction."""

from __future__ import annotations

import pytest

from config.train_models import ModelConfig
from models.hier_model import HierGNNModel

pytestmark = [pytest.mark.unit]


def test_build_defaults():
    model = ModelConfig().build(num_classes=3, num_leads=2)

    assert isinstance(model, HierGNNModel)
    assert model.hparams["num_classes"] == 3
    assert model.hparams["num_leads"] == 2
    assert model.hparams["hidden_channels"] == 128


def test_build_passes_beat_encoder_block_to_the_model():
    config = ModelConfig.model_validate(
        {"hidden_channels": 16, "beat_encoder": {"layers": 2, "mlp_hidden": 32}}
    )
    model = config.build(num_classes=2, num_leads=3)

    assert model.hparams["beat_encoder_layers"] == 2
    assert model.hparams["beat_encoder_mlp_hidden"] == 32
    assert len(model.subgraph_encoder.convs) == 2
