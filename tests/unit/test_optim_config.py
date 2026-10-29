"""Unit tests for the ``OptimConfig`` gradient-clipping validator.

The rule would otherwise only fail deep inside the training loop.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from config.train_models import OptimConfig

pytestmark = pytest.mark.unit


class TestGradClipNorm:
    def test_none_disables_clipping(self):
        assert OptimConfig(grad_clip_norm=None).grad_clip_norm is None

    @pytest.mark.parametrize("value", [0.0, -1.0, -0.5])
    def test_non_positive_rejected(self, value):
        with pytest.raises(ValidationError, match="grad_clip_norm"):
            OptimConfig(grad_clip_norm=value)

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_non_finite_rejected(self, value):
        with pytest.raises(ValidationError, match="grad_clip_norm"):
            OptimConfig(grad_clip_norm=value)
