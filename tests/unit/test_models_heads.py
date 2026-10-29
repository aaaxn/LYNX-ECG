"""Unit tests for ClassificationHead."""

from __future__ import annotations

import pytest
import torch

from models.heads import ClassificationHead

pytestmark = pytest.mark.unit


def test_head_forward_shape():
    head = ClassificationHead(hidden_channels=64, num_classes=5)
    x = torch.randn(3, 64)
    out = head(x)
    assert out.shape == (3, 5)
    assert torch.isfinite(out).all()


def test_head_invalid_num_classes():
    with pytest.raises(ValueError, match="num_classes"):
        ClassificationHead(hidden_channels=16, num_classes=0)
