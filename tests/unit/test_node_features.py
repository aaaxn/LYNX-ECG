"""Unit tests for _build_node_features in dataset.builder."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.builder import _build_node_features


@pytest.mark.unit
class TestBuildNodeFeatures:
    def test_amplitude_only(self):
        amp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        node_t = np.array([0.0, 0.01, 0.02], dtype=np.float32)
        result = _build_node_features(
            amplitude=amp, node_t=node_t, beat_duration_sec=0.02, feature_names=["amplitude"]
        )
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result[:, 0], amp)

    def test_t_norm_beat(self):
        node_t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = _build_node_features(
            amplitude=np.ones(3, dtype=np.float32),
            node_t=node_t,
            beat_duration_sec=1.0,
            feature_names=["amplitude", "t_norm_beat"],
        )
        np.testing.assert_allclose(result[:, 1], [0.0, 0.5, 1.0], atol=1e-6)

    def test_t_norm_beat_uses_beat_duration_not_last_selected_node(self):
        """Regression: when the reducer omits the final beat sample, the
        denominator must be the true beat duration, not node_t[-1].

        Here the beat is 1.0 s long but the last selected node sits at 0.6 s.
        Normalising by node_t[-1]=0.6 would wrongly map it to 1.0; normalising
        by beat_duration_sec=1.0 keeps it at 0.6, comparable across beats.
        """
        node_t = np.array([0.0, 0.2, 0.6], dtype=np.float32)
        result = _build_node_features(
            amplitude=np.ones(3, dtype=np.float32),
            node_t=node_t,
            beat_duration_sec=1.0,
            feature_names=["amplitude", "t_norm_beat"],
        )
        np.testing.assert_allclose(result[:, 1], [0.0, 0.2, 0.6], atol=1e-6)
