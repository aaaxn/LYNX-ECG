"""Unit tests for the paper-facing gate predicate in
scripts/explain/clinical_alignment_gate.py.

``class_alignment_ok`` decides ALIGNED/WEAK per class and, ANDed across
classes, the published GATE: PASS/REVIEW verdict -- so its boundary behavior
is pinned directly rather than only exercised end-to-end.
"""

from __future__ import annotations

import math

import clinical_alignment_gate as cag
import h5py
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


def test_ok_when_lift_beats_1_and_decoy():
    assert cag.class_alignment_ok(1.5, 1.0) is True


def test_fails_when_lift_does_not_strictly_beat_1():
    assert cag.class_alignment_ok(1.0, 0.5) is False


def test_fails_when_lift_equals_decoy():
    assert cag.class_alignment_ok(1.5, 1.5) is False


def test_nan_lift_is_false():
    assert cag.class_alignment_ok(math.nan, 0.5) is False


def test_empirical_feature_median_reads_small_dataset_exactly(tmp_path):
    path = tmp_path / "train.h5"
    node_x = np.array(
        [[-2.0, 0.0], [0.0, 0.25], [2.0, 0.75], [10.0, 1.0]],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as h5:
        h5.create_dataset("node_x", data=node_x)

    median = cag.empirical_feature_median(path, sample_nodes=100, seed=7)

    np.testing.assert_allclose(median, [1.0, 0.5])


def test_make_empirical_baseline_expands_feature_vector():
    inputs_x = torch.ones((3, 2), dtype=torch.float32)

    baseline = cag.make_ig_baseline(
        inputs_x,
        mode="empirical_feature_median",
        empirical_median=np.array([0.25, 0.5], dtype=np.float32),
    )

    assert baseline is not None
    torch.testing.assert_close(
        baseline,
        torch.tensor([[0.25, 0.5], [0.25, 0.5], [0.25, 0.5]]),
    )


def test_make_zero_baseline_delegates_to_integrated_gradients_default():
    inputs_x = torch.ones((3, 2), dtype=torch.float32)

    assert cag.make_ig_baseline(inputs_x, mode="zero", empirical_median=None) is None
