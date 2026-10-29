"""Unit tests for clinical-region alignment (explain.clinical_alignment)."""

from __future__ import annotations

import math

import pytest
import torch

from explain.clinical_alignment import (
    CLINICAL_REGIONS,
    FIXED_PR_WINDOW,
    FIXED_QRS_WINDOW,
    AlignmentScore,
    RegionWindow,
    decoy_region_window,
    expand_subgraph_scores_to_nodes,
    region_alignment,
    region_node_mask,
    region_window,
)
from explain.ecg_taxonomy import MORPHOLOGY_CLASSES

pytestmark = pytest.mark.unit


def test_clinical_regions_cover_all_six_classes():
    assert set(CLINICAL_REGIONS) == {"1dAVb", "RBBB", "LBBB", "AF", "SB", "ST"}


def test_only_morphology_classes_have_a_region_window():
    for cls in ("1dAVb", "RBBB", "LBBB"):
        assert region_window(cls) == CLINICAL_REGIONS[cls]
    for cls in ("AF", "SB", "ST", "NORM"):
        assert region_window(cls) is None


def test_region_window_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lo < hi"):
        RegionWindow(0.1, -0.1)


def test_decoy_region_window_uses_the_other_fiducial_region():
    assert decoy_region_window("1dAVb") == FIXED_QRS_WINDOW
    assert decoy_region_window("RBBB") == FIXED_PR_WINDOW
    assert decoy_region_window("LBBB") == FIXED_PR_WINDOW
    assert decoy_region_window("AF") is None


def test_decoy_region_window_never_matches_the_expected_window():
    for cls in MORPHOLOGY_CLASSES:
        assert decoy_region_window(cls) != region_window(cls)


def test_region_node_mask_selects_window_around_rpeak():
    beat_r_peak_t = torch.tensor([0.3, 0.5])
    node_beat = torch.tensor([0, 0, 0, 1, 1])
    node_t = torch.tensor([0.0, 0.28, 0.40, 0.10, 0.46])
    window = RegionWindow(-0.06, 0.12)
    mask = region_node_mask(node_t, node_beat, beat_r_peak_t, window)
    assert mask.tolist() == [False, True, True, False, True]


def test_region_node_mask_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="differ"):
        region_node_mask(
            torch.zeros(3),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(1),
            RegionWindow(-0.1, 0.1),
        )


def test_region_node_mask_rejects_out_of_range_beat():
    with pytest.raises(ValueError, match="references beat"):
        region_node_mask(
            torch.zeros(2),
            torch.tensor([0, 3]),
            torch.zeros(2),
            RegionWindow(-0.1, 0.1),
        )


def test_alignment_lift_above_one_when_importance_concentrates_in_region():
    region = torch.tensor([True, True, False, False, False, False, False, False, False, False])
    importance = torch.tensor([5.0, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    s = region_alignment(importance, region)
    assert isinstance(s, AlignmentScore)
    assert s.region_node_fraction == pytest.approx(0.2)
    assert s.lift > 1.0
    assert s.lift == pytest.approx((10.0 / 10.8) / 0.2, rel=1e-5)


def test_alignment_lift_about_one_for_uniform_importance():
    region = torch.tensor([True, True, False, False, False, False, False, False, False, False])
    importance = torch.ones(10)
    s = region_alignment(importance, region)
    assert s.lift == pytest.approx(1.0, abs=1e-6)


def test_alignment_lift_below_one_when_importance_avoids_region():
    region = torch.tensor([True, True, False, False, False, False, False, False, False, False])
    importance = torch.tensor([0.1, 0.1, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    s = region_alignment(importance, region)
    assert s.lift < 1.0


def test_alignment_uses_importance_magnitude_not_sign():
    region = torch.tensor([True, True, False, False])
    pos = torch.tensor([3.0, 3.0, 1.0, 1.0])
    neg = torch.tensor([-3.0, -3.0, -1.0, -1.0])
    assert region_alignment(pos, region).lift == pytest.approx(region_alignment(neg, region).lift)


def test_alignment_nan_when_no_region_nodes():
    region = torch.zeros(5, dtype=torch.bool)
    s = region_alignment(torch.ones(5), region)
    assert math.isnan(s.lift)
    assert s.n_region_nodes == 0


def test_alignment_nan_when_zero_total_importance():
    region = torch.tensor([True, False, False])
    s = region_alignment(torch.zeros(3), region)
    assert math.isnan(s.lift)
    assert math.isnan(s.region_importance_fraction)


def test_alignment_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        region_alignment(torch.ones(4), torch.ones(3, dtype=torch.bool))


def test_alignment_rejects_non_bool_mask():
    with pytest.raises(ValueError, match="must be bool"):
        region_alignment(torch.ones(3), torch.ones(3))


def test_expand_subgraph_scores_to_nodes():
    subgraph_scores = torch.tensor([10.0, 20.0, 30.0])
    subgraph_node_batch = torch.tensor([0, 0, 1, 1, 1, 2])
    node_scores = expand_subgraph_scores_to_nodes(subgraph_scores, subgraph_node_batch)
    assert torch.equal(node_scores, torch.tensor([10.0, 10.0, 20.0, 20.0, 20.0, 30.0]))
