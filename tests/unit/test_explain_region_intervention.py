"""Public contracts for graph-local PR-QRS-T region interventions."""

from __future__ import annotations

import pytest
import torch

from explain.ecg_taxonomy import ECGRegion
from explain.region_intervention import (
    FixedRegionProtocol,
    RegionIntervals,
    fixed_region_intervals,
    graph_local_region_masks,
)

pytestmark = pytest.mark.unit


def test_fixed_region_protocols_preserve_legacy_and_standardized_t_windows():
    r_peak_t = torch.tensor([0.5, 0.7])

    legacy_1davb = fixed_region_intervals(
        r_peak_t,
        class_name="1dAVb",
        protocol=FixedRegionProtocol.LEGACY,
    )
    legacy_rbbb = fixed_region_intervals(
        r_peak_t,
        class_name="RBBB",
        protocol=FixedRegionProtocol.LEGACY,
    )
    standardized = fixed_region_intervals(
        r_peak_t,
        class_name="LBBB",
        protocol=FixedRegionProtocol.STANDARDIZED,
    )

    assert torch.allclose(legacy_1davb.starts[0], torch.tensor([0.15, 0.44, 0.65]))
    assert torch.allclose(legacy_1davb.ends[0], torch.tensor([0.43, 0.62, 0.93]))
    assert torch.allclose(legacy_rbbb.ends[0], torch.tensor([0.43, 0.62, 0.83]))
    assert torch.allclose(standardized.ends[0], torch.tensor([0.43, 0.62, 0.95]))
    assert torch.equal(standardized.available, torch.ones((2, 3), dtype=torch.bool))

    legacy_boundary = graph_local_region_masks(
        torch.tensor([0.43]),
        torch.tensor([0]),
        legacy_1davb,
    )
    assert legacy_boundary.pr.item() is True
    assert legacy_boundary.qrs.item() is False


def test_graph_local_masks_assign_qrs_onset_only_to_qrs_and_clip_to_each_subgraph():
    intervals = RegionIntervals(
        starts=torch.tensor(
            [
                [0.0, 1.0, 3.0],
                [10.0, 11.0, 13.0],
            ]
        ),
        ends=torch.tensor(
            [
                [1.0, 2.0, 4.0],
                [11.0, 12.0, 14.0],
            ]
        ),
        available=torch.ones((2, 3), dtype=torch.bool),
    )
    node_t = torch.tensor([0.5, 1.0, 1.5, 3.5, 10.5, 11.0, 11.5, 13.5])
    node_subgraph = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    masks = graph_local_region_masks(node_t, node_subgraph, intervals)

    assert torch.equal(
        masks[ECGRegion.PR],
        torch.tensor([True, False, False, False, True, False, False, False]),
    )
    assert torch.equal(
        masks[ECGRegion.QRS],
        torch.tensor([False, True, True, False, False, True, True, False]),
    )
    assert torch.equal(
        masks[ECGRegion.T],
        torch.tensor([False, False, False, True, False, False, False, True]),
    )
    assert not torch.any(masks.pr & masks.qrs)
    assert not torch.any(masks.pr & masks.t)
    assert not torch.any(masks.qrs & masks.t)


def test_adjacent_beats_cannot_claim_nodes_from_each_others_intervals():
    intervals = RegionIntervals(
        starts=torch.tensor([[0.0, 0.1, 0.3], [0.4, 0.5, 0.7]]),
        ends=torch.tensor([[0.1, 0.2, 0.4], [0.5, 0.6, 0.8]]),
        available=torch.ones((2, 3), dtype=torch.bool),
    )
    node_t = torch.tensor([0.45, 0.15])
    node_subgraph = torch.tensor([0, 1])

    masks = graph_local_region_masks(node_t, node_subgraph, intervals)

    assert not masks.for_coalition(set()).any()
    assert not masks.for_coalition({ECGRegion.PR, ECGRegion.QRS, ECGRegion.T}).any()


def test_unavailable_subgraph_remains_unmasked_without_imputation():
    intervals = RegionIntervals(
        starts=torch.tensor([[0.0, 1.0, 3.0], [10.0, 11.0, 13.0]]),
        ends=torch.tensor([[1.0, 2.0, 4.0], [11.0, 12.0, 14.0]]),
        available=torch.tensor([[True, True, True], [False, False, False]]),
    )
    node_t = torch.tensor([0.5, 1.5, 3.5, 10.5, 11.5, 13.5])
    node_subgraph = torch.tensor([0, 0, 0, 1, 1, 1])

    masks = graph_local_region_masks(node_t, node_subgraph, intervals)

    assert torch.equal(
        masks.for_coalition({ECGRegion.PR, ECGRegion.QRS, ECGRegion.T}),
        torch.tensor([True, True, True, False, False, False]),
    )


def test_region_intervals_reject_overlapping_qrs_and_t():
    with pytest.raises(ValueError, match="QRS and T intervals must be disjoint"):
        RegionIntervals(
            starts=torch.tensor([[0.0, 1.0, 2.0]]),
            ends=torch.tensor([[1.0, 2.0, 3.0]]),
            available=torch.ones((1, 3), dtype=torch.bool),
        )


def test_per_region_availability_deletes_marked_regions_and_keeps_unmarked_t_intact():
    intervals = RegionIntervals(
        starts=torch.tensor([[0.0, 1.0, 3.0], [10.0, 11.0, float("nan")]]),
        ends=torch.tensor([[1.0, 2.0, 4.0], [11.0, 12.0, float("nan")]]),
        available=torch.tensor([[True, True, True], [True, True, False]]),
    )
    node_t = torch.tensor([0.5, 1.5, 3.5, 10.5, 11.5, 13.5])
    node_subgraph = torch.tensor([0, 0, 0, 1, 1, 1])

    masks = graph_local_region_masks(node_t, node_subgraph, intervals)

    assert torch.equal(masks[ECGRegion.PR], torch.tensor([True, False, False, True, False, False]))
    assert torch.equal(masks[ECGRegion.QRS], torch.tensor([False, True, False, False, True, False]))
    assert torch.equal(masks[ECGRegion.T], torch.tensor([False, False, True, False, False, False]))


def test_region_intervals_ignore_overlap_between_unavailable_neighbors():
    intervals = RegionIntervals(
        starts=torch.tensor([[0.0, 1.0, 2.0]]),
        ends=torch.tensor([[1.0, 2.5, 3.0]]),
        available=torch.tensor([[True, True, False]]),
    )
    assert intervals.num_subgraphs == 1
