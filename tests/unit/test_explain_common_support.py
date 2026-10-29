"""Common-support region definitions shared by all four methods."""

from __future__ import annotations

import pytest
import torch

from explain.common_support import (
    PROMINENCE_SUPPORT_METHOD_ORDER,
    AvailabilityRule,
    RegionDefinitionMethod,
    prominence_intervals_for_exam,
    prominence_support_intervals_for_exam,
)
from explain.delineation import DelineationFailureReason, DelineationMethod
from explain.delineation_qc import DelineationRecord, FiducialSamples
from explain.region_intervention import ECGRegion, graph_local_region_masks

pytestmark = pytest.mark.unit


def _record(
    graph_idx: int,
    *,
    method: DelineationMethod,
    candidate: str,
    complete: bool,
) -> DelineationRecord:
    if complete:
        fiducials = FiducialSamples(
            p_onset_sample=110,
            qrs_onset_sample=130,
            qrs_offset_sample=170,
            t_onset_sample=190,
            t_offset_sample=240,
        )
        reasons = []
        durations = {"pr": 20.0, "qrs": 40.0, "t": 50.0}
        available_regions = [ECGRegion.PR, ECGRegion.QRS, ECGRegion.T]
    else:
        fiducials = FiducialSamples(
            p_onset_sample=110,
            qrs_onset_sample=130,
            qrs_offset_sample=170,
            t_onset_sample=None,
            t_offset_sample=None,
        )
        reasons = [DelineationFailureReason.MISSING_T_OFFSET]
        durations = {"pr": 20.0, "qrs": 40.0}
        available_regions = [ECGRegion.PR, ECGRegion.QRS]
    return DelineationRecord(
        graph_idx=graph_idx,
        exam_idx=3,
        exam_id="exam-3",
        positive_classes=["RBBB"],
        lead=graph_idx,
        beat_index=0,
        start_sample=100,
        end_sample=260,
        boundary_type=0,
        r_peak_sample=150,
        polarity=1,
        method=method,
        candidate=candidate,
        signal_orientation="original",
        polarity_corrected=False,
        complete=complete,
        available_regions=available_regions,
        reasons=reasons,
        fiducials=fiducials,
        durations_ms=durations,
        suspicious_duration_flags=[],
        warning_messages=[],
        error_detail=None,
    )


def test_prominence_support_uses_every_complete_prominence_unit_without_dwt() -> None:
    records = [
        _record(
            10,
            method=DelineationMethod.PROMINENCE,
            candidate="selected",
            complete=True,
        ),
        _record(
            11,
            method=DelineationMethod.PROMINENCE,
            candidate="selected",
            complete=False,
        ),
    ]

    result = prominence_support_intervals_for_exam(
        records,
        graph_indices=(10, 11),
        beat_r_peak_t=torch.tensor([0.05, 0.05]),
        class_name="RBBB",
        sampling_rate=1000,
        selected_prominence_candidate="selected",
    )

    assert result.common_mask.tolist() == [True, False]
    assert tuple(result.intervals) == PROMINENCE_SUPPORT_METHOD_ORDER
    assert PROMINENCE_SUPPORT_METHOD_ORDER == (
        RegionDefinitionMethod.FIXED_LEGACY,
        RegionDefinitionMethod.FIXED_STANDARDIZED,
        RegionDefinitionMethod.PROMINENCE,
    )
    assert all(
        intervals.available.tolist() == [[True, True, True], [False, False, False]]
        for intervals in result.intervals.values()
    )
    node_t = torch.tensor([0.02, 0.04, 0.10, 0.02, 0.04, 0.10])
    node_subgraph = torch.tensor([0, 0, 0, 1, 1, 1])
    for intervals in result.intervals.values():
        masks = graph_local_region_masks(node_t, node_subgraph, intervals)
        unsupported = node_subgraph == 1
        assert all(not torch.any(masks[region] & unsupported) for region in ECGRegion)


def test_prominence_availability_rules_bracket_a_t_missing_beat():
    records = [
        _record(
            10,
            method=DelineationMethod.PROMINENCE,
            candidate="selected",
            complete=True,
        ),
        _record(
            11,
            method=DelineationMethod.PROMINENCE,
            candidate="selected",
            complete=False,
        ),
    ]

    beat_complete = prominence_intervals_for_exam(
        records,
        graph_indices=(10, 11),
        beat_r_peak_t=torch.tensor([0.05, 0.05]),
        sampling_rate=1000,
        selected_prominence_candidate="selected",
        availability_rule=AvailabilityRule.BEAT_COMPLETE,
    )
    per_region = prominence_intervals_for_exam(
        records,
        graph_indices=(10, 11),
        beat_r_peak_t=torch.tensor([0.05, 0.05]),
        sampling_rate=1000,
        selected_prominence_candidate="selected",
        availability_rule=AvailabilityRule.PER_REGION,
    )

    assert beat_complete.available.tolist() == [[True, True, True], [False, False, False]]
    assert per_region.available.tolist() == [[True, True, True], [True, True, False]]
    assert per_region.starts[1, 0].item() == pytest.approx(0.010)
    assert per_region.ends[1, 1].item() == pytest.approx(0.070)
