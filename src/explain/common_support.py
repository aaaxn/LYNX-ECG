"""Paired PR-QRS-T definitions on audited delineation support.

The four-method path uses the exact DWT/Prominence intersection. The separate
three-method path uses selected-Prominence completeness alone and never reads a
DWT record when constructing support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch

from explain.delineation import DelineationMethod
from explain.delineation_qc import DelineationRecord
from explain.ecg_taxonomy import REGION_ORDER, ECGRegion
from explain.region_intervention import (
    FixedRegionProtocol,
    RegionIntervals,
    fixed_region_intervals,
)


class RegionDefinitionMethod(StrEnum):
    """Region definitions compared by the common-support robustness suite."""

    FIXED_LEGACY = "fixed_legacy"
    FIXED_STANDARDIZED = "fixed_standardized"
    DWT = "dwt"
    PROMINENCE = "prominence"


METHOD_ORDER: tuple[RegionDefinitionMethod, ...] = tuple(RegionDefinitionMethod)
PROMINENCE_SUPPORT_METHOD_ORDER = (
    RegionDefinitionMethod.FIXED_LEGACY,
    RegionDefinitionMethod.FIXED_STANDARDIZED,
    RegionDefinitionMethod.PROMINENCE,
)


@dataclass(frozen=True)
class CommonSupportIntervals:
    """Per-method intervals sharing one exact lead-beat availability mask."""

    common_mask: torch.Tensor
    intervals: dict[RegionDefinitionMethod, RegionIntervals]


def _records_by_graph(
    records: list[DelineationRecord],
    *,
    method: DelineationMethod,
    candidate: str,
) -> dict[int, DelineationRecord]:
    selected = [
        record for record in records if record.method is method and record.candidate == candidate
    ]
    by_graph = {record.graph_idx: record for record in selected}
    if len(by_graph) != len(selected):
        raise ValueError(f"duplicate {method.value}/{candidate} records")
    return by_graph


class AvailabilityRule(StrEnum):
    """How a delineated beat's regions enter an intervention.

    ``beat_complete`` keeps a beat only when all three regions were marked, so
    an unmarked T withholds the beat's PR and QRS as well. ``per_region``
    keeps each marked region on its own, so a beat missing only its T still
    contributes its PR and its QRS. The two rules attenuate in opposite
    directions: ``beat_complete`` dilutes every region of an unmarked beat,
    while ``per_region`` deletes T from fewer beats than PR and QRS.
    """

    BEAT_COMPLETE = "beat_complete"
    PER_REGION = "per_region"


def _region_availability_matrix(records: list[DelineationRecord]) -> torch.Tensor:
    return torch.tensor(
        [[region in record.available_regions for region in REGION_ORDER] for record in records],
        dtype=torch.bool,
    )


def _broadcast_region_mask(mask: torch.Tensor) -> torch.Tensor:
    """Expand a per-unit availability mask to one owned column per region."""
    return mask[:, None].expand(-1, len(REGION_ORDER)).contiguous()


def _delineated_intervals(
    records: list[DelineationRecord],
    *,
    available: torch.Tensor,
    sampling_rate: int,
) -> RegionIntervals:
    """Build intervals from partial fiducials; ``available`` has shape [S, 3]."""
    if available.shape != (len(records), len(REGION_ORDER)):
        raise ValueError("available must have shape [S, 3] aligned with the records")
    starts = torch.full(
        (len(records), len(REGION_ORDER)),
        float("nan"),
        dtype=torch.float32,
        device=available.device,
    )
    ends = torch.full_like(starts, float("nan"))
    for local_index, record in enumerate(records):
        row = available[local_index]
        if not bool(row.any()):
            continue
        fiducials = record.fiducials
        origin = record.start_sample
        scale = 1.0 / sampling_rate
        region_bounds = {
            ECGRegion.PR: (fiducials.p_onset_sample, fiducials.qrs_onset_sample),
            ECGRegion.QRS: (fiducials.qrs_onset_sample, fiducials.qrs_offset_sample),
            ECGRegion.T: (fiducials.t_onset_sample, fiducials.t_offset_sample),
        }
        for region_index, region in enumerate(ECGRegion):
            if not bool(row[region_index]):
                continue
            start_sample, end_sample = region_bounds[region]
            if start_sample is None or end_sample is None:
                raise ValueError(
                    f"available region {region} lacks fiducials for graph_idx {record.graph_idx}"
                )
            starts[local_index, region_index] = (start_sample - origin) * scale
            ends[local_index, region_index] = (end_sample - origin) * scale
    return RegionIntervals(starts=starts, ends=ends, available=available.clone())


def prominence_support_intervals_for_exam(
    records: list[DelineationRecord],
    *,
    graph_indices: tuple[int, ...],
    beat_r_peak_t: torch.Tensor,
    class_name: str,
    sampling_rate: int,
    selected_prominence_candidate: str,
) -> CommonSupportIntervals:
    """Build fixed/Prominence definitions on selected-Prominence support only.

    A graph unit is available exactly when the selected Prominence candidate
    has a complete, ordered PR/QRS/T triplet. Fixed definitions are restricted
    to that same mask; no DWT record is read or required. Unsupported units
    remain intact under every compared method.
    """
    if not graph_indices or len(set(graph_indices)) != len(graph_indices):
        raise ValueError("graph_indices must be non-empty and unique")
    if beat_r_peak_t.shape != (len(graph_indices),):
        raise ValueError("beat_r_peak_t must align one-to-one with graph_indices")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if not selected_prominence_candidate:
        raise ValueError("selected prominence candidate must be non-empty")

    prominence_by_graph = _records_by_graph(
        records,
        method=DelineationMethod.PROMINENCE,
        candidate=selected_prominence_candidate,
    )
    if set(prominence_by_graph) != set(graph_indices):
        raise ValueError("selected Prominence records do not match the exam graph units")
    prominence_records = [prominence_by_graph[index] for index in graph_indices]
    for local_index, record in enumerate(prominence_records):
        expected_r = (record.r_peak_sample - record.start_sample) / sampling_rate
        if not math.isclose(
            float(beat_r_peak_t[local_index]),
            expected_r,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"stored R timing differs for graph_idx {record.graph_idx}")

    available = torch.tensor(
        [record.complete for record in prominence_records],
        dtype=torch.bool,
        device=beat_r_peak_t.device,
    )
    available_regions = _broadcast_region_mask(available)

    def fixed(protocol: FixedRegionProtocol) -> RegionIntervals:
        intervals = fixed_region_intervals(
            beat_r_peak_t,
            class_name=class_name,
            protocol=protocol,
        )
        return RegionIntervals(
            starts=intervals.starts,
            ends=intervals.ends,
            available=available_regions.clone(),
        )

    intervals = {
        RegionDefinitionMethod.FIXED_LEGACY: fixed(FixedRegionProtocol.LEGACY),
        RegionDefinitionMethod.FIXED_STANDARDIZED: fixed(FixedRegionProtocol.STANDARDIZED),
        RegionDefinitionMethod.PROMINENCE: _delineated_intervals(
            prominence_records,
            available=available_regions,
            sampling_rate=sampling_rate,
        ),
    }
    return CommonSupportIntervals(common_mask=available, intervals=intervals)


def _validated_prominence_records(
    records: list[DelineationRecord],
    *,
    graph_indices: tuple[int, ...],
    beat_r_peak_t: torch.Tensor,
    sampling_rate: int,
    selected_prominence_candidate: str,
) -> list[DelineationRecord]:
    if not graph_indices or len(set(graph_indices)) != len(graph_indices):
        raise ValueError("graph_indices must be non-empty and unique")
    if beat_r_peak_t.shape != (len(graph_indices),):
        raise ValueError("beat_r_peak_t must align one-to-one with graph_indices")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if not selected_prominence_candidate:
        raise ValueError("selected prominence candidate must be non-empty")
    prominence_by_graph = _records_by_graph(
        records,
        method=DelineationMethod.PROMINENCE,
        candidate=selected_prominence_candidate,
    )
    if set(prominence_by_graph) != set(graph_indices):
        raise ValueError("selected Prominence records do not match the exam graph units")
    prominence_records = [prominence_by_graph[index] for index in graph_indices]
    for local_index, record in enumerate(prominence_records):
        expected_r = (record.r_peak_sample - record.start_sample) / sampling_rate
        if not math.isclose(
            float(beat_r_peak_t[local_index]),
            expected_r,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"stored R timing differs for graph_idx {record.graph_idx}")
    return prominence_records


def prominence_intervals_for_exam(
    records: list[DelineationRecord],
    *,
    graph_indices: tuple[int, ...],
    beat_r_peak_t: torch.Tensor,
    sampling_rate: int,
    selected_prominence_candidate: str,
    availability_rule: AvailabilityRule,
) -> RegionIntervals:
    """Build Prominence-delineated intervals under one availability rule.

    The intervals are class-independent. Under ``beat_complete`` a subgraph
    enters only with a complete triplet; under ``per_region`` each marked
    region enters on its own. Units outside the rule's support keep every
    node in every coalition state.
    """
    prominence_records = _validated_prominence_records(
        records,
        graph_indices=graph_indices,
        beat_r_peak_t=beat_r_peak_t,
        sampling_rate=sampling_rate,
        selected_prominence_candidate=selected_prominence_candidate,
    )
    rule = AvailabilityRule(availability_rule)
    if rule is AvailabilityRule.BEAT_COMPLETE:
        complete = torch.tensor(
            [record.complete for record in prominence_records],
            dtype=torch.bool,
            device=beat_r_peak_t.device,
        )
        available = _broadcast_region_mask(complete)
    else:
        available = _region_availability_matrix(prominence_records).to(beat_r_peak_t.device)
    return _delineated_intervals(
        prominence_records,
        available=available,
        sampling_rate=sampling_rate,
    )
