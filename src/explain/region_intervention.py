"""Graph-local PR-QRS-T region definitions and node masks.

A region belongs to one lead-beat subgraph. Its effective node mask is the
intersection of the region interval with the nodes represented by that
subgraph; intervals never claim nodes owned by an adjacent beat. This preserves
the legacy fixed-window clipping semantics while giving coalition analyses
three disjoint players.
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from enum import StrEnum

import torch

from explain.clinical_alignment import (
    FIXED_PR_WINDOW,
    FIXED_QRS_WINDOW,
    FIXED_STANDARDIZED_T_WINDOW,
)
from explain.ecg_taxonomy import MORPHOLOGY_CLASSES, REGION_ORDER, ECGRegion


class FixedRegionProtocol(StrEnum):
    """Supported fixed-window definitions."""

    LEGACY = "legacy"
    STANDARDIZED = "standardized"


@dataclass(frozen=True)
class RegionIntervals:
    """Per-subgraph PR, QRS, and T intervals on one shared time axis.

    ``starts``, ``ends``, and ``available`` have shape ``[S, 3]`` in
    ``(PR, QRS, T)`` order. ``available`` marks each region of each subgraph
    on its own; unavailable cells may contain non-finite placeholders and
    produce no node masks. Beat-complete protocols mark all three regions of
    a subgraph together; per-region protocols may keep a subgraph's PR and
    QRS while its T stays unavailable. Bounds are closed to preserve the
    legacy fixed-window semantics. PR may end exactly at QRS onset; when it
    does, QRS owns the shared boundary node. QRS and T require a strict gap.
    """

    starts: torch.Tensor
    ends: torch.Tensor
    available: torch.Tensor

    def __post_init__(self) -> None:
        region_count = len(REGION_ORDER)
        if self.starts.ndim != 2 or self.starts.shape[1:] != (region_count,):
            raise ValueError(
                f"starts must have shape [S, {region_count}], got {tuple(self.starts.shape)}"
            )
        if self.ends.shape != self.starts.shape:
            raise ValueError(
                f"ends must match starts shape {tuple(self.starts.shape)}, "
                f"got {tuple(self.ends.shape)}"
            )
        if self.available.shape != self.starts.shape:
            raise ValueError(
                f"available must match starts shape {tuple(self.starts.shape)}, "
                f"got {tuple(self.available.shape)}"
            )
        if not self.starts.is_floating_point() or not self.ends.is_floating_point():
            raise ValueError("starts and ends must be floating-point tensors")
        if self.available.dtype != torch.bool:
            raise ValueError(f"available must be bool, got {self.available.dtype}")
        if not (self.starts.device == self.ends.device == self.available.device):
            raise ValueError("starts, ends, and available must be on the same device")

        starts = self.starts[self.available]
        ends = self.ends[self.available]
        if starts.numel() == 0:
            return
        if not torch.isfinite(starts).all() or not torch.isfinite(ends).all():
            raise ValueError("available region intervals must be finite")
        if torch.any(starts >= ends):
            raise ValueError("every available region interval requires start < end")
        pr_and_qrs = self.available[:, 0] & self.available[:, 1]
        if torch.any(self.ends[pr_and_qrs, 0] > self.starts[pr_and_qrs, 1]):
            raise ValueError("PR and QRS intervals must be disjoint")
        qrs_and_t = self.available[:, 1] & self.available[:, 2]
        if torch.any(self.ends[qrs_and_t, 1] >= self.starts[qrs_and_t, 2]):
            raise ValueError("QRS and T intervals must be disjoint")

    @property
    def num_subgraphs(self) -> int:
        return self.starts.shape[0]


@dataclass(frozen=True)
class RegionNodeMasks:
    """Disjoint graph-local node masks for PR, QRS, and T."""

    pr: torch.Tensor
    qrs: torch.Tensor
    t: torch.Tensor

    def __post_init__(self) -> None:
        if self.pr.shape != self.qrs.shape or self.pr.shape != self.t.shape:
            raise ValueError("PR, QRS, and T node masks must share one shape")
        if any(mask.dtype != torch.bool for mask in (self.pr, self.qrs, self.t)):
            raise ValueError("PR, QRS, and T node masks must be bool")
        if not (self.pr.device == self.qrs.device == self.t.device):
            raise ValueError("PR, QRS, and T node masks must be on the same device")
        if torch.any((self.pr & self.qrs) | (self.pr & self.t) | (self.qrs & self.t)):
            raise ValueError("PR, QRS, and T node masks must be disjoint")

    def __getitem__(self, region: ECGRegion) -> torch.Tensor:
        if region is ECGRegion.PR:
            return self.pr
        if region is ECGRegion.QRS:
            return self.qrs
        if region is ECGRegion.T:
            return self.t
        raise KeyError(region)

    def for_coalition(self, regions: Set[ECGRegion]) -> torch.Tensor:
        """Return the union mask for ``regions``; the empty coalition is all false."""
        coalition = torch.zeros_like(self.pr)
        for region in regions:
            coalition |= self[region]
        return coalition


def fixed_region_intervals(
    beat_r_peak_t: torch.Tensor,
    *,
    class_name: str,
    protocol: FixedRegionProtocol,
) -> RegionIntervals:
    """Build fixed graph-local intervals around each subgraph's stored R anchor.

    Both protocols use the existing P/PR and QRS approximations. The legacy T
    proxy is width-matched to the expected diagnostic region (PR for 1dAVb,
    QRS for bundle-branch blocks); the standardized T approximation is the
    common ``[+0.15, +0.45]`` second window.
    """
    if beat_r_peak_t.ndim != 1:
        raise ValueError(f"beat_r_peak_t must be one-dimensional, got {tuple(beat_r_peak_t.shape)}")
    if not beat_r_peak_t.is_floating_point():
        raise ValueError("beat_r_peak_t must be floating point")
    if not torch.isfinite(beat_r_peak_t).all():
        raise ValueError("beat_r_peak_t must be finite")
    if class_name not in MORPHOLOGY_CLASSES:
        raise ValueError(f"fixed PR-QRS-T regions require 1dAVb, RBBB, or LBBB; got {class_name!r}")

    protocol = FixedRegionProtocol(protocol)
    if protocol is FixedRegionProtocol.LEGACY:
        t_end = 0.43 if class_name == "1dAVb" else 0.33
    else:
        t_end = FIXED_STANDARDIZED_T_WINDOW.hi

    start_offsets = beat_r_peak_t.new_tensor(
        [FIXED_PR_WINDOW.lo, FIXED_QRS_WINDOW.lo, FIXED_STANDARDIZED_T_WINDOW.lo]
    )
    end_offsets = beat_r_peak_t.new_tensor([FIXED_PR_WINDOW.hi, FIXED_QRS_WINDOW.hi, t_end])
    starts = beat_r_peak_t[:, None] + start_offsets
    ends = beat_r_peak_t[:, None] + end_offsets
    available = torch.ones(
        (beat_r_peak_t.shape[0], len(REGION_ORDER)),
        dtype=torch.bool,
        device=beat_r_peak_t.device,
    )
    return RegionIntervals(starts=starts, ends=ends, available=available)


def graph_local_region_masks(
    node_t: torch.Tensor,
    node_subgraph: torch.Tensor,
    intervals: RegionIntervals,
) -> RegionNodeMasks:
    """Map per-subgraph intervals to disjoint masks aligned with graph nodes.

    ``node_t`` and interval bounds use the same time axis. Each node indexes
    only its own row through ``node_subgraph``, which makes clipping to the
    represented subgraph implicit and prevents adjacent beats from sharing a
    node.
    """
    if node_t.ndim != 1 or node_subgraph.shape != node_t.shape:
        raise ValueError(
            "node_t and node_subgraph must be one-dimensional tensors with equal shape"
        )
    if not node_t.is_floating_point():
        raise ValueError("node_t must be floating point")
    if node_subgraph.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"node_subgraph must be an integer tensor, got {node_subgraph.dtype}")
    if node_t.device != node_subgraph.device or node_t.device != intervals.starts.device:
        raise ValueError("node_t, node_subgraph, and intervals must be on the same device")
    if node_subgraph.numel():
        if int(node_subgraph.min()) < 0:
            raise ValueError("node_subgraph ids must be non-negative")
        if int(node_subgraph.max()) >= intervals.num_subgraphs:
            raise ValueError(
                f"node_subgraph references id {int(node_subgraph.max())}, but intervals "
                f"contain {intervals.num_subgraphs} subgraphs"
            )

    starts = intervals.starts[node_subgraph]
    ends = intervals.ends[node_subgraph]
    available = intervals.available[node_subgraph]

    within_intervals = available & (node_t[:, None] >= starts) & (node_t[:, None] <= ends)
    pr, qrs, t = within_intervals.unbind(dim=1)
    pr = pr & ~qrs
    return RegionNodeMasks(pr=pr, qrs=qrs, t=t)
