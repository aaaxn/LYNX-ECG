"""Per-class clinical-region alignment of node importance.

The evidence for the paper's central claim: does the model's node-level
importance (Integrated-Gradients attribution, or attention expanded to nodes via
:func:`expand_subgraph_scores_to_nodes`) concentrate in the
fiducial region that *defines* each diagnosis? If yes, the graph model is
clinically interpretable, not merely self-consistent.

Region windows are expressed relative to each beat's R-peak in seconds and
follow standard ECG criteria.

Scope: a within-beat region is only meaningful for the *morphology* classes
(1dAVb, RBBB, LBBB). The *rhythm* classes (AF, SB, ST) are diagnosed from
inter-beat rhythm (P-wave presence/absence pattern, RR regularity, rate), not a
single within-beat window, so they are marked :data:`RHYTHM` and excluded from
the node-region score. The beat-sequence and rate interventions in
``scripts/explain/rhythm_lead_audit.py`` test them instead; do not fabricate a
window.

The importance magnitude is what matters for alignment, so signed attributions
are reduced by absolute value before the concentration is measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from explain.ecg_taxonomy import EXPECTED_REGION, ECGRegion

RHYTHM: Literal["rhythm"] = "rhythm"


@dataclass(frozen=True)
class RegionWindow:
    """A time window relative to a beat's R-peak, in seconds.

    ``lo``/``hi`` are offsets from the R-peak (negative = before it). A window
    spanning the QRS therefore has ``lo < 0 < hi``.
    """

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo >= self.hi:
            raise ValueError(f"RegionWindow requires lo < hi, got lo={self.lo}, hi={self.hi}")


FIXED_PR_WINDOW = RegionWindow(-0.35, -0.07)
FIXED_QRS_WINDOW = RegionWindow(-0.06, 0.12)
FIXED_STANDARDIZED_T_WINDOW = RegionWindow(0.15, 0.45)

_MORPHOLOGY_WINDOWS: dict[ECGRegion, RegionWindow] = {
    ECGRegion.PR: FIXED_PR_WINDOW,
    ECGRegion.QRS: FIXED_QRS_WINDOW,
}

_DECOY_REGION: dict[ECGRegion, ECGRegion] = {
    ECGRegion.PR: ECGRegion.QRS,
    ECGRegion.QRS: ECGRegion.PR,
}

CLINICAL_REGIONS: dict[str, RegionWindow | Literal["rhythm"]] = {
    **{name: _MORPHOLOGY_WINDOWS[region] for name, region in EXPECTED_REGION.items()},
    "AF": RHYTHM,
    "SB": RHYTHM,
    "ST": RHYTHM,
}


def region_window(class_name: str) -> RegionWindow | None:
    """Return the fiducial window for ``class_name``, or ``None`` if it has no
    within-beat region, i.e. a rhythm class (:data:`RHYTHM`) or any class outside
    this morphology taxonomy. Prefer this over indexing
    :data:`CLINICAL_REGIONS` directly so callers never treat the :data:`RHYTHM`
    sentinel, or a class with no entry, as a window.
    """
    spec = CLINICAL_REGIONS.get(class_name)
    return spec if isinstance(spec, RegionWindow) else None


def decoy_region_window(class_name: str) -> RegionWindow | None:
    """Return the window of the other fiducial region, used as a decoy."""
    expected = EXPECTED_REGION.get(class_name)
    if expected is None:
        return None
    return _MORPHOLOGY_WINDOWS[_DECOY_REGION[expected]]


@dataclass
class AlignmentScore:
    """How concentrated node importance is in a class's fiducial region.

    - ``lift``: ``region_importance_fraction / region_node_fraction``. >1 means
      importance concentrates in the region more than a uniform spread would;
      ~1 is chance; <1 is anti-concentration. ``nan`` if undefined (no region
      nodes, or zero total importance).
    - ``region_importance_fraction``: share of total |importance| inside the region.
    - ``region_node_fraction``: share of nodes inside the region.
    - ``n_region_nodes`` / ``n_nodes``: counts behind the fractions.
    """

    lift: float
    region_importance_fraction: float
    region_node_fraction: float
    n_region_nodes: int
    n_nodes: int


def region_node_mask(
    node_t: torch.Tensor,
    node_beat: torch.Tensor,
    beat_r_peak_t: torch.Tensor,
    window: RegionWindow,
) -> torch.Tensor:
    """Mark nodes whose beat-local time falls in the window around their R-peak.

    - ``node_t`` ``[N]``: each node's time within its beat, **in seconds from beat
      start**. WARNING: this is NOT the model feature ``inputs['x'][:, 1]``, which
      is ``t_norm_beat`` (dimensionless, ~[0, 1]); using that here silently
      produces wrong masks. Obtain ``node_t`` (and ``beat_r_peak_t``) from the
      HDF5 metadata via :mod:`explain.ecg_timing`.
    - ``node_beat`` ``[N]``: each node's beat (subgraph) id.
    - ``beat_r_peak_t`` ``[num_beats]``: each beat's R-peak time (seconds from
      beat start), indexable by ``node_beat``.
    Returns a ``[N]`` boolean mask.
    """
    if node_t.shape != node_beat.shape:
        raise ValueError(
            f"node_t {tuple(node_t.shape)} and node_beat {tuple(node_beat.shape)} differ"
        )
    if node_beat.numel() and int(node_beat.max()) >= beat_r_peak_t.numel():
        raise ValueError(
            f"node_beat references beat {int(node_beat.max())} but beat_r_peak_t has "
            f"{beat_r_peak_t.numel()} entries"
        )
    r_per_node = beat_r_peak_t[node_beat]
    return (node_t >= r_per_node + window.lo) & (node_t <= r_per_node + window.hi)


def region_alignment(importance: torch.Tensor, region_mask: torch.Tensor) -> AlignmentScore:
    """Lift of importance mass in the region vs the region's share of nodes.

    ``importance`` ``[N]`` may be signed (e.g. Integrated Gradients); its
    absolute value is used, since importance *magnitude*, not sign, is what
    should land in the clinical region. ``region_mask`` ``[N]`` is boolean.
    """
    if importance.shape != region_mask.shape:
        raise ValueError(
            f"importance {tuple(importance.shape)} and region_mask "
            f"{tuple(region_mask.shape)} must match"
        )
    if region_mask.dtype != torch.bool:
        raise ValueError(f"region_mask must be bool, got {region_mask.dtype}")

    mag = importance.abs()
    n_nodes = mag.numel()
    n_region = int(region_mask.sum())
    total = float(mag.sum())
    region_total = float(mag[region_mask].sum())

    region_node_fraction = (n_region / n_nodes) if n_nodes else float("nan")
    region_importance_fraction = (region_total / total) if total > 0 else float("nan")
    if n_region == 0 or total == 0:
        lift = float("nan")
    else:
        lift = region_importance_fraction / region_node_fraction

    return AlignmentScore(
        lift=lift,
        region_importance_fraction=region_importance_fraction,
        region_node_fraction=region_node_fraction,
        n_region_nodes=n_region,
        n_nodes=n_nodes,
    )


def expand_subgraph_scores_to_nodes(
    subgraph_scores: torch.Tensor, subgraph_node_batch: torch.Tensor
) -> torch.Tensor:
    """Broadcast per-subgraph importance to per-node importance.

    Attention is computed at subgraph (beat) granularity; to score it as
    node-level importance, each node inherits its subgraph's score.
    ``subgraph_scores`` is ``[num_subgraphs]`` and the result is ``[N]``
    aligned with the node feature rows.
    """
    if subgraph_scores.dim() != 1:
        raise ValueError(f"subgraph_scores must be 1-D, got shape {tuple(subgraph_scores.shape)}")
    num_subgraphs = int(subgraph_node_batch.max()) + 1 if subgraph_node_batch.numel() else 0
    if subgraph_scores.numel() != num_subgraphs:
        raise ValueError(
            f"subgraph_scores has {subgraph_scores.numel()} entries but "
            f"subgraph_node_batch implies {num_subgraphs} subgraphs (ids 0..max); "
            f"ids must be contiguous from 0"
        )
    return subgraph_scores[subgraph_node_batch]
