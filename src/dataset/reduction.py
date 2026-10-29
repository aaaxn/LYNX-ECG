"""Node-reduction strategies.

Each reduction *method* is a :class:`BeatReducer` that owns everything
method-specific: any lead-level pre-pass (e.g. the turning_points lowpass),
the per-beat selection, the output QC contract, the node-amplitude source, and
the filename token. The builder drives a single reducer uniformly, so a new
method is added by writing one class and registering it in :func:`build_reducer`,
not by threading ``if method == ...`` branches through the build pipeline, QC,
and filename derivation.

The selection algorithms themselves live in :mod:`dataset.reducers`; the
strategies here only wire them to config, the lead pre-pass, and QC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from config.models import NodeReductionConfig
from dataset.qc import QcContract
from dataset.reducers import reduce_none, reduce_turning_points, reduce_uniform


@dataclass(frozen=True)
class LeadPrep:
    """Lead-level precomputation shared by every beat of one lead.

    ``filtered`` is the full-lead filtered signal (turning_points lowpass), or
    ``None`` when the reducer needs no pre-pass. ``max_gap_samples`` is the
    resolved gap-fill threshold in samples, or ``None``.
    """

    filtered: np.ndarray | None = None
    max_gap_samples: int | None = None


@dataclass(frozen=True)
class LeadRejected:
    """A lead the reducer cannot process (e.g. too short to filter).

    ``reason`` is a canonical skip name (see :mod:`dataset.qc`); ``details`` is
    attached to the resulting lead-level ``SkipEvent``.
    """

    reason: str
    details: dict[str, int]


@dataclass(frozen=True)
class Beat:
    """Per-beat input to a reducer.

    ``seg`` is the raw beat segment; ``filtered_seg`` is the corresponding slice
    of :attr:`LeadPrep.filtered` (``None`` when no pre-pass ran).
    ``r_peak_local_idx`` is the R-peak position within the beat;
    ``max_gap_samples`` carries :attr:`LeadPrep.max_gap_samples` to the
    selection.
    """

    seg: np.ndarray
    r_peak_local_idx: int
    filtered_seg: np.ndarray | None = None
    max_gap_samples: int | None = None


class BeatReducer(ABC):
    """A node-reduction method. Stateless across leads; built once per build."""

    @property
    @abstractmethod
    def filename_token(self) -> str:
        """The reducer's token in the dataset filename (e.g. ``k30``, ``kall``)."""

    def prepare_lead(self, lead_signal: np.ndarray, sampling_rate: int) -> LeadPrep | LeadRejected:
        """Lead-level pre-pass. Default: nothing to precompute."""
        return LeadPrep()

    @abstractmethod
    def select(self, beat: Beat) -> np.ndarray:
        """Return the selected local node indices ``[int64]`` for one beat."""

    @abstractmethod
    def qc_contract(self, beat: Beat) -> QcContract:
        """The output contract the validator enforces for this beat."""

    def node_amplitude(self, beat: Beat, selected: np.ndarray) -> np.ndarray:
        """Amplitude feature for the selected nodes. Default: raw beat signal."""
        return beat.seg[selected].astype(np.float32, copy=False)


class NoneReducer(BeatReducer):
    """Keep every sample. Output count equals the beat length."""

    @property
    def filename_token(self) -> str:
        return "kall"

    def select(self, beat: Beat) -> np.ndarray:
        return reduce_none(beat.seg)

    def qc_contract(self, beat: Beat) -> QcContract:
        return QcContract(expected_count=int(beat.seg.shape[0]))


class UniformReducer(BeatReducer):
    """Evenly spaced samples; a fixed node count."""

    def __init__(self, target_nodes: int) -> None:
        self.target_nodes = target_nodes

    @property
    def filename_token(self) -> str:
        return f"k{self.target_nodes}"

    def select(self, beat: Beat) -> np.ndarray:
        return reduce_uniform(beat.seg, self.target_nodes)

    def qc_contract(self, beat: Beat) -> QcContract:
        n = int(beat.seg.shape[0])
        return QcContract(expected_count=min(self.target_nodes, n))


class TurningPointsReducer(BeatReducer):
    """Local extrema of a lowpass-filtered signal; variable node count.

    The lowpass signal only selects the nodes. The node amplitude is the
    original (unfiltered) beat sample, the default
    :meth:`BeatReducer.node_amplitude`.
    """

    def __init__(self, *, cutoff_hz: float, max_gap_ms: float | None) -> None:
        self.cutoff_hz = cutoff_hz
        self.max_gap_ms = max_gap_ms

    @property
    def filename_token(self) -> str:
        # "original" names the node-amplitude source. The paper's dataset file
        # names contain it, so the token keeps it.
        token = f"{self.cutoff_hz:g}hz-original"
        if self.max_gap_ms is not None:
            token += f"-mgap{self.max_gap_ms:g}ms"
        return token

    def prepare_lead(self, lead_signal: np.ndarray, sampling_rate: int) -> LeadPrep | LeadRejected:
        from scipy.signal import sosfiltfilt

        from dataset.filtering import cached_lowpass_sos

        sos = cached_lowpass_sos(sampling_rate=int(sampling_rate), cutoff_hz=float(self.cutoff_hz))
        max_ntaps = 2 * sos.shape[0] + 1
        min_signal_len = max_ntaps * 3 + 1
        if lead_signal.size < min_signal_len:
            return LeadRejected(
                reason="signal_too_short_for_filter",
                details={
                    "signal_length": int(lead_signal.size),
                    "min_signal_len": int(min_signal_len),
                },
            )
        filtered = sosfiltfilt(sos, lead_signal).astype(np.float32)
        max_gap_samples: int | None = None
        if self.max_gap_ms is not None:
            max_gap_samples = max(1, round(self.max_gap_ms * sampling_rate / 1000.0))
        return LeadPrep(filtered=filtered, max_gap_samples=max_gap_samples)

    def select(self, beat: Beat) -> np.ndarray:
        if beat.filtered_seg is None:
            raise ValueError(
                "turning_points requires a filtered signal; prepare_lead must run first"
            )
        return reduce_turning_points(
            beat.filtered_seg,
            beat.r_peak_local_idx,
            max_gap_samples=beat.max_gap_samples,
        )

    def qc_contract(self, beat: Beat) -> QcContract:
        return QcContract(expected_count=None, anchors=(beat.r_peak_local_idx,))


def build_reducer(config: NodeReductionConfig) -> BeatReducer:
    """Construct the :class:`BeatReducer` for a validated node-reduction config.

    This is the single method-dispatch point; every other part of the pipeline
    drives the returned reducer uniformly. The ``assert``s document invariants
    already enforced by ``NodeReductionConfig`` validation (the required
    sub-config is present for each method).
    """
    method = config.method
    if method == "none":
        return NoneReducer()
    if method == "uniform":
        assert config.uniform is not None
        return UniformReducer(config.uniform.target_nodes)
    if method == "turning_points":
        assert config.turning_points is not None
        return TurningPointsReducer(
            cutoff_hz=config.turning_points.cutoff_hz,
            max_gap_ms=config.turning_points.max_gap_ms,
        )
    raise ValueError(f"Unknown reduction method: {method!r}")
