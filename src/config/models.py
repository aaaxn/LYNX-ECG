"""Pydantic models for the current dataset-build configuration contract.

All models use ``extra="forbid"`` so config typos are errors, not silent.
Cross-field rules are enforced via ``model_validator(mode="after")``.
Warnings for redundant-but-valid configs use ``warnings.warn``; semantic
violations raise ``ValueError`` (which pydantic surfaces as ``ValidationError``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.vocabulary import RPeakPolarityMode


def _default_edge_attr_names() -> list[Literal["delta_t", "delta_y"]]:
    return ["delta_t", "delta_y"]


def _default_node_features() -> list[Literal["amplitude", "t_norm_beat"]]:
    return ["amplitude", "t_norm_beat"]


class UniformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_nodes: int = Field(gt=0)


class TurningPointsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cutoff_hz: float = Field(default=40.0, gt=0.0)
    max_gap_ms: float | None = Field(default=None, gt=0.0)


class NodeReductionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["none", "uniform", "turning_points"]

    uniform: UniformConfig | None = None
    turning_points: TurningPointsConfig | None = None

    @model_validator(mode="after")
    def _validate_method_subconfig(self) -> NodeReductionConfig:
        method_to_field = {
            "uniform": "uniform",
            "turning_points": "turning_points",
        }

        if self.method == "none":
            for field_name in method_to_field.values():
                if getattr(self, field_name) is not None:
                    warnings.warn(
                        f"{field_name} sub-config is ignored when method='none'",
                        stacklevel=2,
                    )
            return self

        expected_field = method_to_field.get(self.method)
        if expected_field is None:
            return self

        if getattr(self, expected_field) is None:
            raise ValueError(f"method={self.method!r} requires a '{expected_field}' sub-config")

        for field_name in method_to_field.values():
            if field_name != expected_field and getattr(self, field_name) is not None:
                raise ValueError(
                    f"method={self.method!r} does not accept '{field_name}' sub-config"
                )

        return self


class SignalFilterConfig(BaseModel):
    """Source-level pre-filter applied to each lead at its native rate, before
    R-peak detection, resampling, and graph construction.

    A zero-phase Butterworth band-pass. Every paper dataset config uses
    highpass 0.5 Hz, lowpass 40 Hz, order 4. Leave unset (``None``) to build
    graphs from the unfiltered source signal.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["bandpass_butterworth_zero_phase"] = "bandpass_butterworth_zero_phase"
    highpass_hz: float = Field(gt=0.0)
    lowpass_hz: float = Field(gt=0.0)
    order: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> SignalFilterConfig:
        if not (self.highpass_hz < self.lowpass_hz):
            raise ValueError(
                "signal_filter requires highpass_hz < lowpass_hz, got "
                f"highpass_hz={self.highpass_hz} >= lowpass_hz={self.lowpass_hz}"
            )
        return self


class AmplitudeNormalizationConfig(BaseModel):
    """Per-exam amplitude rescaling applied as a source transform, after the
    optional band-pass and resample and before R-peak detection and graph
    construction.

    Divides every lead of an exam by a single per-exam scale so the node
    amplitude feature lives in a gain-invariant canonical scale, independent of
    the storage units / recording gain of the source. Because one scalar is
    applied to all leads, inter-lead amplitude ratios are preserved (and the
    visibility-graph topology, which is invariant to a positive uniform scale,
    is unchanged, only the stored amplitude feature value differs).

    ``p95_lead_median``: the scale is the median across ``graph_leads`` of each
    lead's 95th-percentile absolute amplitude. The inner p95 is robust to
    isolated spikes within a lead; the outer median is robust to a single
    dead/noisy lead. Leads whose index is out of the signal's bounds are
    excluded from the statistic (they are reported per-lead downstream).

    Leave unset (``None``) to store raw source amplitudes. Enabling it changes
    the meaning of ``qc.beat_min_amplitude`` from an absolute physical threshold
    to one relative to the exam's canonical scale.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["p95_lead_median"] = "p95_lead_median"


class GraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_attr_names: list[Literal["delta_t", "delta_y"]] = Field(
        default_factory=_default_edge_attr_names
    )


class RPeakDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    polarity_mode: RPeakPolarityMode = "auto"


class QCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_valid_beats_per_lead: int = Field(default=3, ge=1)
    min_valid_leads_per_exam: int = Field(default=1, ge=1)
    rpeak_min_count: int = Field(default=3, ge=3)
    rpeak_hr_range_bpm: tuple[float, float] = (30.0, 240.0)
    beat_rr_range_seconds: tuple[float, float] = (0.25, 4.0)
    beat_min_amplitude: float = Field(default=1e-4, gt=0)
    min_samples_per_beat: int = Field(default=3, ge=2)
    skip_duration_check_for_boundary_beats: bool = True

    @model_validator(mode="after")
    def _validate_ranges(self) -> QCConfig:
        lo, hi = self.rpeak_hr_range_bpm
        if not (lo > 0 and hi > lo):
            raise ValueError(
                f"rpeak_hr_range_bpm must satisfy 0 < lo < hi, got {self.rpeak_hr_range_bpm}"
            )
        lo, hi = self.beat_rr_range_seconds
        if not (lo > 0 and hi > lo):
            raise ValueError(
                f"beat_rr_range_seconds must satisfy 0 < lo < hi, got {self.beat_rr_range_seconds}"
            )
        return self


class ReportingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    write_summary_json: bool = True
    write_skipped_jsonl: bool = True
    reports_dir: str = "reports"


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smoke_during_build: bool = True
    format: Literal["png", "pdf"] = "png"
    dpi: int = Field(default=120, gt=0)


class BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: int = 16
    chunk_size: int = Field(default=4096, gt=0)
    sort_output: bool = True
    overwrite: bool = False

    @model_validator(mode="after")
    def _validate_jobs(self) -> BuildConfig:
        if self.jobs == 0 or self.jobs < -1:
            raise ValueError(f"jobs must be -1 or >= 1, got {self.jobs}")
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_source: Literal["ptbxl", "codetest", "code100"]
    source_variant: str
    sampling_rate: int
    graph_leads: list[int]

    resample_to_hz: int | None = Field(default=None, gt=0)

    limit_exams: int | None = None
    output_path: str | None = None

    node_features: list[Literal["amplitude", "t_norm_beat"]] = Field(
        default_factory=_default_node_features
    )

    signal_filter: SignalFilterConfig | None = None

    amplitude_normalization: AmplitudeNormalizationConfig | None = None

    node_reduction: NodeReductionConfig
    graph: GraphConfig = Field(default_factory=GraphConfig)
    rpeak_detection: RPeakDetectionConfig = Field(default_factory=RPeakDetectionConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)

    @property
    def effective_sampling_rate(self) -> int:
        """The rate the built graphs live at: ``resample_to_hz`` if set, else the source rate."""
        return self.resample_to_hz if self.resample_to_hz is not None else self.sampling_rate

    @model_validator(mode="after")
    def _validate_top_level(self) -> DatasetConfig:
        from dataset.sources import validate_source_variant

        validate_source_variant(self.dataset_source, self.source_variant, self.sampling_rate)

        if self.resample_to_hz is not None and self.resample_to_hz == self.sampling_rate:
            raise ValueError(
                f"resample_to_hz={self.resample_to_hz} equals the source sampling_rate; "
                "omit it (a no-op resample is a config mistake)"
            )

        if not self.graph_leads:
            raise ValueError("graph_leads cannot be empty")
        if any(lead < 0 for lead in self.graph_leads):
            raise ValueError(f"graph_leads must be non-negative, got {self.graph_leads}")
        if len(set(self.graph_leads)) != len(self.graph_leads):
            raise ValueError(f"graph_leads must be unique, got {self.graph_leads}")

        if self.limit_exams is not None and self.limit_exams <= 0:
            raise ValueError(f"limit_exams must be positive when set, got {self.limit_exams}")

        if "amplitude" not in self.node_features:
            raise ValueError(
                "node_features must contain 'amplitude' (column 0 is required for "
                "the delta_y edge attribute formula)"
            )
        if self.node_features[0] != "amplitude":
            raise ValueError("node_features[0] must be 'amplitude' (delta_y reads column 0)")
        if len(set(self.node_features)) != len(self.node_features):
            raise ValueError(f"node_features must be unique, got {self.node_features}")

        if (
            self.node_reduction.method == "turning_points"
            and self.node_reduction.turning_points is not None
        ):
            nyquist = self.effective_sampling_rate / 2.0
            cutoff = self.node_reduction.turning_points.cutoff_hz
            if cutoff >= nyquist:
                raise ValueError(
                    f"turning_points cutoff_hz={cutoff} must be below Nyquist "
                    f"frequency ({nyquist} Hz for effective sampling rate "
                    f"{self.effective_sampling_rate})"
                )

        if self.signal_filter is not None:
            nyquist = self.sampling_rate / 2.0
            if self.signal_filter.lowpass_hz >= nyquist:
                raise ValueError(
                    f"signal_filter lowpass_hz={self.signal_filter.lowpass_hz} must be "
                    f"below Nyquist frequency ({nyquist} Hz for source sampling rate "
                    f"{self.sampling_rate})"
                )

        return self


def load_config(path: Path | str) -> DatasetConfig:
    """Load and validate a config JSON file. Returns a DatasetConfig or raises."""
    p = Path(path)
    with p.open("r") as f:
        data = json.load(f)
    return DatasetConfig.model_validate(data)
