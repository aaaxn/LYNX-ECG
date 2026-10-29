"""Labeled protocol and prominence-candidate overlays for delineation QC."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dataset.raw_signal import load_preprocessed_exam_signal
from dataset.sources import resolve_source_path
from ecg_artifact_io import staged_artifact
from explain.common_support import RegionDefinitionMethod
from explain.delineation import DelineationMethod
from explain.delineation_qc import DelineationRecord
from explain.delineation_workflow import DelineationQCReport
from explain.ecg_taxonomy import REGION_ORDER, TWELVE_LEAD_DISPLAY_NAMES
from explain.region_intervention import FixedRegionProtocol, fixed_region_intervals

METHOD_LABELS: dict[RegionDefinitionMethod, str] = {
    RegionDefinitionMethod.FIXED_LEGACY: "Legacy fixed",
    RegionDefinitionMethod.FIXED_STANDARDIZED: "Standardized fixed",
    RegionDefinitionMethod.DWT: "DWT",
    RegionDefinitionMethod.PROMINENCE: "Prominence",
}

REGION_COLORS: dict[str, str] = {
    "pr": "#0072B2",
    "qrs": "#D55E00",
    "t": "#009E73",
}

_BASE_COLUMNS = (
    METHOD_LABELS[RegionDefinitionMethod.FIXED_LEGACY],
    METHOD_LABELS[RegionDefinitionMethod.FIXED_STANDARDIZED],
    METHOD_LABELS[RegionDefinitionMethod.DWT],
)


def _fixed_regions(
    r_peak_sample: int,
    *,
    sampling_rate: int,
    class_name: str,
    protocol: FixedRegionProtocol,
) -> dict[str, tuple[float, float]]:
    r_peak_t = torch.tensor([r_peak_sample / sampling_rate], dtype=torch.float64)
    intervals = fixed_region_intervals(
        r_peak_t,
        class_name=class_name,
        protocol=protocol,
    )
    starts = intervals.starts[0].cpu().numpy() - r_peak_sample / sampling_rate
    ends = intervals.ends[0].cpu().numpy() - r_peak_sample / sampling_rate
    return {
        name: (float(starts[index]), float(ends[index])) for index, name in enumerate(REGION_ORDER)
    }


def _delineated_regions(record: DelineationRecord) -> dict[str, tuple[float, float]] | None:
    if not record.complete:
        return None
    p_onset, qrs_onset, qrs_offset, t_onset, t_offset = record.fiducials.complete_triplet()
    return {
        "pr": (
            float(p_onset - record.r_peak_sample),
            float(qrs_onset - record.r_peak_sample),
        ),
        "qrs": (
            float(qrs_onset - record.r_peak_sample),
            float(qrs_offset - record.r_peak_sample),
        ),
        "t": (
            float(t_onset - record.r_peak_sample),
            float(t_offset - record.r_peak_sample),
        ),
    }


def _seconds(
    regions_in_samples: dict[str, tuple[float, float]], sampling_rate: int
) -> dict[str, tuple[float, float]]:
    return {
        name: (start / sampling_rate, end / sampling_rate)
        for name, (start, end) in regions_in_samples.items()
    }


def _draw_regions(ax: plt.Axes, regions: dict[str, tuple[float, float]]) -> None:
    for name in REGION_ORDER:
        start, end = regions[name]
        ax.axvspan(start, end, color=REGION_COLORS[name], alpha=0.18, linewidth=0)
        ax.axvline(start, color=REGION_COLORS[name], linewidth=0.55, alpha=0.75)
        ax.axvline(end, color=REGION_COLORS[name], linewidth=0.55, alpha=0.75)


def _duration_label(regions: dict[str, tuple[float, float]]) -> str:
    values = {name: (end - start) * 1_000 for name, (start, end) in regions.items()}
    return " | ".join(f"{name.upper()} {values[name]:.0f} ms" for name in REGION_ORDER)


def _annotate_complete(ax: plt.Axes, regions: dict[str, tuple[float, float]]) -> None:
    ax.text(
        0.02,
        0.90,
        f"complete | {_duration_label(regions)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color="#174D2B",
    )


def _annotate_incomplete(ax: plt.Axes, reasons: str) -> None:
    ax.text(
        0.02,
        0.90,
        f"incomplete: {reasons}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#8B0000",
        wrap=True,
    )


def _draw_r_anchor(ax: plt.Axes) -> None:
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.7)


def _protocol_columns(report: DelineationQCReport, *, selected_only: bool) -> tuple[str, ...]:
    selected = report.config.selected_prominence_candidate
    candidates = [selected] if selected_only else list(report.config.prominence_candidates)
    return _BASE_COLUMNS + tuple(
        f"Prominence: {candidate}{' (selected)' if candidate == selected else ''}"
        for candidate in candidates
    )


def _record_index(
    report: DelineationQCReport,
) -> dict[tuple[DelineationMethod, str, int], DelineationRecord]:
    return {
        (record.method, record.candidate, record.graph_idx): record for record in report.records
    }


def write_delineation_review_pdf(
    dataset_path: Path,
    report: DelineationQCReport,
    output_path: Path,
    *,
    overwrite: bool = False,
    selected_prominence_only: bool = False,
) -> int:
    """Render 12 leads with fixed, DWT, and prominence region overlays."""
    dataset_path = dataset_path.resolve()
    if report.dataset.path != str(dataset_path):
        raise ValueError("report dataset path does not match the overlay dataset")
    raw_source_path = resolve_source_path(
        report.dataset.dataset_source, report.dataset.source_variant
    ).resolve()
    if str(raw_source_path) != report.dataset.raw_source_path:
        raise ValueError("report raw-source path does not match the configured source")
    selected_candidate = report.config.selected_prominence_candidate
    prominence_candidates = (
        [selected_candidate]
        if selected_prominence_only
        else list(report.config.prominence_candidates)
    )
    protocol_columns = _protocol_columns(report, selected_only=selected_prominence_only)
    records = _record_index(report)
    pages = 0

    with (
        h5py.File(dataset_path, "r") as h5,
        staged_artifact(output_path, overwrite=overwrite) as staged,
    ):
        sampling_rate = int(h5.attrs["sampling_rate"])
        graph_exam = np.asarray(h5["graph_exam_index"][:], dtype=np.int64)
        graph_lead = np.asarray(h5["graph_lead"][:], dtype=np.int64)
        graph_beat = np.asarray(h5["graph_beat_index"][:], dtype=np.int64)
        graph_start = np.asarray(h5["graph_start_sample"][:], dtype=np.int64)
        graph_end = np.asarray(h5["graph_end_sample"][:], dtype=np.int64)
        graph_rpeak = np.asarray(h5["graph_r_peak_sample"][:], dtype=np.int64)

        with PdfPages(staged) as pdf:
            for sample in report.review_sample:
                exam_signal = load_preprocessed_exam_signal(h5, sample.exam_idx)
                for beat_index in sample.beat_indices:
                    fig, axes = plt.subplots(
                        len(TWELVE_LEAD_DISPLAY_NAMES),
                        len(protocol_columns),
                        figsize=(19, 22),
                        sharex=False,
                        sharey="row",
                        constrained_layout=True,
                    )
                    fig.suptitle(
                        f"{sample.class_name} | exam {sample.exam_id} "
                        f"(index {sample.exam_idx}) | stored beat {beat_index}",
                        fontsize=14,
                    )
                    for column, title in enumerate(protocol_columns):
                        axes[0, column].set_title(title, fontsize=10)

                    exam_mask = graph_exam == sample.exam_idx
                    for lead, lead_name in enumerate(TWELVE_LEAD_DISPLAY_NAMES):
                        graph_matches = np.flatnonzero(
                            exam_mask & (graph_lead == lead) & (graph_beat == beat_index)
                        )
                        for column in range(len(protocol_columns)):
                            ax = axes[lead, column]
                            _draw_r_anchor(ax)
                            ax.grid(axis="x", alpha=0.12)
                            ax.tick_params(labelsize=6)
                        axes[lead, 0].set_ylabel(lead_name, rotation=0, labelpad=16, fontsize=8)

                        if graph_matches.size != 1:
                            for column in range(len(protocol_columns)):
                                axes[lead, column].text(
                                    0.5,
                                    0.5,
                                    "stored beat unavailable",
                                    transform=axes[lead, column].transAxes,
                                    ha="center",
                                    va="center",
                                    fontsize=7,
                                )
                            continue

                        graph_idx = int(graph_matches[0])
                        start = int(graph_start[graph_idx])
                        end = int(graph_end[graph_idx])
                        r_peak = int(graph_rpeak[graph_idx])
                        samples = np.arange(start, end + 1)
                        x = (samples - r_peak) / sampling_rate
                        y = exam_signal.signal[lead, start : end + 1]
                        if y.shape[0] != x.shape[0]:
                            raise ValueError(
                                f"exam {sample.exam_idx} graph {graph_idx}: waveform slice "
                                "does not match stored beat bounds"
                            )
                        for column in range(len(protocol_columns)):
                            axes[lead, column].plot(x, y, color="#111111", linewidth=0.7)
                            axes[lead, column].set_xlim(float(x[0]), float(x[-1]))

                        fixed_legacy = _fixed_regions(
                            r_peak,
                            sampling_rate=sampling_rate,
                            class_name=sample.class_name,
                            protocol=FixedRegionProtocol.LEGACY,
                        )
                        fixed_standardized = _fixed_regions(
                            r_peak,
                            sampling_rate=sampling_rate,
                            class_name=sample.class_name,
                            protocol=FixedRegionProtocol.STANDARDIZED,
                        )
                        _draw_regions(axes[lead, 0], fixed_legacy)
                        _annotate_complete(axes[lead, 0], fixed_legacy)
                        _draw_regions(axes[lead, 1], fixed_standardized)
                        _annotate_complete(axes[lead, 1], fixed_standardized)

                        delineated_columns = [
                            (len(_BASE_COLUMNS) - 1, DelineationMethod.DWT, "dwt"),
                            *[
                                (column, DelineationMethod.PROMINENCE, candidate)
                                for column, candidate in enumerate(
                                    prominence_candidates, start=len(_BASE_COLUMNS)
                                )
                            ],
                        ]
                        for column, method, candidate in delineated_columns:
                            record = records[(method, candidate, graph_idx)]
                            delineated = _delineated_regions(record)
                            if delineated is None:
                                reasons = ", ".join(reason.value for reason in record.reasons)
                                _annotate_incomplete(axes[lead, column], reasons)
                            else:
                                regions = _seconds(delineated, sampling_rate)
                                _draw_regions(axes[lead, column], regions)
                                _annotate_complete(axes[lead, column], regions)

                    legend = [
                        Patch(facecolor=color, alpha=0.25, label=name.upper())
                        for name, color in REGION_COLORS.items()
                    ]
                    legend.append(
                        Line2D(
                            [0],
                            [0],
                            color="#555555",
                            linestyle="--",
                            linewidth=0.7,
                            label="Stored R anchor (t=0)",
                        )
                    )
                    fig.legend(handles=legend, loc="upper right", ncol=3, fontsize=8)
                    pdf.savefig(fig, dpi=150)
                    plt.close(fig)
                    pages += 1
    return pages
