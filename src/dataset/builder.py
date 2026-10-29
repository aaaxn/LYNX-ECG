"""Per-exam pipeline orchestration.

``process_exam`` is the unit of parallel work: given one ``ExamSample``
and a ``DatasetConfig``, it runs the full per-lead pipeline (R-peak
detection → beat segmentation → QC → node reduction → visibility graph)
and returns an ``ExamResult``. Edge attributes are not computed at build
time: the dataset contract derives them at load time (see :mod:`dataset.edge_attrs`).

All recoverable failures (R-peak detection failures, degenerate beats,
leads with too few valid beats, exams with no valid leads) are returned
as structured ``SkipEvent`` entries; only logic errors / bugs raise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from config.models import DatasetConfig
from dataset.beats import segment_beats
from dataset.graph_ops import build_visibility_graph
from dataset.labels import validate_label_definition, validate_label_vector
from dataset.loaders import ExamSample, prepare_exam_sample
from dataset.normalization import DegenerateAmplitudeError
from dataset.qc import (
    SkipEvent,
    validate_beat_anchor,
    validate_beat_signal,
    validate_selected_output,
)
from dataset.records import (
    ExamMetadata,
    ExamResult,
    GraphRecord,
    LeadResult,
    LeadSummary,
    pack_graphs,
)
from dataset.reduction import Beat, BeatReducer, LeadRejected, build_reducer
from dataset.rpeaks import RPeakFailure, RPeakResult, detect_rpeaks


def _build_node_features(
    *,
    amplitude: np.ndarray,
    node_t: np.ndarray,
    beat_duration_sec: float,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Compose ``node_x`` from the configured feature names. ``amplitude`` lands in column 0.

    ``beat_duration_sec`` is the beat's full first-to-last-sample span in
    seconds. ``t_norm_beat`` divides each node time by it, NOT by the last
    *selected* node time: reducers such as turning_points (without
    ``max_gap_ms``) may omit the final sample, so normalising by ``node_t[-1]``
    would map a node only part-way through the beat to 1.0 and make the feature
    incomparable across beats.
    """
    n_nodes = int(node_t.size)
    node_x = np.empty((n_nodes, len(feature_names)), dtype=np.float32)
    for col_idx, name in enumerate(feature_names):
        if name == "amplitude":
            node_x[:, col_idx] = amplitude.astype(np.float32, copy=False)
        elif name == "t_norm_beat":
            denom = beat_duration_sec if beat_duration_sec > 1e-12 else 1.0
            node_x[:, col_idx] = node_t.astype(np.float32, copy=False) / np.float32(denom)
        else:  # pragma: no cover
            raise ValueError(f"Unknown node feature: {name!r}")
    return node_x


def _process_lead(
    *,
    exam_sample: ExamSample,
    lead_idx: int,
    config: DatasetConfig,
    reducer: BeatReducer,
) -> LeadResult:
    """Run the lead-isolated pipeline for one lead. Returns LeadResult (valid or not)."""
    exam_id = exam_sample.exam_id
    sampling_rate = int(exam_sample.sampling_rate)
    skips: list[SkipEvent] = []

    if lead_idx >= exam_sample.signal.shape[0] or lead_idx < 0:
        skips.append(
            SkipEvent(
                level="lead",
                reason="lead_index_out_of_bounds",
                exam_id=exam_id,
                lead=lead_idx,
                details={"signal_leads": int(exam_sample.signal.shape[0])},
            )
        )
        return LeadResult(lead=lead_idx, valid=False, skips=skips)

    lead_signal = exam_sample.signal[lead_idx].astype(np.float32, copy=False)

    prep = reducer.prepare_lead(lead_signal, sampling_rate)
    if isinstance(prep, LeadRejected):
        skips.append(
            SkipEvent(
                level="lead",
                reason=prep.reason,
                exam_id=exam_id,
                lead=lead_idx,
                details=dict(prep.details),
            )
        )
        return LeadResult(lead=lead_idx, valid=False, skips=skips)

    rpeak_res = detect_rpeaks(
        lead_signal,
        sampling_rate=sampling_rate,
        polarity_mode=config.rpeak_detection.polarity_mode,
        min_count=int(config.qc.rpeak_min_count),
        hr_range_bpm=tuple(config.qc.rpeak_hr_range_bpm),
    )

    if isinstance(rpeak_res, RPeakFailure):
        skips.append(
            SkipEvent(
                level="lead",
                reason=rpeak_res.reason,
                exam_id=exam_id,
                lead=lead_idx,
                details=dict(rpeak_res.details),
            )
        )
        return LeadResult(lead=lead_idx, valid=False, skips=skips)

    assert isinstance(rpeak_res, RPeakResult)

    spans = segment_beats(signal_length=int(lead_signal.size), peaks=rpeak_res.peaks)

    graphs: list[GraphRecord] = []
    feature_names = list(config.node_features)
    for beat_idx, span in enumerate(spans):
        seg = lead_signal[span.start_sample : span.end_sample + 1]
        duration = (span.end_sample - span.start_sample + 1) / float(sampling_rate)

        qc_reason = validate_beat_signal(
            seg,
            duration_seconds=duration,
            rr_range_seconds=tuple(config.qc.beat_rr_range_seconds),
            min_amplitude=float(config.qc.beat_min_amplitude),
            min_samples=int(config.qc.min_samples_per_beat),
            boundary_type=span.boundary_type,
            skip_duration_check_for_boundary=config.qc.skip_duration_check_for_boundary_beats,
        )
        if qc_reason is None:
            qc_reason = validate_beat_anchor(span.start_sample, span.r_peak_sample, span.end_sample)
        if qc_reason is not None:
            skips.append(
                SkipEvent(
                    level="beat",
                    reason=qc_reason,
                    exam_id=exam_id,
                    lead=lead_idx,
                    beat_index=beat_idx,
                    details={
                        "boundary_type": int(span.boundary_type),
                        "duration_sec": float(duration),
                        "start_sample": int(span.start_sample),
                        "end_sample": int(span.end_sample),
                        "r_peak_sample": int(span.r_peak_sample),
                    },
                )
            )
            continue

        filtered_seg = (
            prep.filtered[span.start_sample : span.end_sample + 1]
            if prep.filtered is not None
            else None
        )
        beat = Beat(
            seg=seg,
            r_peak_local_idx=int(span.r_peak_local_idx),
            filtered_seg=filtered_seg,
            max_gap_samples=prep.max_gap_samples,
        )

        try:
            selected_local = reducer.select(beat)
        except ValueError as exc:
            skips.append(
                SkipEvent(
                    level="beat",
                    reason="pip_invalid_output",
                    exam_id=exam_id,
                    lead=lead_idx,
                    beat_index=beat_idx,
                    details={"exception": repr(exc)},
                )
            )
            continue

        selection_reason = validate_selected_output(
            selected_local,
            n_beat_samples=int(seg.size),
            contract=reducer.qc_contract(beat),
        )
        if selection_reason is not None:
            skips.append(
                SkipEvent(
                    level="beat",
                    reason=selection_reason,
                    exam_id=exam_id,
                    lead=lead_idx,
                    beat_index=beat_idx,
                    details={
                        "method": config.node_reduction.method,
                        "n_beat_samples": int(seg.size),
                        "selected_size": int(selected_local.size),
                    },
                )
            )
            continue

        node_amplitude = reducer.node_amplitude(beat, selected_local)
        node_original_sample = selected_local.astype(np.int64, copy=False) + int(span.start_sample)
        node_t = (selected_local.astype(np.float64) / float(sampling_rate)).astype(np.float32)

        node_x = _build_node_features(
            amplitude=node_amplitude,
            node_t=node_t,
            beat_duration_sec=(int(span.end_sample) - int(span.start_sample))
            / float(sampling_rate),
            feature_names=feature_names,
        )

        edge_index = build_visibility_graph(node_amplitude)

        graphs.append(
            GraphRecord(
                exam_id=exam_id,
                lead=lead_idx,
                beat_index=beat_idx,
                boundary_type=int(span.boundary_type),
                start_sample=int(span.start_sample),
                end_sample=int(span.end_sample),
                r_peak_sample=int(span.r_peak_sample),
                rpeak_polarity=int(rpeak_res.polarity),
                original_num_nodes=int(seg.size),
                node_x=node_x,
                node_t=node_t,
                node_original_sample=node_original_sample,
                edge_index=edge_index,
            )
        )

    if len(graphs) >= int(config.qc.min_valid_beats_per_lead):
        return LeadResult(
            lead=lead_idx,
            valid=True,
            graphs=graphs,
            skips=skips,
            rpeak_info={
                "polarity": int(rpeak_res.polarity),
                "num_peaks": int(rpeak_res.peaks.size),
                **{k: v for k, v in rpeak_res.info.items() if k != "polarity"},
            },
        )

    skips.append(
        SkipEvent(
            level="lead",
            reason="too_few_valid_beats",
            exam_id=exam_id,
            lead=lead_idx,
            details={
                "valid_beats": len(graphs),
                "min_valid_beats_per_lead": int(config.qc.min_valid_beats_per_lead),
                "candidate_beats": len(spans),
            },
        )
    )
    return LeadResult(lead=lead_idx, valid=False, skips=skips)


def process_exam(exam_sample: ExamSample, config: DatasetConfig) -> ExamResult:
    """Process one exam through the full per-lead pipeline.

    Expects a **raw native-rate** sample: the config-driven source transforms
    (band-pass, resample, amplitude normalization) run here, as the first step,
    so they execute inside the parallel workers rather than in the single loader
    process. Label metadata and values are validated before graph work; a
    ``LabelContractError`` propagates instead of becoming an exam skip. A
    degenerate per-exam amplitude scale is captured as a counted
    ``amplitude_norm_zero_scale`` skip; any other transform failure is captured
    as an ``exam_unexpected_error`` skip like any other per-exam failure.
    """
    label_names, label_taxonomy = validate_label_definition(
        exam_sample.label_names,
        exam_sample.label_taxonomy,
        context=f"exam {exam_sample.exam_id}",
    )
    labels = validate_label_vector(
        exam_sample.labels,
        num_classes=len(label_names),
        context=f"exam {exam_sample.exam_id} labels",
    )
    exam_sample = replace(
        exam_sample,
        labels=labels,
        label_names=label_names,
        label_taxonomy=label_taxonomy,
    )

    try:
        exam_sample = prepare_exam_sample(
            exam_sample,
            signal_filter=config.signal_filter,
            resample_to_hz=config.resample_to_hz,
            amplitude_normalization=config.amplitude_normalization,
            graph_leads=config.graph_leads,
        )
        reducer = build_reducer(config.node_reduction)
        leads_results: list[LeadResult] = []
        for lead_idx in config.graph_leads:
            ld = _process_lead(
                exam_sample=exam_sample, lead_idx=int(lead_idx), config=config, reducer=reducer
            )
            leads_results.append(ld)

        lead_summaries = [
            LeadSummary(
                lead=ld.lead,
                valid=ld.valid,
                n_graphs=len(ld.graphs),
                skips=ld.skips,
                rpeak_info=ld.rpeak_info,
            )
            for ld in leads_results
        ]

        valid_leads = sum(1 for ld in leads_results if ld.valid)
        if valid_leads < int(config.qc.min_valid_leads_per_exam):
            return ExamResult(
                exam_id=exam_sample.exam_id,
                success=False,
                metadata=None,
                leads=lead_summaries,
                skips=[
                    SkipEvent(
                        level="exam",
                        reason="exam_no_valid_leads",
                        exam_id=exam_sample.exam_id,
                        details={
                            "valid_leads": valid_leads,
                            "min_valid_leads_per_exam": int(config.qc.min_valid_leads_per_exam),
                        },
                    )
                ],
            )

        metadata = ExamMetadata(
            exam_id=exam_sample.exam_id,
            patient_id=exam_sample.patient_id,
            strat_fold=int(exam_sample.strat_fold),
            labels=exam_sample.labels,
            label_names=exam_sample.label_names,
            label_taxonomy=exam_sample.label_taxonomy,
        )
        graphs = [g for ld in leads_results if ld.valid for g in ld.graphs]
        return ExamResult(
            exam_id=exam_sample.exam_id,
            success=True,
            metadata=metadata,
            leads=lead_summaries,
            packed=pack_graphs(graphs) if graphs else None,
            skips=[],
        )
    except DegenerateAmplitudeError as exc:
        return ExamResult(
            exam_id=exam_sample.exam_id,
            success=False,
            metadata=None,
            leads=[],
            skips=[
                SkipEvent(
                    level="exam",
                    reason="amplitude_norm_zero_scale",
                    exam_id=exam_sample.exam_id,
                    details={"message": str(exc)},
                )
            ],
        )
    except Exception as exc:  # pragma: no cover
        return ExamResult(
            exam_id=exam_sample.exam_id,
            success=False,
            metadata=None,
            leads=[],
            skips=[
                SkipEvent(
                    level="exam",
                    reason="exam_unexpected_error",
                    exam_id=exam_sample.exam_id,
                    details={"exception": repr(exc)},
                )
            ],
            error=repr(exc),
        )


def summarize_exam_result(result: ExamResult) -> dict[str, Any]:
    """Small struct used by the main process for progress logging."""
    return {
        "exam_id": result.exam_id,
        "success": bool(result.success),
        "n_graphs": sum(ld.n_graphs for ld in result.leads if ld.valid),
        "n_skips": len(result.all_skips()),
        "valid_leads": sum(1 for ld in result.leads if ld.valid),
    }
