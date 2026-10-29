"""Model-output-blind delineation QC over every label-defined control exam."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

import explain.delineation_workflow as workflow
from config.models import DatasetConfig
from dataset.hdf5_writer import DatasetWriter
from dataset.loaders import CODE_6CLASS_LABEL_NAMES
from dataset.raw_signal import ExamSignal
from dataset.records import ExamMetadata
from explain.delineation import (
    BeatDelineation,
    DelineationMethod,
    LeadDelineation,
    MorphologyLandmarks,
)
from explain.delineation_qc import DelineationQCConfig
from explain.delineation_workflow import (
    ALL_EXAM_COHORT_KEYS,
    build_delineation_qc_report,
    load_delineation_qc_report,
    write_delineation_qc_report,
)
from tests.fixtures.synthetic_graph_hdf5 import make_beat_graph, write_exam

pytestmark = pytest.mark.integration


def _config() -> DelineationQCConfig:
    return DelineationQCConfig.model_validate(
        {
            "rpeak_source": "graph_r_peak_sample",
            "signal_source": "preprocessed_graph_build_pipeline",
            "dwt_signal_orientation": "original",
            "prominence_negative_polarity_policy": "invert",
            "prominence_candidates": {
                "pathology_permissive": {
                    "max_qrs_interval": 250,
                    "max_pr_interval": 500,
                    "max_r_rise_time": 200,
                    "typical_st_segment": 150,
                    "max_p_basepoint_interval": 100,
                    "max_r_basepoint_interval": 250,
                    "max_t_basepoint_interval": 200,
                }
            },
            "selected_prominence_candidate": "pathology_permissive",
            "selection_rationale": "Preselected from model-output-blind technical QC.",
            "duration_flag_ranges_ms": {
                "pr": {"minimum": 60.0, "maximum": 600.0},
                "qrs": {"minimum": 30.0, "maximum": 300.0},
                "t": {"minimum": 40.0, "maximum": 500.0},
            },
            "common_support_exam_summary_threshold": 0.5,
            "review_sample": {"exams_per_class": 1, "beats_per_exam": 1},
        }
    )


def _dataset(path):
    config = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1],
            "node_features": ["amplitude", "t_norm_beat"],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": 8}},
        }
    )
    labels = np.zeros((7, len(CODE_6CLASS_LABEL_NAMES)), dtype=np.uint8)
    labels[0, 0] = 1
    labels[1, 1] = 1
    labels[2, 2] = 1
    labels[3, 3] = 1
    labels[4, 4] = 1
    labels[5, 5] = 1
    writer = DatasetWriter(
        path,
        config=config,
        label_taxonomy="codetest_6class",
        label_names=CODE_6CLASS_LABEL_NAMES,
        created_by_version="test",
    )
    for exam_idx, label_row in enumerate(labels):
        exam_id = f"E{exam_idx}"
        write_exam(
            writer,
            ExamMetadata(
                exam_id=exam_id,
                patient_id=f"P{exam_idx}",
                strat_fold=exam_idx + 1,
                labels=label_row,
                label_names=CODE_6CLASS_LABEL_NAMES,
                label_taxonomy="codetest_6class",
            ),
            [
                make_beat_graph(
                    exam_id=exam_id,
                    lead=lead,
                    beat_index=0,
                    sampling_rate=400.0,
                )
                for lead in (0, 1)
            ],
        )
    writer.finalize()
    with h5py.File(path, "r+") as h5:
        graph_exam = h5["graph_exam_index"][:]
        graph_lead = h5["graph_lead"][:]
        negative_polarity = (graph_exam == 6) & (graph_lead == 1)
        h5["graph_rpeak_polarity"][negative_polarity] = -1
    return path


def _patch_signal_and_delineator(tmp_path, monkeypatch):
    raw_path = tmp_path / "codetest_raw.hdf5"
    raw_path.touch()
    calls: list[tuple[DelineationMethod, int, dict | None]] = []

    def fake_signal(_h5, exam_idx: int) -> ExamSignal:
        return ExamSignal(
            exam_id=f"E{exam_idx}",
            signal=np.zeros((2, 1000), dtype=np.float32),
            sampling_rate=400,
            dataset_source="codetest",
            source_variant="raw_400hz",
        )

    def fake_delineate(
        _signal,
        *,
        r_peak_samples,
        sampling_rate,
        method,
        candidate,
        polarity,
        prominence_kwargs=None,
    ) -> LeadDelineation:
        calls.append((method, polarity, prominence_kwargs))
        beats = tuple(
            BeatDelineation(
                r_peak_sample=int(r_peak),
                landmarks=MorphologyLandmarks(
                    p_onset_sample=int(r_peak) - 100,
                    qrs_onset_sample=int(r_peak) - 50,
                    qrs_offset_sample=int(r_peak) + 50,
                    t_onset_sample=int(r_peak) + 100,
                    t_offset_sample=int(r_peak) + 200,
                ),
                reasons=(),
            )
            for r_peak in r_peak_samples
        )
        return LeadDelineation(
            method=method,
            candidate=candidate,
            sampling_rate=sampling_rate,
            polarity=polarity,
            polarity_corrected=method is DelineationMethod.PROMINENCE and polarity == -1,
            r_peak_samples=tuple(int(value) for value in r_peak_samples),
            beats=beats,
        )

    monkeypatch.setattr(workflow, "resolve_source_path", lambda *_args: raw_path)
    monkeypatch.setattr(workflow, "load_preprocessed_exam_signal", fake_signal)
    monkeypatch.setattr(workflow, "delineate_lead", fake_delineate)
    return calls


def test_all_exam_qc_round_trips_controls_support_and_hdf5_identity(tmp_path, monkeypatch):
    dataset_path = _dataset(tmp_path / "codetest.h5")
    calls = _patch_signal_and_delineator(tmp_path, monkeypatch)

    report = build_delineation_qc_report(
        dataset_path,
        _config(),
        exam_scope="all_exams",
    )
    output = tmp_path / "all-exam-qc.json"
    write_delineation_qc_report(output, report)
    loaded = load_delineation_qc_report(output)

    assert loaded.exam_scope == "all_exams"
    assert loaded.dataset.analyzed_exam_indices == list(range(7))
    assert loaded.dataset.morphology_positive_exam_indices == [0, 1, 2]
    assert loaded.dataset.label_defined_cohorts is not None
    assert loaded.dataset.label_defined_cohorts.positive_by_target == {
        "1dAVb": [0],
        "RBBB": [1],
        "LBBB": [2],
    }
    assert loaded.dataset.label_defined_cohorts.target_negative_by_target == {
        "1dAVb": [1, 2, 3, 4, 5, 6],
        "RBBB": [0, 2, 3, 4, 5, 6],
        "LBBB": [0, 1, 3, 4, 5, 6],
    }
    assert loaded.dataset.label_defined_cohorts.six_class_negative == [6]
    assert {record.exam_idx for record in loaded.records} == set(range(7))
    assert all(record.positive_classes == [] for record in loaded.records if record.exam_idx >= 3)

    selected_records = [
        record for record in loaded.records if record.candidate in {"dwt", "pathology_permissive"}
    ]
    assert len(selected_records) == 7 * 2 * 2
    assert loaded.common_support.overall.n_units == 14
    assert loaded.common_support.overall.n_common == 14
    assert list(loaded.cohort_method_summaries) == list(ALL_EXAM_COHORT_KEYS)
    assert list(loaded.cohort_common_support) == list(ALL_EXAM_COHORT_KEYS)
    assert loaded.cohort_common_support["six_class_negative"].n_units == 2
    assert loaded.cohort_common_support["six_class_negative"].n_common == 2
    assert loaded.technical_assessment.model_outputs_consulted is False
    assert loaded.technical_assessment.clinical_fiducial_ground_truth == "unavailable"
    assert any(method is DelineationMethod.DWT and polarity == -1 for method, polarity, _ in calls)
    assert any(
        method is DelineationMethod.PROMINENCE
        and polarity == -1
        and kwargs == _config().selected_prominence_kwargs.model_dump()
        for method, polarity, kwargs in calls
    )

    valid_payload = json.loads(output.read_text())
    corruptions = (
        lambda payload: payload.update(producer="wrong"),
        lambda payload: payload.update(unexpected=True),
        lambda payload: payload["cohort_common_support"]["six_class_negative"].update(
            n_common=0,
            common_rate=0.0,
        ),
    )
    for corrupt in corruptions:
        payload = json.loads(json.dumps(valid_payload))
        corrupt(payload)
        output.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="invalid region_delineation_qc report"):
            load_delineation_qc_report(output)
    output.write_text(json.dumps(valid_payload))

    with h5py.File(dataset_path, "r+") as h5:
        h5["graph_lead"][0] = 9
    with pytest.raises(ValueError, match="differs from HDF5"):
        load_delineation_qc_report(output)


def test_label_positive_qc_covers_rhythm_only_exams_and_rejects_other_cohorts(
    tmp_path, monkeypatch
):
    dataset_path = _dataset(tmp_path / "codetest.h5")
    _patch_signal_and_delineator(tmp_path, monkeypatch)

    report = build_delineation_qc_report(dataset_path, _config(), exam_scope="label_positive")
    output = tmp_path / "label-positive-qc.json"
    write_delineation_qc_report(output, report)
    loaded = load_delineation_qc_report(output)

    assert loaded.dataset.analyzed_exam_indices == [0, 1, 2, 3, 4, 5]
    assert loaded.dataset.morphology_positive_exam_indices == [0, 1, 2]
    assert loaded.dataset.label_defined_cohorts is None
    assert {record.exam_idx for record in loaded.records} == set(range(6))
    workflow.validate_delineation_report_against_hdf5(loaded)

    dropped_rhythm_exam = loaded.model_copy(
        update={
            "dataset": loaded.dataset.model_copy(update={"analyzed_exam_indices": [0, 1, 2, 3, 4]})
        }
    )
    with pytest.raises(ValueError, match="label-positive cohort differs"):
        workflow.validate_delineation_report_against_hdf5(dropped_rhythm_exam)
