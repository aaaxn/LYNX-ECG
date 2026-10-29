"""QC/report contracts for model-output-blind morphology delineation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from explain.delineation import DelineationFailureReason, DelineationMethod
from explain.delineation_qc import (
    DelineationQCConfig,
    DelineationRecord,
    build_common_support,
    build_method_summaries,
    select_review_sample,
)
from explain.delineation_workflow import (
    DelineationDatasetBinding,
    DelineationQCReport,
    _technical_assessment,
    _validate_delineation_qc_report,
)
from explain.ecg_taxonomy import ECGRegion

pytestmark = pytest.mark.unit


def _config_payload() -> dict:
    default = {
        "max_qrs_interval": 180,
        "max_pr_interval": 300,
        "max_r_rise_time": 120,
        "typical_st_segment": 150,
        "max_p_basepoint_interval": 100,
        "max_r_basepoint_interval": 100,
        "max_t_basepoint_interval": 200,
    }
    permissive = {
        **default,
        "max_qrs_interval": 250,
        "max_pr_interval": 500,
        "max_r_rise_time": 200,
        "max_r_basepoint_interval": 250,
    }
    return {
        "rpeak_source": "graph_r_peak_sample",
        "signal_source": "preprocessed_graph_build_pipeline",
        "dwt_signal_orientation": "original",
        "prominence_negative_polarity_policy": "invert",
        "prominence_candidates": {"default": default, "pathology_permissive": permissive},
        "selected_prominence_candidate": "pathology_permissive",
        "selection_rationale": "Selected from delineation-only coverage and duration QC.",
        "duration_flag_ranges_ms": {
            "pr": {"minimum": 60.0, "maximum": 600.0},
            "qrs": {"minimum": 30.0, "maximum": 300.0},
            "t": {"minimum": 40.0, "maximum": 500.0},
        },
        "common_support_exam_summary_threshold": 0.5,
        "review_sample": {"exams_per_class": 1, "beats_per_exam": 2},
    }


def _record(
    graph_idx: int,
    *,
    method: DelineationMethod,
    candidate: str,
    complete: bool,
    exam_idx: int = 0,
    classes: tuple[str, ...] = ("1dAVb",),
    lead: int = 0,
    beat_index: int = 0,
    polarity: int = 1,
    suspicious_duration_flags: tuple[str, ...] = (),
) -> DelineationRecord:
    if complete:
        fiducials = {
            "p_onset_sample": 10,
            "qrs_onset_sample": 20,
            "qrs_offset_sample": 30,
            "t_onset_sample": 40,
            "t_offset_sample": 60,
        }
        available_regions = [ECGRegion.PR, ECGRegion.QRS, ECGRegion.T]
        durations_ms = {"pr": 100.0, "qrs": 100.0, "t": 200.0}
    else:
        fiducials = {
            "p_onset_sample": None,
            "qrs_onset_sample": 20,
            "qrs_offset_sample": 30,
            "t_onset_sample": 40,
            "t_offset_sample": 60,
        }
        available_regions = [ECGRegion.QRS, ECGRegion.T]
        durations_ms = {"qrs": 100.0, "t": 200.0}
    return DelineationRecord(
        graph_idx=graph_idx,
        exam_idx=exam_idx,
        exam_id=f"E{exam_idx}",
        positive_classes=list(classes),
        lead=lead,
        beat_index=beat_index,
        start_sample=0,
        end_sample=99,
        boundary_type=0,
        r_peak_sample=25,
        polarity=polarity,
        method=method,
        candidate=candidate,
        signal_orientation=(
            "polarity_corrected"
            if method is DelineationMethod.PROMINENCE and polarity == -1
            else "original"
        ),
        polarity_corrected=method is DelineationMethod.PROMINENCE and polarity == -1,
        complete=complete,
        available_regions=available_regions,
        reasons=[] if complete else [DelineationFailureReason.MISSING_P_ONSET],
        fiducials=fiducials,
        durations_ms=durations_ms,
        suspicious_duration_flags=list(suspicious_duration_flags),
        warning_messages=[],
        error_detail=None,
    )


def test_delineation_qc_config_requires_selected_candidate():
    config = DelineationQCConfig.model_validate(_config_payload())

    assert config.selected_prominence_kwargs.max_pr_interval == 500

    missing = _config_payload()
    missing["selected_prominence_candidate"] = "not_a_candidate"
    with pytest.raises(ValidationError, match="selected_prominence_candidate"):
        DelineationQCConfig.model_validate(missing)


def test_delineation_record_allows_no_positive_morphology_and_rejects_duplicates():
    negative = _record(
        0,
        method=DelineationMethod.DWT,
        candidate="dwt",
        complete=True,
        classes=(),
    )
    assert negative.positive_classes == []

    duplicate = negative.model_dump()
    duplicate["positive_classes"] = ["RBBB", "RBBB"]
    with pytest.raises(ValidationError, match="duplicates"):
        DelineationRecord.model_validate(duplicate)

    duplicate_flags = negative.model_dump()
    duplicate_flags["suspicious_duration_flags"] = ["qrs", "qrs"]
    with pytest.raises(ValidationError, match=r"suspicious_duration_flags.*duplicates"):
        DelineationRecord.model_validate(duplicate_flags)


def test_delineation_record_rejects_unordered_complete_fiducials():
    payload = _record(
        0,
        method=DelineationMethod.DWT,
        candidate="dwt",
        complete=True,
    ).model_dump()
    payload["fiducials"]["qrs_offset_sample"] = payload["r_peak_sample"]

    with pytest.raises(ValidationError, match="available_regions must match the fiducials"):
        DelineationRecord.model_validate(payload)


def test_method_summaries_report_failures_durations_and_all_required_dimensions():
    records = [
        _record(
            0,
            method=DelineationMethod.DWT,
            candidate="dwt",
            complete=True,
            suspicious_duration_flags=("pr", "t"),
        ),
        _record(
            1,
            method=DelineationMethod.DWT,
            candidate="dwt",
            complete=False,
            lead=1,
            beat_index=1,
            polarity=-1,
        ),
    ]

    summary = build_method_summaries(records)[0]

    assert summary.overall.n_units == 2
    assert summary.overall.n_complete == 1
    assert summary.overall.completeness_rate == 0.5
    assert summary.overall.failure_reason_counts == {"missing_p_onset": 1}
    assert summary.overall.suspicious_duration_counts == {"pr": 1, "t": 1}
    assert summary.overall.n_suspicious_duration_units == 1
    assert summary.overall.durations_ms["qrs"].median == 100.0
    assert set(summary.by_class) == {"1dAVb"}
    assert set(summary.by_exam) == {"0"}
    assert set(summary.by_lead) == {"0", "1"}
    assert set(summary.by_beat_index) == {"0", "1"}
    assert set(summary.by_polarity) == {"-1", "1"}

    tampered = summary.overall.model_dump()
    tampered["n_suspicious_duration_units"] = 0
    with pytest.raises(ValidationError, match="largest per-duration count"):
        type(summary.overall).model_validate(tampered)


def test_common_support_is_exact_selected_method_intersection_without_exam_filtering():
    records = [
        _record(0, method=DelineationMethod.DWT, candidate="dwt", complete=True),
        _record(1, method=DelineationMethod.DWT, candidate="dwt", complete=True, beat_index=1),
        _record(0, method=DelineationMethod.PROMINENCE, candidate="default", complete=True),
        _record(
            1, method=DelineationMethod.PROMINENCE, candidate="default", complete=True, beat_index=1
        ),
        _record(
            0, method=DelineationMethod.PROMINENCE, candidate="pathology_permissive", complete=True
        ),
        _record(
            1,
            method=DelineationMethod.PROMINENCE,
            candidate="pathology_permissive",
            complete=False,
            beat_index=1,
        ),
    ]

    support = build_common_support(
        records,
        selected_prominence_candidate="pathology_permissive",
        exam_summary_threshold=0.5,
    )

    assert support.common_graph_indices == [0]
    assert [unit.common for unit in support.membership] == [True, False]
    assert support.overall.n_units == 2
    assert support.overall.n_common == 1
    assert support.overall.common_rate == 0.5
    assert support.exams_at_or_above_threshold == [0]
    assert len(support.membership) == 2


def test_report_contract_round_trips_and_rejects_inconsistent_sections(tmp_path):
    config = DelineationQCConfig.model_validate(_config_payload())
    records: list[DelineationRecord] = []
    classes_by_exam = {0: ("1dAVb",), 1: ("RBBB",), 2: ("LBBB",)}
    for method, candidate in (
        (DelineationMethod.DWT, "dwt"),
        (DelineationMethod.PROMINENCE, "default"),
        (DelineationMethod.PROMINENCE, "pathology_permissive"),
    ):
        for exam_idx, classes in classes_by_exam.items():
            for beat_index in (0, 1):
                graph_idx = exam_idx * 2 + beat_index
                records.append(
                    _record(
                        graph_idx,
                        method=method,
                        candidate=candidate,
                        complete=True,
                        exam_idx=exam_idx,
                        classes=classes,
                        beat_index=beat_index,
                    )
                )
    summaries = build_method_summaries(records)
    support = build_common_support(
        records,
        selected_prominence_candidate="pathology_permissive",
        exam_summary_threshold=0.5,
    )
    review_sample = select_review_sample(
        records,
        class_order=("1dAVb", "RBBB", "LBBB"),
        required_leads=(0,),
        exams_per_class=1,
        beats_per_exam=2,
    )
    report = DelineationQCReport(
        producer="region_delineation_qc",
        exam_scope="morphology_positive",
        dataset=DelineationDatasetBinding(
            path=str((tmp_path / "dataset.h5").resolve()),
            raw_source_path=str((tmp_path / "raw.h5").resolve()),
            fingerprint={
                "num_classes": 6,
                "num_leads": 12,
                "class_names": ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"],
                "sampling_rate": 100,
                "node_feature_names": ["amplitude", "t_norm_beat"],
                "edge_attr_names": ["delta_t", "delta_y"],
                "amplitude_normalization": "p95_lead_median",
                "node_reduction": {"method": "turning_points"},
            },
            dataset_source="codetest",
            source_variant="raw_400hz",
            sampling_rate=100,
            graph_leads=[0],
            total_exams=827,
            total_graphs=6,
            morphology_positive_exam_indices=[0, 1, 2],
            analyzed_exam_indices=[0, 1, 2],
            label_defined_cohorts=None,
        ),
        config=config,
        dependency_versions={"neurokit2": "0.2.13"},
        records=records,
        method_summaries=summaries,
        common_support=support,
        cohort_method_summaries={},
        cohort_common_support={},
        review_sample=review_sample,
        technical_assessment=_technical_assessment(summaries, support, config),
    )

    def rebuild(
        changed_records: list[DelineationRecord],
        *,
        changed_config: DelineationQCConfig = config,
    ) -> DelineationQCReport:
        changed_summaries = build_method_summaries(changed_records)
        changed_support = build_common_support(
            changed_records,
            selected_prominence_candidate=changed_config.selected_prominence_candidate,
            exam_summary_threshold=changed_config.common_support_exam_summary_threshold,
        )
        changed_review = select_review_sample(
            changed_records,
            class_order=("1dAVb", "RBBB", "LBBB"),
            required_leads=(0,),
            exams_per_class=1,
            beats_per_exam=2,
        )
        return report.model_copy(
            update={
                "config": changed_config,
                "records": changed_records,
                "method_summaries": changed_summaries,
                "common_support": changed_support,
                "review_sample": changed_review,
                "technical_assessment": _technical_assessment(
                    changed_summaries,
                    changed_support,
                    changed_config,
                ),
            }
        )

    loaded = _validate_delineation_qc_report(
        DelineationQCReport.model_validate_json(report.model_dump_json())
    )

    assert loaded == report
    without_default = [record for record in records if record.candidate != "default"]
    with pytest.raises(ValueError, match="method/candidate coverage"):
        _validate_delineation_qc_report(rebuild(without_default))

    wrong_duration_records = list(records)
    target_record = next(
        record
        for record in records
        if record.graph_idx == 0 and record.method is DelineationMethod.DWT
    )
    assert target_record.durations_ms is not None
    wrong_duration_records[records.index(target_record)] = target_record.model_copy(
        update={"durations_ms": {**target_record.durations_ms, "qrs": 101.0}}
    )
    with pytest.raises(ValueError, match="durations"):
        _validate_delineation_qc_report(rebuild(wrong_duration_records))

    changed_config_payload = config.model_dump()
    changed_config_payload["duration_flag_ranges_ms"]["qrs"]["maximum"] = 50.0
    changed_config = DelineationQCConfig.model_validate(changed_config_payload)
    with pytest.raises(ValueError, match="duration flags"):
        _validate_delineation_qc_report(rebuild(records, changed_config=changed_config))

    valid = json.loads(report.model_dump_json())
    assert (
        _validate_delineation_qc_report(DelineationQCReport.model_validate_json(json.dumps(valid)))
        == report
    )
    corruptions = {
        "producer": lambda payload: payload.update(producer="wrong_producer"),
        "method summaries": lambda payload: payload["method_summaries"].pop(),
        "common support": lambda payload: payload["common_support"].update(
            exams_at_or_above_threshold=[]
        ),
        "review sample": lambda payload: payload["review_sample"][0].update(exam_id="tampered"),
        "technical assessment": lambda payload: payload["technical_assessment"][
            "statements"
        ].append("tampered"),
    }
    for corrupt in corruptions.values():
        wrong = json.loads(json.dumps(valid))
        corrupt(wrong)
        with pytest.raises(ValueError):
            _validate_delineation_qc_report(
                DelineationQCReport.model_validate_json(json.dumps(wrong))
            )


def test_review_sample_is_deterministic_and_deduplicates_multilabel_exams():
    records: list[DelineationRecord] = []
    graph_idx = 0
    exam_classes = {
        0: ("1dAVb", "RBBB"),
        1: ("1dAVb",),
        2: ("RBBB",),
        3: ("LBBB",),
    }
    for exam_idx, classes in exam_classes.items():
        for beat_index in (0, 1, 2):
            for lead in (0, 1):
                records.append(
                    _record(
                        graph_idx,
                        method=DelineationMethod.DWT,
                        candidate="dwt",
                        complete=True,
                        exam_idx=exam_idx,
                        classes=classes,
                        lead=lead,
                        beat_index=beat_index,
                    )
                )
                graph_idx += 1

    selected = select_review_sample(
        records,
        class_order=("1dAVb", "RBBB", "LBBB"),
        required_leads=(0, 1),
        exams_per_class=1,
        beats_per_exam=2,
    )

    assert [(item.class_name, item.exam_idx) for item in selected] == [
        ("1dAVb", 0),
        ("RBBB", 2),
        ("LBBB", 3),
    ]
    assert all(item.beat_indices == [0, 2] for item in selected)
