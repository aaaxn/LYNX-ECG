from __future__ import annotations

import math
from pathlib import Path

import pytest

from explain.coalition import EIGHT_STATE_ORDER, EIGHT_STATE_REGIONS, FACTORIAL_TERM_ORDER
from explain.coalition_workflow import CoalitionStateModel
from explain.common_support import PROMINENCE_SUPPORT_METHOD_ORDER, RegionDefinitionMethod
from explain.delineation_qc import ProminenceParameters
from explain.factorial_aggregation import build_factorial_method_result
from explain.prominence_factorial_workflow import (
    ProminenceFactorialClassResult,
    ProminenceFactorialExamResult,
    ProminenceFactorialSeedRun,
    ProminenceSupportFactorialReport,
    build_prominence_factorial_report,
    validate_prominence_factorial_report_integrity,
)
from explain.report import (
    ArtifactPathBinding,
    DatasetBinding,
    DatasetFingerprintModel,
    RepresentationBinding,
)

pytestmark = pytest.mark.unit


def _states(effects: dict[str, float]) -> list[CoalitionStateModel]:
    states: list[CoalitionStateModel] = []
    for state in EIGHT_STATE_ORDER:
        deleted = set() if state == "intact" else set(state.split("+"))
        signs = {region: (-1.0 if region in deleted else 1.0) for region in ("pr", "qrs", "t")}
        logit = sum(
            (effect / 2.0) * math.prod(signs[region] for region in term.split("*"))
            for term, effect in effects.items()
        )
        states.append(
            CoalitionStateModel(
                state=state,
                deleted_regions=list(EIGHT_STATE_REGIONS[state]),
                probability=1.0 / (1.0 + math.exp(-logit)),
                logit=logit,
                removed_nodes=len(deleted),
                removed_subgraphs=0,
                failure=None,
            )
        )
    return states


def _dataset(tmp_path: Path) -> DatasetBinding:
    return DatasetBinding(
        path=str((tmp_path / "dataset.h5").resolve()),
        fingerprint=DatasetFingerprintModel(
            num_classes=6,
            num_leads=12,
            class_names=["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"],
            sampling_rate=300,
            node_feature_names=["amplitude", "t_norm_beat"],
            edge_attr_names=["delta_t", "delta_y"],
            amplitude_normalization="p95_lead_median",
            node_reduction={"method": "none"},
        ),
    )


def _run(tmp_path: Path, seed: int) -> ProminenceFactorialSeedRun:
    exams: list[ProminenceFactorialExamResult] = []
    for exam_index in (0, 1):
        methods = []
        for method_index, method in enumerate(PROMINENCE_SUPPORT_METHOD_ORDER):
            pr = 3.0 + exam_index + seed / 10.0 + method_index
            effects = {term: 0.1 for term in FACTORIAL_TERM_ORDER}
            effects.update({"pr": pr, "qrs": pr + 2.0, "t": pr - 1.0})
            methods.append(build_factorial_method_result(method, _states(effects)))
        exams.append(
            ProminenceFactorialExamResult(
                exam_index=exam_index,
                exam_id=str(exam_index),
                graph_indices=[exam_index, exam_index + 10],
                prominence_graph_indices=[exam_index],
                prominence_support_rate=0.5,
                methods=methods,
            )
        )
    return ProminenceFactorialSeedRun(
        seed=seed,
        delineation_qc_report=ArtifactPathBinding(
            path=str((tmp_path / "delineation.json").resolve())
        ),
        delineation_qc_report_sha256="d" * 64,
        checkpoint_path=str((tmp_path / f"checkpoint-{seed}.pt").resolve()),
        checkpoint_sha256=f"{seed}" * 64,
        training_preset_path=str((tmp_path / f"preset-{seed}.json").resolve()),
        training_preset_sha256=f"{seed + 3}" * 64,
        dataset=_dataset(tmp_path),
        dataset_hdf5_sha256="a" * 64,
        representation=RepresentationBinding.model_validate(
            {
                "subgraph_representation": {
                    "adjacency": "stored_visibility",
                    "edge_attributes": "derived",
                },
                "symmetrize_edges": False,
            }
        ),
        selected_prominence_candidate="selected",
        selected_prominence_parameters=ProminenceParameters(
            max_qrs_interval=204,
            max_pr_interval=500,
            max_r_rise_time=200,
            typical_st_segment=164,
            max_p_basepoint_interval=100,
            max_r_basepoint_interval=250,
            max_t_basepoint_interval=200,
        ),
        candidate_selection_rationale="model-output-blind technical search",
        classes=[
            ProminenceFactorialClassResult(
                class_name="1dAVb",
                class_index=0,
                n_positive_exams=2,
                n_zero_prominence_support_exams=0,
                n_not_evaluated_exams=0,
                exams=exams,
            )
        ],
    )


def test_prominence_factorial_exam_rejects_any_incomplete_eight_state_game() -> None:
    complete_states = _states({term: 0.1 for term in FACTORIAL_TERM_ORDER})
    failed_state = complete_states[-1].model_copy(
        update={
            "probability": None,
            "logit": None,
            "failure": "non_finite_score",
        }
    )
    incomplete_states = [*complete_states[:-1], failed_state]
    methods = [
        build_factorial_method_result(
            method,
            incomplete_states if method is RegionDefinitionMethod.PROMINENCE else complete_states,
        )
        for method in PROMINENCE_SUPPORT_METHOD_ORDER
    ]

    with pytest.raises(ValueError, match="finite eight-state games"):
        ProminenceFactorialExamResult(
            exam_index=0,
            exam_id="0",
            graph_indices=[0],
            prominence_graph_indices=[0],
            prominence_support_rate=1.0,
            methods=methods,
        )


def test_prominence_factorial_report_rejects_tampered_segment_contrast(tmp_path: Path) -> None:
    report = build_prominence_factorial_report(
        [_run(tmp_path, seed) for seed in (1, 2, 3)],
        bootstrap_seed=0,
        bootstrap_resamples=100,
    )
    tampered = report.segment_main_effect_contrasts[0].model_copy(
        update={"mean_difference_a_minus_b": 999.0}
    )
    payload = report.model_copy(
        update={
            "segment_main_effect_contrasts": [
                tampered,
                *report.segment_main_effect_contrasts[1:],
            ]
        }
    ).model_dump()

    parsed = ProminenceSupportFactorialReport.model_validate(payload)
    with pytest.raises(ValueError, match="segment main-effect contrasts"):
        validate_prominence_factorial_report_integrity(parsed)
