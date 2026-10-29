"""Integration tests for ``process_exam`` (the per-exam pipeline)."""

from __future__ import annotations

import numpy as np
import pytest

from config.models import DatasetConfig
from dataset.builder import process_exam
from dataset.edge_attrs import compute_edge_attrs
from dataset.labels import LabelContractError
from dataset.loaders import ExamSample
from dataset.qc import SkipEvent
from dataset.records import ExamResult, GraphRecord
from dataset.schema import GRAPH_SCALAR_SPECS

pytestmark = pytest.mark.integration

_TP = {"method": "turning_points", "turning_points": {"cutoff_hz": 40.0, "max_gap_ms": 80.0}}


def _all_graphs(res: ExamResult) -> list[GraphRecord]:
    """Slice the exam's packed IPC payload back into per-graph records."""
    packed = res.packed
    if packed is None:
        return []
    node_ptr = np.concatenate([[0], np.cumsum(packed.node_counts)])
    edge_ptr = np.concatenate([[0], np.cumsum(packed.edge_counts)])
    graphs = []
    for i in range(packed.n_graphs):
        n0, n1 = int(node_ptr[i]), int(node_ptr[i + 1])
        e0, e1 = int(edge_ptr[i]), int(edge_ptr[i + 1])
        scalars = {
            spec.graph_field: int(packed.scalars[spec.name][i])
            for spec in GRAPH_SCALAR_SPECS
            if spec.graph_field != "reduced_num_nodes"
        }
        graphs.append(
            GraphRecord(
                exam_id=res.exam_id,
                node_x=packed.node_x[n0:n1],
                node_t=packed.node_t[n0:n1],
                node_original_sample=packed.node_original_sample[n0:n1],
                edge_index=packed.edge_index[:, e0:e1].astype(np.int64),
                **scalars,
            )
        )
    return graphs


def _graphs_by_lead(res: ExamResult) -> dict[int, list[GraphRecord]]:
    by_lead: dict[int, list[GraphRecord]] = {ld.lead: [] for ld in res.leads}
    for g in _all_graphs(res):
        by_lead[g.lead].append(g)
    return by_lead


def _cfg(**overrides) -> DatasetConfig:
    base = {
        "dataset_source": "ptbxl",
        "source_variant": "500hz",
        "sampling_rate": 500,
        "graph_leads": [0, 1, 2],
        "node_reduction": _TP,
    }
    base.update(overrides)
    return DatasetConfig.model_validate(base)


def test_process_exam_synthetic_clean(synthetic_ecg_clean):
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1, 2],
            "node_reduction": _TP,
        }
    )
    exam = ExamSample(
        exam_id="e0",
        patient_id="p_e0",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_clean] * 3, axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert isinstance(res, ExamResult)
    assert res.success
    assert res.metadata is not None
    assert len(res.leads) == 3
    assert all(ld.valid for ld in res.leads)
    by_lead = _graphs_by_lead(res)
    for ld in res.leads:
        assert ld.n_graphs >= 3
        assert len(by_lead[ld.lead]) == ld.n_graphs
        for g in by_lead[ld.lead]:
            assert isinstance(g, GraphRecord)
            assert g.edge_index.dtype == np.int64
            assert bool(np.all(g.edge_index[0] < g.edge_index[1]))
            nodes = g.node_original_sample.tolist()
            assert int(g.start_sample) in nodes
            assert int(g.end_sample) in nodes
            assert int(g.r_peak_sample) in nodes
            # max_gap_ms=80 at 400 Hz: no two consecutive nodes more than 32 samples apart.
            assert int(np.diff(g.node_original_sample).max()) <= 32


def test_process_exam_rejects_malformed_labels_before_graph_work(synthetic_ecg_clean):
    cfg = _cfg()
    exam = ExamSample(
        exam_id="bad-labels",
        patient_id="p",
        strat_fold=1,
        labels=np.array([0, 2, 0, 0, 0], dtype=np.int64),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        label_taxonomy="synthetic_test_labels",
        signal=np.stack([synthetic_ecg_clean] * 3, axis=0).astype(np.float32),
        sampling_rate=500,
    )

    with pytest.raises(LabelContractError, match=r"exam bad-labels.*binary"):
        process_exam(exam, cfg)


def test_process_exam_uniform_method_no_anchors(synthetic_ecg_clean):
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1, 2],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": 30}},
        }
    )
    exam = ExamSample(
        exam_id="u1",
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_clean] * 3, axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert res.success
    for graphs in _graphs_by_lead(res).values():
        for g in graphs:
            assert g.reduced_num_nodes == 30


def test_process_exam_flat_signal_no_valid_leads(synthetic_ecg_flat):
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1, 2],
            "node_reduction": _TP,
        }
    )
    exam = ExamSample(
        exam_id="flat",
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_flat] * 3, axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert not res.success
    all_skips = res.all_skips()
    assert any(s.level == "exam" and s.reason == "exam_no_valid_leads" for s in all_skips)
    assert sum(1 for s in all_skips if s.level == "lead") == 3


def test_process_exam_amplitude_norm_zero_scale_skip(synthetic_ecg_flat):
    """A flat exam with normalization on is skipped as ``amplitude_norm_zero_scale``
    BEFORE the per-lead loop, not as the generic ``exam_no_valid_leads``."""
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1, 2],
            "node_reduction": _TP,
            "amplitude_normalization": {"method": "p95_lead_median"},
        }
    )
    exam = ExamSample(
        exam_id="flat",
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_flat] * 3, axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert not res.success
    skips = res.all_skips()
    assert [s.reason for s in skips] == ["amplitude_norm_zero_scale"]
    assert skips[0].level == "exam"
    assert res.leads == []


def test_process_exam_normalization_is_gain_invariant(synthetic_ecg_clean):
    """The same exam at two gains, with normalization on, yields identical node
    amplitudes (column 0), the whole point of the feature."""
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1, 2],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": 30}},
            "amplitude_normalization": {"method": "p95_lead_median"},
        }
    )

    def _amplitudes(gain: float) -> np.ndarray:
        exam = ExamSample(
            exam_id="g",
            patient_id="p",
            strat_fold=1,
            labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
            label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
            label_taxonomy="codetest_6class",
            signal=(np.stack([synthetic_ecg_clean] * 3, axis=0) * gain).astype(np.float32),
            sampling_rate=400,
        )
        res = process_exam(exam, cfg)
        assert res.success
        return np.concatenate([g.node_x[:, 0] for g in _all_graphs(res)])

    np.testing.assert_allclose(_amplitudes(1.0), _amplitudes(9.7), rtol=1e-4, atol=1e-4)


def test_process_exam_lead_index_out_of_range(synthetic_ecg_clean):
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 99],
            "node_reduction": _TP,
        }
    )
    exam = ExamSample(
        exam_id="oob",
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_clean] * 3, axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert res.success
    leads = {ld.lead: ld for ld in res.leads}
    assert leads[0].valid
    assert not leads[99].valid
    assert any(s.reason == "lead_index_out_of_bounds" for s in leads[99].skips)
    assert all(s.reason != "rpeak_no_valid_polarity" for s in leads[99].skips)


def test_process_exam_inverted_signal_uses_polarity_aware(synthetic_ecg_inverted):
    """Even with inverted signal, polarity_mode='auto' should yield valid graphs."""
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1],
            "node_reduction": _TP,
        }
    )
    exam = ExamSample(
        exam_id="inv",
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_inverted] * 2, axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert res.success
    for graphs in _graphs_by_lead(res).values():
        for g in graphs:
            assert g.rpeak_polarity in (1, -1)


def test_process_exam_records_have_consistent_t_and_sample(synthetic_ecg_clean):
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0],
            "node_reduction": _TP,
        }
    )
    exam = ExamSample(
        exam_id="cf",
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_clean], axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert res.success
    for g in _all_graphs(res):
        expected = (g.node_original_sample - g.start_sample).astype(np.float64) / 400.0
        np.testing.assert_allclose(g.node_t.astype(np.float64), expected, atol=1e-5)
        assert float(g.node_t[0]) == 0.0
        assert int(g.edge_index.max()) < g.reduced_num_nodes
        if g.edge_index.shape[1] > 0:
            derived = compute_edge_attrs(g.edge_index, g.node_t, g.node_x, ["delta_t", "delta_y"])
            assert bool(np.all(derived[:, 0] > 0))


def test_skip_events_have_exam_lead_beat(synthetic_ecg_flat):
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1],
            "node_reduction": _TP,
        }
    )
    exam = ExamSample(
        exam_id="evflat",
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_flat] * 2, axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert not res.success
    for skip in res.all_skips():
        assert isinstance(skip, SkipEvent)
        assert skip.level in ("exam", "lead", "beat")
        assert skip.exam_id == "evflat" or skip.level == "exam"


def test_process_exam_turning_points(synthetic_ecg_clean):
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0, 1],
            "node_reduction": {
                "method": "turning_points",
                "turning_points": {"cutoff_hz": 40.0},
            },
        }
    )
    exam = ExamSample(
        exam_id="tp0",
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.stack([synthetic_ecg_clean] * 2, axis=0).astype(np.float32),
        sampling_rate=400,
    )
    res = process_exam(exam, cfg)
    assert isinstance(res, ExamResult)
    assert res.success
    by_lead = _graphs_by_lead(res)
    for ld in res.leads:
        assert ld.valid
        assert ld.n_graphs >= 3
        for g in by_lead[ld.lead]:
            assert isinstance(g, GraphRecord)
            assert g.node_x.shape[0] >= 2
            assert g.edge_index.dtype == np.int64
            if g.edge_index.shape[1] > 0:
                assert bool(np.all(g.edge_index[0] < g.edge_index[1]))
            assert int(g.r_peak_sample) in g.node_original_sample.tolist()


def _tp_exam(n_samples: int, *, exam_id: str) -> ExamSample:
    return ExamSample(
        exam_id=exam_id,
        patient_id="p",
        strat_fold=1,
        labels=np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"),
        label_taxonomy="codetest_6class",
        signal=np.zeros((1, n_samples), dtype=np.float32),
        sampling_rate=400,
    )


def test_process_exam_turning_points_short_lead_skips_cleanly():
    """Regression: a lead shorter than sosfiltfilt's padlen must skip, not crash.

    For a 4th-order Butterworth, scipy pads by edge=15 and needs len>15, so
    length 15 is too short and length 16 is the first valid length. The old
    guard (min_padlen=12) let 12-15 through, then sosfiltfilt raised ValueError,
    which the catch-all mislabelled. The fix must produce a clean lead-level
    'signal_too_short_for_filter' skip and never trip the defensive catch-all.
    """
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "codetest",
            "source_variant": "raw_400hz",
            "sampling_rate": 400,
            "graph_leads": [0],
            "node_reduction": {"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}},
        }
    )

    res15 = process_exam(_tp_exam(15, exam_id="tp15"), cfg)
    reasons15 = {s.reason for s in res15.all_skips()}
    assert "signal_too_short_for_filter" in reasons15
    assert "exam_unexpected_error" not in reasons15

    res16 = process_exam(_tp_exam(16, exam_id="tp16"), cfg)
    reasons16 = {s.reason for s in res16.all_skips()}
    assert "signal_too_short_for_filter" not in reasons16
    assert "exam_unexpected_error" not in reasons16
