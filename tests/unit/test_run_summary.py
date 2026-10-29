"""Unit tests for ``training.run_summary`` (pure functions, no IO)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.run_summary import (
    build_run_summary,
    format_run_summary,
    write_run_summary,
)

pytestmark = pytest.mark.unit

_NODE_REDUCTION = {"method": "none"}


def _preset(num_workers: int = 0):
    from config.train_models import TrainPreset

    return TrainPreset.model_validate(
        {
            "run_name": "demo",
            "dataset_path": "/data/x.h5",
            "test_dataset_path": "/data/test.h5",
            "results_dir": "results",
            "model": {
                "hidden_channels": 32,
                "in_channels": 2,
                "edge_dim": 2,
                "beat_encoder": {"layers": 2, "mlp_hidden": 64},
                "dropout": 0.1,
                "gatv2_heads": 4,
                "gatv2_layers": 1,
                "gatv2_max_edge_step": 1,
                "head_hidden_multiplier": 0.5,
                "head_dropout": 0.04,
            },
            "optim": {
                "epochs": 1,
                "batch_size": 4,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "early_stop_patience": 1,
                "scheduler_patience": 1,
                "scheduler_factor": 0.5,
                "eval_val_every_epochs": 1,
                "grad_clip_norm": 1.0,
                "log_every_steps": 100,
                "num_workers": num_workers,
                "seed": 42,
            },
            "ddp": {"launcher": "single", "gpus": 1},
            "threshold": {"value": 0.5},
            "sampling": {"n_neg": 1},
        }
    )


def _toy_dataset() -> dict:
    """10 exams, 3 classes, 4 graphs each. Simple deterministic fixture."""
    exam_labels = np.zeros((10, 3), dtype=np.uint8)
    exam_labels[:7, 0] = 1
    exam_labels[::2, 1] = 1
    graphs_per_exam = np.full(10, 4, dtype=np.int64)
    train_idx = np.arange(0, 8, dtype=np.int64)
    val_idx = np.array([8], dtype=np.int64)
    test_idx = np.array([9], dtype=np.int64)
    class_names = ["A", "B", "C"]
    return {
        "exam_labels": exam_labels,
        "graphs_per_exam": graphs_per_exam,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "class_names": class_names,
    }


def test_build_run_summary_basic_shape():
    fx = _toy_dataset()
    s = build_run_summary(
        preset=_preset(),
        dataset_path="/data/x.h5",
        sampling_rate=500,
        total_exams=10,
        total_graphs=40,
        num_leads=12,
        class_names=fx["class_names"],
        exam_labels=fx["exam_labels"],
        graphs_per_exam=fx["graphs_per_exam"],
        train_idx=fx["train_idx"],
        val_idx=fx["val_idx"],
        test_idx=fx["test_idx"],
        device="cpu",
        world_size=1,
        node_reduction=_NODE_REDUCTION,
        model_parameter_count=1234,
        trainable_parameter_count=1200,
    )
    assert s["dataset"] == {
        "path": "/data/x.h5",
        "exams": 10,
        "beat_graphs": 40,
        "sampling_rate_hz": 500,
        "leads": 12,
        "node_reduction": {"method": "none"},
    }
    assert s["splits"]["train"] == {"exams": 8, "graphs": 32}
    assert s["splits"]["val"] == {"exams": 1, "graphs": 4}
    assert s["splits"]["test"] == {"exams": 1, "graphs": 4}
    assert s["classes"] == {"count": 3, "names": ["A", "B", "C"]}
    assert s["runtime"]["world_size"] == 1
    assert s["runtime"]["seed"] == 42
    assert s["model"]["parameter_count"] == {"total": 1234, "trainable": 1200}


def test_build_run_summary_prevalence_counts_and_pos_weight():
    fx = _toy_dataset()
    s = build_run_summary(
        preset=_preset(),
        dataset_path="/data/x.h5",
        sampling_rate=500,
        total_exams=10,
        total_graphs=40,
        num_leads=12,
        class_names=fx["class_names"],
        exam_labels=fx["exam_labels"],
        graphs_per_exam=fx["graphs_per_exam"],
        train_idx=fx["train_idx"],
        val_idx=fx["val_idx"],
        test_idx=fx["test_idx"],
        device="cpu",
        world_size=1,
        node_reduction=_NODE_REDUCTION,
    )
    assert s["prevalence"]["A"]["train"]["count"] == 7
    assert s["prevalence"]["A"]["train"]["pct"] == pytest.approx(7 / 8 * 100, rel=1e-6)
    assert s["prevalence"]["A"]["pos_weight_train"] == pytest.approx(1 / 7, rel=1e-6)
    assert s["prevalence"]["B"]["train"]["count"] == 4
    assert s["prevalence"]["B"]["pos_weight_train"] == pytest.approx(1.0)
    assert s["prevalence"]["C"]["train"]["count"] == 0
    assert s["prevalence"]["C"]["pos_weight_train"] is None


def test_build_run_summary_class_count_mismatch_raises():
    fx = _toy_dataset()
    bad_labels = fx["exam_labels"][:, :2]
    with pytest.raises(ValueError, match=r"exam_labels.shape\[1\]"):
        build_run_summary(
            preset=_preset(),
            dataset_path="/data/x.h5",
            sampling_rate=500,
            total_exams=10,
            total_graphs=40,
            num_leads=12,
            class_names=fx["class_names"],
            exam_labels=bad_labels,
            graphs_per_exam=fx["graphs_per_exam"],
            train_idx=fx["train_idx"],
            val_idx=fx["val_idx"],
            test_idx=fx["test_idx"],
            device="cpu",
            world_size=1,
            node_reduction=_NODE_REDUCTION,
        )


def test_write_run_summary_strips_non_finite_floats(tmp_path):
    fx = _toy_dataset()
    s = build_run_summary(
        preset=_preset(),
        dataset_path="/data/x.h5",
        sampling_rate=500,
        total_exams=10,
        total_graphs=40,
        num_leads=12,
        class_names=fx["class_names"],
        exam_labels=fx["exam_labels"],
        graphs_per_exam=fx["graphs_per_exam"],
        train_idx=fx["train_idx"],
        val_idx=fx["val_idx"],
        test_idx=fx["test_idx"],
        device="cpu",
        world_size=1,
        node_reduction=_NODE_REDUCTION,
    )
    s["prevalence"]["A"]["pos_weight_train"] = float("inf")
    out = tmp_path / "run_summary.json"
    write_run_summary(s, out)
    payload = json.loads(out.read_text())
    assert payload["prevalence"]["A"]["pos_weight_train"] is None
    assert payload["prevalence"]["C"]["pos_weight_train"] is None


def test_run_summary_exposes_representation_and_effective_edges():
    from config.train_models import TrainPreset

    fx = _toy_dataset()
    payload = _preset().model_dump()
    payload["subgraph_representation"] = {
        "adjacency": "temporal_chain",
        "edge_attributes": "zero",
    }
    payload["symmetrize_edges"] = True
    preset = TrainPreset.model_validate(payload)
    edges_per_exam = np.arange(10, 20, dtype=np.int64)

    summary = build_run_summary(
        preset=preset,
        dataset_path="/data/x.h5",
        sampling_rate=500,
        total_exams=10,
        total_graphs=40,
        num_leads=12,
        class_names=fx["class_names"],
        exam_labels=fx["exam_labels"],
        graphs_per_exam=fx["graphs_per_exam"],
        train_idx=fx["train_idx"],
        val_idx=fx["val_idx"],
        test_idx=fx["test_idx"],
        device="cpu",
        world_size=1,
        node_reduction={
            "method": "turning_points",
            "cutoff_hz": 40.0,
            "max_gap_ms": 80.0,
        },
        edges_per_exam=edges_per_exam,
    )

    assert summary["subgraph_representation"] == {
        "adjacency": "temporal_chain",
        "edge_attributes": "zero",
        "symmetrize_edges": True,
    }
    assert summary["dataset"]["node_reduction"] == {
        "method": "turning_points",
        "cutoff_hz": 40.0,
        "max_gap_ms": 80.0,
    }
    assert summary["splits"]["train"]["subgraph_edges"] == int(edges_per_exam[:8].sum())
    assert summary["splits"]["val"]["subgraph_edges"] == int(edges_per_exam[[8]].sum())
    assert summary["splits"]["test"]["subgraph_edges"] == int(edges_per_exam[[9]].sum())
    banner = format_run_summary(summary)
    for token in (
        "temporal_chain",
        "zero",
        "symmetrize_edges=True",
        "turning_points",
        "subgraph-edges",
    ):
        assert token in banner


def test_build_run_summary_emits_sampling_block():
    """The sampling block reports train pool sizes and exact per-epoch counts."""
    fx = _toy_dataset()
    labels = fx["exam_labels"].copy()
    labels[[1, 3, 5, 7], 0] = 0

    preset = _preset()
    preset = preset.model_copy(update={"sampling": preset.sampling.model_copy(update={"n_neg": 2})})

    s = build_run_summary(
        preset=preset,
        dataset_path="/data/x.h5",
        sampling_rate=500,
        total_exams=10,
        total_graphs=40,
        num_leads=12,
        class_names=fx["class_names"],
        exam_labels=labels,
        graphs_per_exam=fx["graphs_per_exam"],
        train_idx=fx["train_idx"],
        val_idx=fx["val_idx"],
        test_idx=fx["test_idx"],
        device="cpu",
        world_size=1,
        node_reduction=_NODE_REDUCTION,
    )

    samp = s["sampling"]
    assert samp["pool"] == {"n_pos": 4, "n_neg": 4}
    assert samp["per_epoch"] == {"n_pos": 4, "n_neg": 2, "total": 6}


def test_normalize_class_name_handles_bytes():
    fx = _toy_dataset()
    s = build_run_summary(
        preset=_preset(),
        dataset_path="/data/x.h5",
        sampling_rate=500,
        total_exams=10,
        total_graphs=40,
        num_leads=12,
        class_names=[b"A", b"B", b"C"],
        exam_labels=fx["exam_labels"],
        graphs_per_exam=fx["graphs_per_exam"],
        train_idx=fx["train_idx"],
        val_idx=fx["val_idx"],
        test_idx=fx["test_idx"],
        device="cpu",
        world_size=1,
        node_reduction=_NODE_REDUCTION,
    )
    assert s["classes"]["names"] == ["A", "B", "C"]
