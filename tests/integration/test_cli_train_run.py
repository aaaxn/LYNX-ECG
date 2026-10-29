"""Successful and terminal-path integration contracts for ``cli.train.main``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from config.train_models import SubgraphRepresentationConfig
from dataset.hdf5_writer import DatasetWriter
from tests.fixtures.synthetic_graph_hdf5 import (
    make_beat_graph,
    make_exam_metadata,
    tiny_dataset_config,
    write_exam,
)
from training.checkpoint import reconstruct_model_from_ckpt
from training.hdf5_meta import DatasetFingerprint, assert_test_meta_compatible, read_hdf5_meta
from training.splits import patient_hash_split

pytestmark = pytest.mark.integration

CLASS_NAMES = ("1dAVb", "RBBB")
_LABEL_TAXONOMY = "synthetic_binary_v1"
_TRAIN_LAYOUT = [
    ("E000", 0),
    ("E001", 1),
    ("E002", 0),
    ("E003", 1),
    ("E004", 0),
    ("E005", 1),
    ("E006", 0),
    ("E007", 1),
    ("E008", 0),
    ("E009", 1),
    ("E010", None),
    ("E011", None),
    ("E012", None),
    ("E013", None),
]
_TEST_LAYOUT = [
    ("T000", 0),
    ("T001", 1),
]
_SPLIT = {"strategy": "patient_hash", "val_fraction": 0.2, "seed": 3}
_EXPECTED_REPRESENTATION = SubgraphRepresentationConfig(
    adjacency="temporal_chain",
    edge_attributes="zero",
).model_dump()


def _write_cli_h5(out: Path, layout: list[tuple[str, int | None]]) -> Path:
    writer = DatasetWriter(
        out,
        config=tiny_dataset_config(),
        label_taxonomy=_LABEL_TAXONOMY,
        label_names=CLASS_NAMES,
        created_by_version="test",
    )
    for exam_id, positive_class in layout:
        graphs = [
            make_beat_graph(exam_id=exam_id, lead=lead, beat_index=beat)
            for lead in (0, 1)
            for beat in (0, 1, 2)
        ]
        metadata = make_exam_metadata(
            exam_id,
            positive_class,
            -1,
            label_names=CLASS_NAMES,
            label_taxonomy=_LABEL_TAXONOMY,
        )
        write_exam(writer, metadata, graphs)
    writer.finalize()
    return out


def _build_cli_h5(tmp_path: Path) -> Path:
    return _write_cli_h5(tmp_path / "cli_train.h5", _TRAIN_LAYOUT)


def _build_cli_test_h5(tmp_path: Path) -> Path:
    return _write_cli_h5(tmp_path / "cli_test.h5", _TEST_LAYOUT)


def _preset_dict(
    h5_path: Path,
    test_h5_path: Path,
    results_dir: Path,
    *,
    run_name: str = "cli_smoke",
    **optim_overrides,
) -> dict:
    optim = {
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "eval_val_every_epochs": 1,
        "log_every_steps": 1000,
        "early_stop_patience": 10,
        "num_workers": 0,
        "compile": False,
        "seed": 42,
    }
    optim.update(optim_overrides)
    return {
        "run_name": run_name,
        "dataset_path": str(h5_path),
        "test_dataset_path": str(test_h5_path),
        "results_dir": str(results_dir),
        "subgraph_representation": {
            "adjacency": "temporal_chain",
            "edge_attributes": "zero",
        },
        "model": {
            "hidden_channels": 16,
            "in_channels": 2,
            "edge_dim": 2,
            "beat_encoder": {"layers": 1, "mlp_hidden": 16},
            "dropout": 0.1,
            "gatv2_heads": 2,
            "gatv2_layers": 1,
            "gatv2_max_edge_step": 1,
            "head_hidden_multiplier": 0.5,
            "head_dropout": 0.04,
        },
        "optim": optim,
        "ddp": {"launcher": "single", "gpus": 1},
        "threshold": {"value": 0.5},
        "split": _SPLIT,
        "sampling": {"n_neg": 2},
    }


def _write_preset(tmp_path: Path, h5_path: Path, **kwargs) -> Path:
    path = tmp_path / "preset.json"
    test_h5_path = _build_cli_test_h5(tmp_path)
    path.write_text(json.dumps(_preset_dict(h5_path, test_h5_path, tmp_path / "results", **kwargs)))
    return path


def _assert_checkpoint_contract(path: Path, expected_fingerprint) -> None:
    reconstructed, state = reconstruct_model_from_ckpt(path)
    assert reconstructed.hparams["num_classes"] == len(CLASS_NAMES)
    assert reconstructed.hparams["num_leads"] == 2
    assert state["subgraph_representation"] == _EXPECTED_REPRESENTATION
    assert DatasetFingerprint(**state["dataset_fingerprint"]) == expected_fingerprint
    assert list(state["class_names"]) == list(CLASS_NAMES)
    assert state["symmetrize_edges"] is False


def test_cli_fixture_splits_are_nondegenerate(tmp_path: Path) -> None:
    meta = read_hdf5_meta(_build_cli_h5(tmp_path))
    test_meta = read_hdf5_meta(_build_cli_test_h5(tmp_path))
    assert meta.num_classes == 2
    assert meta.num_leads == 2
    assert list(meta.class_names) == list(CLASS_NAMES)
    assert_test_meta_compatible(meta, test_meta)
    assert meta.patient_ids is not None
    assert test_meta.patient_ids is not None
    assert set(meta.patient_ids).isdisjoint(test_meta.patient_ids)
    train_idx, val_idx = patient_hash_split(
        meta.patient_ids, _SPLIT["val_fraction"], _SPLIT["seed"]
    )
    assert train_idx.size == 12
    assert val_idx.size == 2
    assert int((meta.exam_labels[train_idx].sum(axis=1) == 0).sum()) == 4
    assert test_meta.total_exams == 2
    for labels in (meta.exam_labels[train_idx], meta.exam_labels[val_idx], test_meta.exam_labels):
        assert labels.sum(axis=0).min() >= 1


def test_cli_train_run_succeeds(tmp_path: Path) -> None:
    from cli.train import main

    h5_path = _build_cli_h5(tmp_path)
    meta = read_hdf5_meta(h5_path)
    preset_path = _write_preset(tmp_path, h5_path)

    assert main(["--config", str(preset_path), "--device", "cpu"]) == 0

    run_dir = tmp_path / "results" / "cli_smoke_s42"
    summary = json.loads((run_dir / "run_summary.json").read_text())
    assert summary["subgraph_representation"] == {
        **_EXPECTED_REPRESENTATION,
        "symmetrize_edges": False,
    }
    assert summary["classes"]["names"] == list(CLASS_NAMES)
    assert summary["splits"]["train"]["exams"] == 12
    assert summary["splits"]["val"]["exams"] == 2
    assert summary["splits"]["test"]["exams"] == 2
    assert summary["prevalence"]["1dAVb"]["test"]["count"] == 1
    assert summary["prevalence"]["RBBB"]["test"]["count"] == 1

    snapshot = json.loads((run_dir / "preset.json").read_text())
    assert snapshot["run_name"] == "cli_smoke"
    assert snapshot["subgraph_representation"] == _EXPECTED_REPRESENTATION

    history = json.loads((run_dir / "history.json").read_text())
    assert isinstance(history, list) and history
    assert all(isinstance(row, dict) for row in history)

    best_f1 = run_dir / "best_f1_val.pt"
    assert best_f1.exists()
    assert (run_dir / "best_pr_auc_val.pt").exists()

    test_report = json.loads((run_dir / "test_metrics.json").read_text())
    assert test_report["threshold"] == 0.5
    metrics = test_report["metrics"]
    for key in ("f1/macro", "pr_auc/macro", "auroc/macro"):
        assert math.isfinite(metrics[key])
        assert 0.0 <= metrics[key] <= 1.0
    assert math.isfinite(metrics["brier/macro"])
    assert metrics["brier/macro"] >= 0.0
    assert math.isfinite(metrics["ece/macro"])
    assert metrics["confusion/1dAVb/tp"] >= 0.0

    _assert_checkpoint_contract(best_f1, meta.fingerprint)


def test_cli_train_run_missing_checkpoint_returns_1(tmp_path: Path, monkeypatch, capsys) -> None:
    from cli.train import main

    h5_path = _build_cli_h5(tmp_path)
    preset_path = _write_preset(tmp_path, h5_path)

    monkeypatch.setattr("cli.train.train", lambda **_kwargs: SimpleNamespace(history=[]))

    assert main(["--config", str(preset_path), "--device", "cpu"]) == 1
    run_dir = tmp_path / "results" / "cli_smoke_s42"
    assert not (run_dir / "best_f1_val.pt").exists()
    assert not (run_dir / "test_metrics.json").exists()
    assert (run_dir / "history.json").exists()
    stderr = capsys.readouterr().err
    assert "not produced" in stderr
    assert "skipping final test evaluation" in stderr


def test_cli_train_seed_fixes_the_initial_weights(tmp_path: Path) -> None:
    from cli.train import main

    h5_path = _build_cli_h5(tmp_path)
    weights = []
    for run, outside_seed in (("a", 0), ("b", 1)):
        run_dir = tmp_path / run
        run_dir.mkdir()
        preset_path = _write_preset(run_dir, h5_path)
        torch.manual_seed(outside_seed)
        assert main(["--config", str(preset_path), "--device", "cpu"]) == 0
        ckpt = run_dir / "results" / "cli_smoke_s42" / "best_pr_auc_val.pt"
        weights.append(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state"])
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][key], weights[1][key]) for key in weights[0])


def test_cli_train_seed_flag_writes_one_run_directory_per_seed(tmp_path: Path, monkeypatch) -> None:
    from cli.train import main

    h5_path = _build_cli_h5(tmp_path)
    preset_path = _write_preset(tmp_path, h5_path)
    monkeypatch.setattr("cli.train.train", lambda **_kwargs: SimpleNamespace(history=[]))

    for seed in (7, 8):
        assert main(["--config", str(preset_path), "--device", "cpu", "--seed", str(seed)]) == 1

    for seed in (7, 8):
        snapshot = json.loads(
            (tmp_path / "results" / f"cli_smoke_s{seed}" / "preset.json").read_text()
        )
        assert snapshot["optim"]["seed"] == seed
    assert not (tmp_path / "results" / "cli_smoke_s42").exists()
