"""CLI argument parsing + preset override smoke tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cli.train import _apply_overrides, _parse_args
from config.train_models import SamplingConfig, load_train_preset
from training.loaders import build_loaders
from training.sampling import DynamicPosNegSampler

pytestmark = pytest.mark.integration


def _baseline_preset_dict() -> dict:
    return {
        "run_name": "test_run",
        "dataset_path": "/tmp/none.h5",
        "test_dataset_path": "/tmp/none_test.h5",
        "results_dir": "results",
        "subgraph_representation": {
            "adjacency": "temporal_chain",
            "edge_attributes": "zero",
        },
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
            "batch_size": 2,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "early_stop_patience": 1,
            "scheduler_patience": 1,
            "scheduler_factor": 0.5,
            "eval_val_every_epochs": 1,
            "grad_clip_norm": 1.0,
            "log_every_steps": 100,
            "num_workers": 0,
            "seed": 42,
        },
        "ddp": {"launcher": "single", "gpus": 1},
        "threshold": {"value": 0.5},
        "sampling": {"n_neg": 1},
    }


def test_cli_overrides_applied(tmp_path):
    p = tmp_path / "x.json"
    with p.open("w") as f:
        json.dump(_baseline_preset_dict(), f)
    args = _parse_args(
        [
            "--config",
            str(p),
            "--dataset-path",
            "/data/other.h5",
            "--launcher",
            "ddp",
            "--gpus",
            "2",
            "--seed",
            "99",
        ]
    )
    preset = load_train_preset(p)
    out = _apply_overrides(preset, args)
    assert out.dataset_path == "/data/other.h5"
    assert out.ddp.launcher == "ddp"
    assert out.ddp.gpus == 2
    assert out.optim.seed == 99
    assert out.subgraph_representation.model_dump() == {
        "adjacency": "temporal_chain",
        "edge_attributes": "zero",
    }


def test_cli_test_dataset_path_override_applied(tmp_path):
    p = tmp_path / "x.json"
    with p.open("w") as f:
        json.dump(_baseline_preset_dict(), f)
    args = _parse_args(
        [
            "--config",
            str(p),
            "--dataset-path",
            "/staged/train.h5",
            "--test-dataset-path",
            "/staged/test.h5",
            "--results-dir",
            "/staged/results",
        ]
    )
    out = _apply_overrides(load_train_preset(p), args)
    assert out.dataset_path == "/staged/train.h5"
    assert out.test_dataset_path == "/staged/test.h5"
    assert out.results_dir == "/staged/results"


def test_cli_path_overrides_preserve_preset_defaults(tmp_path):
    """Without path overrides, the corresponding preset values are preserved."""
    p = tmp_path / "x.json"
    with p.open("w") as f:
        json.dump(_baseline_preset_dict(), f)
    args = _parse_args(["--config", str(p)])
    out = _apply_overrides(load_train_preset(p), args)
    assert out.test_dataset_path == "/tmp/none_test.h5"
    assert out.results_dir == "results"


def test_cli_launcher_ddp_relaunches_under_torchrun(tmp_path, monkeypatch):
    """--launcher ddp without torchrun env should execvpe into torchrun with forwarded args."""
    p = tmp_path / "x.json"
    payload = _baseline_preset_dict()
    payload["ddp"]["launcher"] = "ddp"
    payload["ddp"]["gpus"] = 2
    payload["dataset_path"] = str(tmp_path / "nope.h5")
    with p.open("w") as f:
        json.dump(payload, f)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    captured: dict = {}

    def fake_execvpe(file, argv, env):
        captured["file"] = file
        captured["argv"] = list(argv)
        captured["cuda"] = env.get("CUDA_VISIBLE_DEVICES")
        raise SystemExit(0)

    monkeypatch.setattr("cli.train.os.execvpe", fake_execvpe)

    from cli.train import main

    with pytest.raises(SystemExit):
        main(["--config", str(p), "--launcher", "ddp", "--gpus", "2", "--ddp-devices", "0,1"])

    assert captured["file"] == "torchrun"
    assert captured["argv"][:2] == ["torchrun", "--nproc_per_node=2"]
    assert captured["argv"][2:4] == ["-m", "cli.train"]
    assert "--launcher" in captured["argv"]
    assert "ddp" in captured["argv"]
    assert captured["cuda"] == "0,1"


def test_cli_already_under_torchrun_does_not_relaunch(tmp_path, monkeypatch):
    """If RANK/WORLD_SIZE are already set, main proceeds past the relaunch check (no execvpe)."""
    p = tmp_path / "x.json"
    payload = _baseline_preset_dict()
    payload["ddp"]["launcher"] = "ddp"
    payload["dataset_path"] = str(tmp_path / "nope.h5")
    with p.open("w") as f:
        json.dump(payload, f)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("cli.train.ddp.init_distributed", lambda: True)

    called = {"execvpe": False}

    def fake_execvpe(*a, **kw):
        called["execvpe"] = True

    monkeypatch.setattr("cli.train.os.execvpe", fake_execvpe)

    from cli.train import main

    with pytest.raises((FileNotFoundError, OSError)):
        main(["--config", str(p), "--launcher", "ddp"])

    assert called["execvpe"] is False


def test_cli_world_size_matches_preset_passes_guard(tmp_path, monkeypatch):
    """Preexisting torchrun env with WORLD_SIZE == gpus proceeds past the guard."""
    p = tmp_path / "x.json"
    payload = _baseline_preset_dict()
    payload["ddp"]["launcher"] = "ddp"
    payload["ddp"]["gpus"] = 2
    payload["dataset_path"] = str(tmp_path / "nope.h5")
    with p.open("w") as f:
        json.dump(payload, f)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    reached = {"init_distributed": False}

    def spy_init_distributed():
        reached["init_distributed"] = True
        return True

    monkeypatch.setattr("cli.train.ddp.init_distributed", spy_init_distributed)

    from cli.train import main

    with pytest.raises((FileNotFoundError, OSError)):
        main(["--config", str(p), "--launcher", "ddp"])

    assert reached["init_distributed"] is True


def test_cli_world_size_mismatch_fails_before_loaders(tmp_path, monkeypatch):
    """Preexisting torchrun env with WORLD_SIZE != gpus fails before any setup."""
    p = tmp_path / "x.json"
    payload = _baseline_preset_dict()
    payload["ddp"]["launcher"] = "ddp"
    payload["ddp"]["gpus"] = 2
    payload["dataset_path"] = str(tmp_path / "nope.h5")
    with p.open("w") as f:
        json.dump(payload, f)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")

    reached = {"init_distributed": False}

    def spy_init_distributed():
        reached["init_distributed"] = True
        return True

    monkeypatch.setattr("cli.train.ddp.init_distributed", spy_init_distributed)

    from cli.train import main

    with pytest.raises(ValueError, match="world-size mismatch"):
        main(["--config", str(p), "--launcher", "ddp"])

    assert reached["init_distributed"] is False


def test_cli_ddp_devices_cardinality_mismatch_fails_before_exec(tmp_path, monkeypatch):
    """--ddp-devices with more ids than gpus is rejected before execvpe."""
    p = tmp_path / "x.json"
    payload = _baseline_preset_dict()
    payload["ddp"]["launcher"] = "ddp"
    payload["ddp"]["gpus"] = 2
    payload["dataset_path"] = str(tmp_path / "nope.h5")
    with p.open("w") as f:
        json.dump(payload, f)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    called = {"execvpe": False}

    def fake_execvpe(*a, **kw):
        called["execvpe"] = True

    monkeypatch.setattr("cli.train.os.execvpe", fake_execvpe)

    from cli.train import main

    with pytest.raises(ValueError, match="ddp-devices lists 3 device"):
        main(["--config", str(p), "--launcher", "ddp", "--gpus", "2", "--ddp-devices", "0,1,2"])

    assert called["execvpe"] is False


def test_cli_ddp_devices_duplicate_fails_before_exec(tmp_path, monkeypatch):
    """--ddp-devices with duplicate normalized ids is rejected before execvpe."""
    p = tmp_path / "x.json"
    payload = _baseline_preset_dict()
    payload["ddp"]["launcher"] = "ddp"
    payload["ddp"]["gpus"] = 2
    payload["dataset_path"] = str(tmp_path / "nope.h5")
    with p.open("w") as f:
        json.dump(payload, f)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    called = {"execvpe": False}

    def fake_execvpe(*a, **kw):
        called["execvpe"] = True

    monkeypatch.setattr("cli.train.os.execvpe", fake_execvpe)

    from cli.train import main

    with pytest.raises(ValueError, match="duplicate device ids"):
        main(["--config", str(p), "--launcher", "ddp", "--gpus", "2", "--ddp-devices", "0,0"])

    assert called["execvpe"] is False


def test_build_loaders_uses_distributed_eval_samplers(monkeypatch):
    """DDP eval loaders must shard val/test WITHOUT padding duplicates.

    Val/test use the non-padding EvalShardSampler so all-gathered eval
    predictions are duplicate-free.
    """
    created_loaders: list[dict] = []
    eval_samplers: list = []

    class DummyDataset:
        def __init__(
            self,
            path,
            indices,
            num_leads,
            *,
            arrays=None,
            symmetrize_edges=False,
            subgraph_representation=None,
        ):
            self.path = path
            self.indices = list(indices)
            self.num_leads = num_leads

        def __len__(self):
            return len(self.indices)

    class DummyEvalSampler:
        def __init__(self, dataset_len, num_replicas, rank):
            self.dataset_len = dataset_len
            self.num_replicas = num_replicas
            self.rank = rank
            eval_samplers.append(self)

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            created_loaders.append({"dataset": dataset, **kwargs})

    monkeypatch.setattr(
        "training.loaders.GraphIndexArrays",
        type("FakeGIA", (), {"from_hdf5": staticmethod(lambda _path: object())}),
    )
    monkeypatch.setattr("training.loaders.ExamGraphDataset", DummyDataset)
    monkeypatch.setattr("training.loaders.EvalShardSampler", DummyEvalSampler)
    monkeypatch.setattr("training.loaders.PyGDataLoader", DummyLoader)

    build_loaders(
        dataset_path="/tmp/fake.h5",
        num_leads=12,
        train_idx=[0, 1, 2],
        val_idx=[3, 4],
        test_idx=[5, 6],
        exam_labels=np.zeros((7, 6), dtype=np.float32),
        sampling=SamplingConfig(n_neg=2),
        batch_size=2,
        num_workers=0,
        distributed=True,
        seed=123,
        device=torch.device("cpu"),
    )

    assert len(eval_samplers) == 2
    assert [s.dataset_len for s in eval_samplers] == [2, 2]
    assert isinstance(created_loaders[0]["sampler"], DynamicPosNegSampler)
    assert created_loaders[1]["sampler"] is eval_samplers[0]
    assert created_loaders[2]["sampler"] is eval_samplers[1]
    assert "shuffle" not in created_loaders[1]
    assert "shuffle" not in created_loaders[2]


def test_build_loaders_uses_dynamic_pos_neg_sampler(monkeypatch):
    """The train loader gets a DynamicPosNegSampler.

    Pos/neg pools are derived from ``exam_labels[train_idx]``; val/test loaders
    remain on their standard shuffled=False path (no resampling at eval time).
    """
    created_loaders: list[dict] = []

    class DummyDataset:
        def __init__(
            self,
            path,
            indices,
            num_leads,
            *,
            arrays=None,
            symmetrize_edges=False,
            subgraph_representation=None,
        ):
            self.indices = list(indices)

    class DummyLoader:
        def __init__(self, dataset, **kwargs):
            created_loaders.append({"dataset": dataset, **kwargs})

    monkeypatch.setattr(
        "training.loaders.GraphIndexArrays",
        type("FakeGIA", (), {"from_hdf5": staticmethod(lambda _path: object())}),
    )
    monkeypatch.setattr("training.loaders.ExamGraphDataset", DummyDataset)
    monkeypatch.setattr("training.loaders.PyGDataLoader", DummyLoader)

    exam_labels = np.zeros((8, 6), dtype=np.float32)
    exam_labels[2, 0] = 1.0
    exam_labels[5, 1] = 1.0

    build_loaders(
        dataset_path="/tmp/fake.h5",
        num_leads=12,
        train_idx=np.array([2, 3, 5, 7], dtype=np.int64),
        val_idx=np.array([0, 1], dtype=np.int64),
        test_idx=np.array([4, 6], dtype=np.int64),
        exam_labels=exam_labels,
        sampling=SamplingConfig(n_neg=2),
        batch_size=2,
        num_workers=0,
        distributed=False,
        seed=42,
        device=torch.device("cpu"),
    )

    train_loader = created_loaders[0]
    sampler = train_loader["sampler"]
    assert isinstance(sampler, DynamicPosNegSampler)
    assert sampler.pos_positions.tolist() == [0, 2]
    assert sampler.neg_positions.tolist() == [1, 3]
    assert sampler.n_neg == 2
    assert created_loaders[1].get("shuffle") is False
    assert created_loaders[2].get("shuffle") is False
