"""Tests for training.checkpoint, self-describing checkpoint payloads."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import torch

from config.train_models import SubgraphRepresentationConfig
from ecg_checkpoint_io import CheckpointFormatError
from models.hier_model import HierGNNModel
from tests.unit.fixtures import MinimalModel
from training.checkpoint import (
    load_checkpoint,
    load_checkpoint_with_sha256,
    reconstruct_model_from_ckpt,
    save_checkpoint,
)
from training.hdf5_meta import DatasetFingerprint

pytestmark = [pytest.mark.unit]

_FINGERPRINT = DatasetFingerprint(
    num_classes=3,
    num_leads=2,
    class_names=["A", "B", "C"],
    sampling_rate=500,
    node_feature_names=["amplitude", "t_norm_beat"],
    edge_attr_names=["delta_t", "delta_y"],
    amplitude_normalization=None,
    node_reduction={
        "method": "uniform",
        "target_nodes": 30,
    },
)


def _tiny_model() -> HierGNNModel:
    return HierGNNModel(
        num_classes=3,
        num_leads=2,
        in_channels=2,
        edge_dim=2,
        hidden_channels=16,
        beat_encoder_layers=2,
        beat_encoder_mlp_hidden=24,
        dropout=0.2,
        gatv2_heads=2,
        gatv2_layers=1,
    )


def _save(
    tmp_path,
    model,
    *,
    epoch: int = 7,
    symmetrize_edges: bool = False,
    subgraph_representation=None,
):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        epoch=epoch,
        metric_dict={"f1/macro": 0.5},
        class_names=["A", "B", "C"],
        dataset_fingerprint=_FINGERPRINT,
        symmetrize_edges=symmetrize_edges,
        subgraph_representation=(
            subgraph_representation
            if subgraph_representation is not None
            else SubgraphRepresentationConfig()
        ),
    )
    return path


def test_path_replacement_between_hash_and_load_uses_same_open_inode(tmp_path, monkeypatch):
    import ecg_checkpoint_io as checkpoint_io

    path = tmp_path / "snapshot.pt"
    replacement = tmp_path / "replacement.pt"
    torch.save(
        {
            "symmetrize_edges": False,
            "subgraph_representation": {
                "adjacency": "stored_visibility",
                "edge_attributes": "derived",
            },
            "nonce": 7,
        },
        path,
    )
    torch.save(
        {
            "symmetrize_edges": False,
            "subgraph_representation": {
                "adjacency": "temporal_chain",
                "edge_attributes": "zero",
            },
            "nonce": 99,
        },
        replacement,
    )
    expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    real_hash = checkpoint_io.sha256_stream

    def replace_path_after_hash(stream):
        digest = real_hash(stream)
        os.replace(replacement, path)
        return digest

    monkeypatch.setattr(checkpoint_io, "sha256_stream", replace_path_after_hash)
    loaded = load_checkpoint_with_sha256(path)
    monkeypatch.setattr(checkpoint_io, "sha256_stream", real_hash)

    assert loaded.sha256 == expected_digest
    assert loaded.state["nonce"] == 7
    assert load_checkpoint_with_sha256(path).state["nonce"] == 99


def test_reconstruct_round_trips_architecture_and_weights(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model()
    path = _save(tmp_path, model)

    rebuilt, state = reconstruct_model_from_ckpt(path)

    assert rebuilt.hparams == model.hparams
    original_sd = model.state_dict()
    rebuilt_sd = rebuilt.state_dict()
    assert original_sd.keys() == rebuilt_sd.keys()
    for key, tensor in original_sd.items():
        assert torch.equal(tensor, rebuilt_sd[key]), key
    assert state["epoch"] == 7
    assert state["class_names"] == ["A", "B", "C"]
    assert state["dataset_fingerprint"] == {
        "num_classes": 3,
        "num_leads": 2,
        "class_names": ["A", "B", "C"],
        "sampling_rate": 500,
        "node_feature_names": ["amplitude", "t_norm_beat"],
        "edge_attr_names": ["delta_t", "delta_y"],
        "amplitude_normalization": None,
        "node_reduction": {
            "method": "uniform",
            "target_nodes": 30,
        },
    }
    assert state["symmetrize_edges"] is False


def test_save_checkpoint_persists_symmetrize_edges_true(tmp_path):
    model = _tiny_model()
    path = _save(tmp_path, model, symmetrize_edges=True)
    state = load_checkpoint(path)
    assert state["symmetrize_edges"] is True


@pytest.mark.parametrize("invalid", [0, 1, "false", None])
def test_checkpoint_rejects_non_boolean_symmetrize_edges(tmp_path, invalid):
    path = _save(tmp_path, _tiny_model())
    state = load_checkpoint(path)
    state["symmetrize_edges"] = invalid
    torch.save(state, path)

    with pytest.raises(ValueError, match="symmetrize_edges"):
        load_checkpoint(path)


@pytest.mark.parametrize("missing", ["symmetrize_edges", "subgraph_representation"])
def test_checkpoint_without_graph_policy_field_is_rejected(tmp_path, missing):
    path = _save(tmp_path, _tiny_model())
    state = load_checkpoint(path)
    del state[missing]
    torch.save(state, path)

    with pytest.raises(ValueError, match=missing):
        load_checkpoint(path)


def test_save_checkpoint_persists_nondefault_subgraph_representation(tmp_path):
    from config.train_models import SubgraphRepresentationConfig

    representation = SubgraphRepresentationConfig(
        adjacency="temporal_chain",
        edge_attributes="zero",
    )
    path = _save(tmp_path, _tiny_model(), subgraph_representation=representation)

    assert load_checkpoint(path)["subgraph_representation"] == {
        "adjacency": "temporal_chain",
        "edge_attributes": "zero",
    }


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="empty"),
        pytest.param(
            {"adjacency": "stored_visibility"},
            id="missing-edge-attributes",
        ),
        pytest.param(
            {"edge_attributes": "derived"},
            id="missing-adjacency",
        ),
        pytest.param(
            {
                "adjacency": "stored_visibility",
                "edge_attributes": "derived",
                "unexpected": True,
            },
            id="extra-key",
        ),
        pytest.param([], id="wrong-mapping-type"),
        pytest.param(
            {"adjacency": "knn", "edge_attributes": "derived"},
            id="invalid-adjacency",
        ),
        pytest.param(
            {"adjacency": "stored_visibility", "edge_attributes": "learned"},
            id="invalid-edge-attributes",
        ),
    ],
)
def test_checkpoint_representation_resolver_rejects_malformed_present_payload(
    payload,
):
    from training.checkpoint import resolve_subgraph_representation

    with pytest.raises(ValueError, match="subgraph_representation"):
        resolve_subgraph_representation({"subgraph_representation": payload})


def test_checkpoint_representation_resolver_accepts_complete_valid_payload():
    from training.checkpoint import resolve_subgraph_representation

    resolved = resolve_subgraph_representation(
        {
            "subgraph_representation": {
                "adjacency": "temporal_chain",
                "edge_attributes": "zero",
            }
        }
    )

    assert resolved.model_dump() == {
        "adjacency": "temporal_chain",
        "edge_attributes": "zero",
    }


def test_checkpoint_load_seam_rejects_incomplete_present_representation(tmp_path):
    path = tmp_path / "malformed.pt"
    torch.save({"symmetrize_edges": False, "subgraph_representation": {}}, path)

    with pytest.raises(ValueError, match="subgraph_representation"):
        load_checkpoint(path)


def test_reconstruct_rejects_malformed_representation_before_hparams(tmp_path):
    path = tmp_path / "malformed.pt"
    torch.save(
        {
            "symmetrize_edges": False,
            "subgraph_representation": {},
            "model_hparams": None,
        },
        path,
    )

    with pytest.raises(ValueError, match="subgraph_representation"):
        reconstruct_model_from_ckpt(path)


def test_reconstruct_rejects_checkpoint_without_hparams(tmp_path):
    path = _save(tmp_path, MinimalModel())
    assert load_checkpoint(path)["model_hparams"] is None
    with pytest.raises(ValueError, match="model_hparams"):
        reconstruct_model_from_ckpt(path)


def test_hparams_capture_matches_constructor_signature():
    import inspect

    params = [p for p in inspect.signature(HierGNNModel.__init__).parameters if p != "self"]
    model = _tiny_model()
    assert sorted(model.hparams.keys()) == sorted(params)


def test_save_checkpoint_from_compiled_model_has_clean_keys(tmp_path):
    model = _tiny_model()
    path = _save(tmp_path, torch.compile(model))
    state = load_checkpoint(path)
    assert all(not k.startswith("_orig_mod.") for k in state["model_state"])
    rebuilt, _ = reconstruct_model_from_ckpt(path)
    for key, value in rebuilt.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key])


def test_save_checkpoint_atomically_overwrites_with_new_epoch(tmp_path):
    model = _tiny_model()
    path = _save(tmp_path, model, epoch=7)
    assert load_checkpoint(path)["epoch"] == 7

    _save(tmp_path, model, epoch=8)

    assert load_checkpoint(path)["epoch"] == 8
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_checkpoint_serializer_failure_preserves_old_checkpoint(tmp_path, monkeypatch):
    model = _tiny_model()
    path = _save(tmp_path, model, epoch=7)

    def fail_after_partial_write(state, staged_path) -> None:
        staged_path.write_bytes(b"partial checkpoint")
        raise RuntimeError("serialization failed")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", fail_after_partial_write)
        with pytest.raises(RuntimeError, match="serialization failed"):
            _save(tmp_path, model, epoch=8)

    assert load_checkpoint(path)["epoch"] == 7
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_checkpoint_loader_rejection_preserves_old_checkpoint(tmp_path, monkeypatch):
    import training.checkpoint as checkpoint_module

    model = _tiny_model()
    path = _save(tmp_path, model, epoch=7)

    def reject_staged_checkpoint(staged_path):
        assert staged_path != path
        assert staged_path.parent == path.parent
        raise ValueError("staged checkpoint rejected")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_module, "load_checkpoint", reject_staged_checkpoint)
        with pytest.raises(ValueError, match="staged checkpoint rejected"):
            _save(tmp_path, model, epoch=8)

    assert load_checkpoint(path)["epoch"] == 7
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_save_checkpoint_canonicalizes_cuda_state_to_cpu(tmp_path):
    model = _tiny_model().cuda()

    path = tmp_path / "cuda-ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        epoch=1,
        metric_dict={"f1/macro": 0.5},
        class_names=["A", "B", "C"],
        dataset_fingerprint=_FINGERPRINT,
        symmetrize_edges=False,
        subgraph_representation=SubgraphRepresentationConfig(),
    )

    state = load_checkpoint(path)
    for tensor in state["model_state"].values():
        assert tensor.device.type == "cpu"


def test_shared_loader_rejects_numpy_payload_without_fallback(tmp_path):
    path = tmp_path / "numpy.pt"
    torch.save(
        {
            "model_state": {"weight": torch.zeros(2)},
            "thresholds": np.array([0.1, 0.2, 0.3]),
            "epoch": 1,
        },
        path,
    )
    with pytest.raises(CheckpointFormatError, match="unsafe fallback"):
        load_checkpoint(path)


def test_save_checkpoint_rejects_custom_object_before_write(tmp_path):
    class Custom:
        pass

    model = _tiny_model()
    with pytest.raises(CheckpointFormatError, match="unsupported checkpoint object"):
        save_checkpoint(
            tmp_path / "ckpt.pt",
            model=model,
            epoch=1,
            metric_dict={"f1/macro": 0.5, "extra": Custom()},
            class_names=["A", "B", "C"],
            dataset_fingerprint=_FINGERPRINT,
            symmetrize_edges=False,
            subgraph_representation=SubgraphRepresentationConfig(),
        )
    assert list(tmp_path.iterdir()) == []


def test_save_checkpoint_rejects_numpy_scalar_before_write(tmp_path):
    model = _tiny_model()
    with pytest.raises(CheckpointFormatError, match="unsupported checkpoint object"):
        save_checkpoint(
            tmp_path / "ckpt.pt",
            model=model,
            epoch=1,
            metric_dict={"f1/macro": np.float64(0.5)},
            class_names=["A", "B", "C"],
            dataset_fingerprint=_FINGERPRINT,
            symmetrize_edges=False,
            subgraph_representation=SubgraphRepresentationConfig(),
        )
    assert list(tmp_path.iterdir()) == []
