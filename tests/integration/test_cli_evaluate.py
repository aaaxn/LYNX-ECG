"""Integration tests for the ``evaluate`` CLI.

The tests build a minimal synthetic HDF5 dataset (2 leads, 5 classes, 6 exams
with varied label patterns) and a tiny reconstructable HierGNNModel checkpoint,
then invoke the CLI via its Python entry-point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cli.evaluate import main as evaluate_main
from config.models import DatasetConfig
from config.train_models import SubgraphRepresentationConfig
from dataset.hdf5_writer import DatasetWriter
from dataset.records import ExamMetadata, GraphRecord
from models.hier_model import HierGNNModel
from tests.fixtures.synthetic_graph_hdf5 import write_exam
from training.checkpoint import save_checkpoint
from training.hdf5_meta import DatasetFingerprint, read_hdf5_meta

pytestmark = pytest.mark.integration

_CLASS_NAMES = ("1dAVb", "RBBB", "LBBB", "SB", "AF")
_NUM_LEADS = 2
_NUM_CLASSES = 5
_IN_CHANNELS = 2
_EDGE_DIM = 2


def _build_valid_dataset(path: Path) -> Path:
    """Build a minimal 2-lead, 5-class HDF5 with 6 exams and varied labels.

    The label matrix is deliberately varied so every class has at least one
    positive and one negative example, enabling meaningful metric computation.

    Schema is consistent with ``test_cli_qa.py``; only ``labels`` differ so
    that all 5 classes have positive examples.
    """
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "ptbxl",
            "source_variant": "500hz",
            "sampling_rate": 500,
            "graph_leads": [0, 1],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": 20}},
        }
    )
    writer = DatasetWriter(
        path,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=_CLASS_NAMES,
        created_by_version="0.0.1",
    )
    label_variants = [
        np.array([1, 0, 0, 0, 0], dtype=np.uint8),
        np.array([0, 1, 0, 0, 0], dtype=np.uint8),
        np.array([0, 0, 1, 0, 0], dtype=np.uint8),
        np.array([0, 0, 0, 1, 0], dtype=np.uint8),
        np.array([0, 0, 0, 0, 1], dtype=np.uint8),
        np.array([1, 1, 0, 0, 0], dtype=np.uint8),
    ]
    for i, labels in enumerate(label_variants):
        exam_id = str(i + 1)
        graphs = []
        for lead in range(_NUM_LEADS):
            for b in range(3):
                start = 200 + b * 700
                end = start + 600
                rpeak = start + 300
                idx = np.linspace(start, end, 20).round().astype(np.int64)
                idx[0] = start
                idx[-1] = end
                idx[int(np.argmin(np.abs(idx - rpeak)))] = rpeak
                idx = np.unique(idx)
                node_t = ((idx - start) / 500.0).astype(np.float32)
                amplitude = np.linspace(-1, 1, idx.size).astype(np.float32)
                t_norm = node_t / max(float(node_t[-1]), 1e-12)
                node_x = np.column_stack([amplitude, t_norm]).astype(np.float32)
                edges = [(j, j + 1) for j in range(idx.size - 1)] + [(0, idx.size - 1)]
                ei_arr = np.asarray(edges, dtype=np.int64)
                ei = np.column_stack([ei_arr.min(1), ei_arr.max(1)]).T
                graphs.append(
                    GraphRecord(
                        exam_id=exam_id,
                        lead=lead,
                        beat_index=b,
                        boundary_type=1 if b == 0 else (2 if b == 2 else 0),
                        start_sample=start,
                        end_sample=end,
                        r_peak_sample=rpeak,
                        rpeak_polarity=1,
                        original_num_nodes=end - start + 1,
                        node_x=node_x,
                        node_t=node_t,
                        node_original_sample=idx,
                        edge_index=ei,
                    )
                )
        write_exam(
            writer,
            ExamMetadata(
                exam_id=exam_id,
                patient_id=f"p{exam_id}",
                strat_fold=1,
                labels=labels,
                label_names=_CLASS_NAMES,
                label_taxonomy="synthetic_test_labels",
            ),
            graphs,
        )
    writer.finalize()
    return path


def _build_checkpoint(
    tmp_path: Path,
    h5_path: Path,
    *,
    subgraph_representation=None,
) -> Path:
    """Save a tiny reconstructable checkpoint compatible with the test dataset.

    All constructor args are
    passed explicitly; any deviation from hparams captured at init would
    cause reconstruction to fail silently, explicit construction documents
    the exact architecture under test.
    """
    torch.manual_seed(0)
    model = HierGNNModel(
        num_classes=_NUM_CLASSES,
        num_leads=_NUM_LEADS,
        in_channels=_IN_CHANNELS,
        edge_dim=_EDGE_DIM,
        hidden_channels=16,
        beat_encoder_layers=2,
    )
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        epoch=3,
        metric_dict={"f1/macro": 0.42},
        class_names=list(_CLASS_NAMES),
        dataset_fingerprint=read_hdf5_meta(h5_path).fingerprint,
        symmetrize_edges=False,
        subgraph_representation=(
            subgraph_representation
            if subgraph_representation is not None
            else SubgraphRepresentationConfig()
        ),
    )
    return ckpt_path


def test_evaluate_cli_writes_json(tmp_path):
    """The CLI exits 0 and writes a valid JSON report."""
    h5 = _build_valid_dataset(tmp_path / "data.h5")
    ckpt = _build_checkpoint(tmp_path, h5)
    out = tmp_path / "metrics.json"

    code = evaluate_main(
        [
            "--ckpt",
            str(ckpt),
            "--dataset",
            str(h5),
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--output",
            str(out),
        ]
    )

    assert code == 0
    assert out.exists(), "JSON output file was not created"
    payload = json.loads(out.read_text())
    assert payload["threshold"] == 0.5
    assert "metrics" in payload

    m = payload["metrics"]
    assert "f1/macro" in m
    assert "brier/macro" in m
    assert "ece/macro" in m
    assert f"confusion/{_CLASS_NAMES[0]}/tp" in m


def test_evaluate_cli_reconstructs_checkpointed_representation(tmp_path, monkeypatch):
    from training.dataset import ExamGraphDataset as RealDataset

    h5 = _build_valid_dataset(tmp_path / "data.h5")
    representation = SubgraphRepresentationConfig(
        adjacency="temporal_chain", edge_attributes="zero"
    )
    ckpt = _build_checkpoint(tmp_path, h5, subgraph_representation=representation)
    observed = []

    class SpyDataset(RealDataset):
        def __init__(self, *args, subgraph_representation=None, **kwargs):
            observed.append(subgraph_representation)
            super().__init__(
                *args,
                subgraph_representation=subgraph_representation,
                **kwargs,
            )

    monkeypatch.setattr("cli.evaluate.ExamGraphDataset", SpyDataset)

    code = evaluate_main(
        [
            "--ckpt",
            str(ckpt),
            "--dataset",
            str(h5),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )

    assert code == 0
    assert observed == [representation]


def test_evaluate_cli_rejects_class_name_mismatch(tmp_path):
    """A checkpoint whose class_names differ from the dataset must return 2."""
    h5 = _build_valid_dataset(tmp_path / "data.h5")

    torch.manual_seed(1)
    model = HierGNNModel(
        num_classes=_NUM_CLASSES,
        num_leads=_NUM_LEADS,
        in_channels=_IN_CHANNELS,
        edge_dim=_EDGE_DIM,
        hidden_channels=16,
        beat_encoder_layers=2,
    )
    ckpt_wrong = tmp_path / "ckpt_wrong.pt"
    base_fp = read_hdf5_meta(h5).fingerprint
    wrong_fp = DatasetFingerprint(
        num_classes=base_fp.num_classes,
        num_leads=base_fp.num_leads,
        class_names=["AF", "SB", "LBBB", "RBBB", "1dAVb"],
        sampling_rate=base_fp.sampling_rate,
        node_feature_names=base_fp.node_feature_names,
        edge_attr_names=base_fp.edge_attr_names,
        amplitude_normalization=base_fp.amplitude_normalization,
        node_reduction=base_fp.node_reduction,
    )
    save_checkpoint(
        ckpt_wrong,
        model=model,
        epoch=1,
        metric_dict={},
        class_names=["AF", "SB", "LBBB", "RBBB", "1dAVb"],
        dataset_fingerprint=wrong_fp,
        symmetrize_edges=False,
        subgraph_representation=SubgraphRepresentationConfig(),
    )

    code = evaluate_main(
        [
            "--ckpt",
            str(ckpt_wrong),
            "--dataset",
            str(h5),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )

    assert code == 2


def test_evaluate_cli_rejects_fingerprint_mismatch(tmp_path):
    """A checkpoint whose fingerprint sampling_rate differs from the dataset must return 2."""
    h5 = _build_valid_dataset(tmp_path / "data.h5")

    torch.manual_seed(2)
    model = HierGNNModel(
        num_classes=_NUM_CLASSES,
        num_leads=_NUM_LEADS,
        in_channels=_IN_CHANNELS,
        edge_dim=_EDGE_DIM,
        hidden_channels=16,
        beat_encoder_layers=2,
    )
    ckpt_wrong = tmp_path / "ckpt_wrong_sr.pt"
    base_fp = read_hdf5_meta(h5).fingerprint
    wrong_fp = DatasetFingerprint(
        num_classes=base_fp.num_classes,
        num_leads=base_fp.num_leads,
        class_names=base_fp.class_names,
        sampling_rate=100,
        node_feature_names=base_fp.node_feature_names,
        edge_attr_names=base_fp.edge_attr_names,
        amplitude_normalization=base_fp.amplitude_normalization,
        node_reduction=base_fp.node_reduction,
    )
    save_checkpoint(
        ckpt_wrong,
        model=model,
        epoch=1,
        metric_dict={},
        class_names=list(_CLASS_NAMES),
        dataset_fingerprint=wrong_fp,
        symmetrize_edges=False,
        subgraph_representation=SubgraphRepresentationConfig(),
    )

    code = evaluate_main(
        [
            "--ckpt",
            str(ckpt_wrong),
            "--dataset",
            str(h5),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )

    assert code == 2


def test_evaluate_cli_fails_closed_on_missing_fingerprint(tmp_path, capsys):
    """A checkpoint without a dataset_fingerprint must fail closed (exit 2)."""
    h5 = _build_valid_dataset(tmp_path / "data.h5")
    ckpt = _build_checkpoint(tmp_path, h5)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    del state["dataset_fingerprint"]
    torch.save(state, ckpt)

    code = evaluate_main(
        [
            "--ckpt",
            str(ckpt),
            "--dataset",
            str(h5),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )

    assert code == 2
    assert "dataset_fingerprint" in capsys.readouterr().out


def test_evaluate_cli_missing_file_returns_2(tmp_path):
    """Pointing --ckpt at a nonexistent path must return 2."""
    h5 = _build_valid_dataset(tmp_path / "data.h5")
    ckpt_nonexistent = tmp_path / "does_not_exist.pt"

    code = evaluate_main(
        [
            "--ckpt",
            str(ckpt_nonexistent),
            "--dataset",
            str(h5),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )

    assert code == 2


def test_evaluate_cli_rejects_malformed_fingerprint(tmp_path, capsys):
    """A dataset_fingerprint payload missing a required field must return exit 2,
    not crash with TypeError."""
    h5 = _build_valid_dataset(tmp_path / "data.h5")
    ckpt = _build_checkpoint(tmp_path, h5)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    del state["dataset_fingerprint"]["sampling_rate"]
    torch.save(state, ckpt)

    code = evaluate_main(
        [
            "--ckpt",
            str(ckpt),
            "--dataset",
            str(h5),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )
    assert code == 2
    assert "malformed" in capsys.readouterr().out


def test_evaluate_cli_rejects_malformed_checkpoint_representation(tmp_path):
    h5 = _build_valid_dataset(tmp_path / "data.h5")
    ckpt = _build_checkpoint(tmp_path, h5)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state["subgraph_representation"] = {
        "adjacency": "knn",
        "edge_attributes": "zero",
    }
    torch.save(state, ckpt)

    code = evaluate_main(
        [
            "--ckpt",
            str(ckpt),
            "--dataset",
            str(h5),
            "--device",
            "cpu",
            "--num-workers",
            "0",
        ]
    )

    assert code == 2
