"""Integration tests for training.inference's composed checkpoint-eval
helpers: :func:`predict_on_dataset` and :func:`load_checkpoint_and_predict`.

These exercise the real collaborators end-to-end: a real HierGNNModel
checkpoint scored against a real (tiny, synthetic) current-contract HDF5. That is
the seam
the module-level mocks in tests/unit/test_inference.py deliberately avoid.
Fixture pattern mirrors tests/integration/test_cli_evaluate.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from config.models import DatasetConfig
from config.train_models import SubgraphRepresentationConfig
from dataset.hdf5_writer import DatasetWriter
from dataset.records import ExamMetadata, GraphRecord
from models.hier_model import HierGNNModel
from tests.fixtures.synthetic_graph_hdf5 import write_exam
from training.checkpoint import save_checkpoint
from training.hdf5_meta import read_hdf5_meta
from training.inference import (
    CheckpointPredictions,
    load_checkpoint_and_dataset,
    load_checkpoint_and_predict,
)

pytestmark = pytest.mark.integration

_CLASS_NAMES = ("1dAVb", "RBBB", "LBBB")
_NUM_LEADS = 2
_NUM_CLASSES = 3
_IN_CHANNELS = 2
_EDGE_DIM = 2


def _build_dataset(path: Path, *, n_exams: int = 6) -> Path:
    """Minimal 2-lead, 3-class HDF5 with varied labels (every class has both
    a positive and a negative exam)."""
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "ptbxl",
            "source_variant": "500hz",
            "sampling_rate": 500,
            "graph_leads": [0, 1],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": 12}},
        }
    )
    writer = DatasetWriter(
        path,
        config=cfg,
        label_taxonomy="synthetic_v1",
        label_names=_CLASS_NAMES,
        created_by_version="test",
    )
    for i in range(n_exams):
        exam_id = str(i + 1)
        labels = np.zeros(_NUM_CLASSES, dtype=np.uint8)
        labels[i % _NUM_CLASSES] = 1
        graphs = []
        for lead in range(_NUM_LEADS):
            for b in range(2):
                start = 200 + b * 700
                end = start + 600
                rpeak = start + 300
                idx = np.linspace(start, end, 12).round().astype(np.int64)
                idx[0] = start
                idx[-1] = end
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
                        boundary_type=1 if b == 0 else 2,
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
                label_taxonomy="synthetic_v1",
            ),
            graphs,
        )
    writer.finalize()
    return path


def _build_checkpoint(ckpt_path: Path, h5_path: Path, *, seed: int = 0) -> HierGNNModel:
    torch.manual_seed(seed)
    model = HierGNNModel(
        num_classes=_NUM_CLASSES,
        num_leads=_NUM_LEADS,
        in_channels=_IN_CHANNELS,
        edge_dim=_EDGE_DIM,
        hidden_channels=16,
        beat_encoder_layers=1,
    )
    save_checkpoint(
        ckpt_path,
        model=model,
        epoch=1,
        metric_dict={"f1/macro": 0.5},
        class_names=list(_CLASS_NAMES),
        dataset_fingerprint=read_hdf5_meta(h5_path).fingerprint,
        symmetrize_edges=False,
        subgraph_representation=SubgraphRepresentationConfig(
            adjacency="temporal_chain",
            edge_attributes="zero",
        ),
    )
    return model


def test_load_checkpoint_and_predict_shapes(tmp_path):
    h5 = _build_dataset(tmp_path / "data.h5", n_exams=6)
    ckpt = tmp_path / "ckpt.pt"
    _build_checkpoint(ckpt, h5)

    result = load_checkpoint_and_predict(ckpt, h5, torch.device("cpu"), batch_size=4, num_workers=0)

    assert isinstance(result, CheckpointPredictions)
    assert result.meta.total_exams == 6
    assert list(result.meta.class_names) == list(_CLASS_NAMES)
    assert result.predictions.logits.shape == (6, _NUM_CLASSES)
    assert result.predictions.labels.shape == (6, _NUM_CLASSES)
    assert np.isfinite(result.predictions.logits).all()
    assert result.state["class_names"] == list(_CLASS_NAMES)
    assert result.state["subgraph_representation"] == {
        "adjacency": "temporal_chain",
        "edge_attributes": "zero",
    }


def test_load_checkpoint_and_predict_deterministic_given_same_inputs(tmp_path):
    h5 = _build_dataset(tmp_path / "data.h5", n_exams=5)
    ckpt = tmp_path / "ckpt.pt"
    _build_checkpoint(ckpt, h5)

    r1 = load_checkpoint_and_predict(ckpt, h5, torch.device("cpu"), batch_size=4, num_workers=0)
    r2 = load_checkpoint_and_predict(ckpt, h5, torch.device("cpu"), batch_size=4, num_workers=0)

    np.testing.assert_array_equal(r1.predictions.labels, r2.predictions.labels)
    np.testing.assert_allclose(r1.predictions.logits, r2.predictions.logits, atol=1e-6)


def test_load_checkpoint_and_dataset_reconstructs_real_nondefault_representation(tmp_path):
    h5 = _build_dataset(tmp_path / "data.h5", n_exams=3)
    ckpt = tmp_path / "ckpt.pt"
    _build_checkpoint(ckpt, h5)

    with load_checkpoint_and_dataset(ckpt, h5, torch.device("cpu")) as bundle:
        assert bundle.dataset.subgraph_representation.model_dump() == {
            "adjacency": "temporal_chain",
            "edge_attributes": "zero",
        }
        item = bundle.dataset[0]
        assert item.edge_index.size(1) == 4 * 11
        assert item.edge_attr.shape == (4 * 11, 2)
        assert item.edge_attr.dtype == torch.float32
        assert torch.count_nonzero(item.edge_attr).item() == 0
