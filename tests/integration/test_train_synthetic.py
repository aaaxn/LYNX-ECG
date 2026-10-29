"""End-to-end integration test: the real training loop + the real model on a
synthetic, in-repo HDF5 file.

This closes the seam between the two places the real ``HierGNNModel`` and the
real :func:`training.train.train` loop otherwise meet:

* ``tests/smoke/test_train_one_epoch.py`` exercises both, but requires a real
  built PTB-XL HDF5 and auto-skips wherever that file is absent (every
  machine without data, and any future CI).
* Unit-level loop tests (``tests/unit/test_train_*.py``) exercise ``train()``
  against ``tests/unit/fixtures.py::MinimalModel``, a stub returning constant
  zero logits, never the real architecture.

Building a tiny but schema-valid HDF5 in ``tmp_path`` (the same pattern
``tests/integration/test_dataset_supergraph.py`` already uses for a bare
model forward pass) lets this test run the full pipeline. Optimizer
construction over real parameters, scheduler stepping, threshold evaluation
on real logits, checkpoint writing, and the ``reconstruct_model_from_ckpt``
round-trip, all with no external data and no GPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from config.train_models import (
    ModelConfig,
    OptimConfig,
    SamplingConfig,
    SubgraphRepresentationConfig,
    ThresholdConfig,
)
from dataset.hdf5_writer import DatasetWriter
from tests.fixtures.synthetic_graph_hdf5 import make_beat_graph as _make_graph
from tests.fixtures.synthetic_graph_hdf5 import make_exam_metadata as _make_exam_metadata
from tests.fixtures.synthetic_graph_hdf5 import tiny_dataset_config as _dataset_config
from tests.fixtures.synthetic_graph_hdf5 import write_exam
from training.checkpoint import reconstruct_model_from_ckpt
from training.dataset import ExamGraphDataset
from training.hdf5_meta import DatasetFingerprint, HDF5Meta, read_hdf5_meta
from training.loaders import build_loaders
from training.train import TrainResult, train

pytestmark = pytest.mark.integration

CLASS_NAMES = ("1dAVb", "RBBB")
_LABEL_TAXONOMY = "synthetic_binary_v1"

_POSITIVE_CLASS_BY_EXAM = [0, 1, 0, 1, 0, 1, 0, 1, None, None]

TRAIN_IDX = np.array([0, 1, 2, 3, 4, 5, 8, 9], dtype=np.int64)
VAL_IDX = np.arange(6, 8, dtype=np.int64)


def _build_synthetic_h5(tmp_path: Path) -> Path:
    """Build a 10-exam, 2-lead, 3-beats-per-lead, 2-class synthetic HDF5.

    Deterministic (seeded via ``make_beat_graph``'s hash-derived RNG); the
    label assignment alternates classes across exams so both classes have
    positive AND negative support in both the train (positions 0-5, 8-9) and
    val (positions 6-7) splits used by the tests below. Exams 8-9 have no
    positive label and form the negative pool of the train sampler.
    """
    out = tmp_path / "synthetic_train.h5"
    cfg = _dataset_config()
    writer = DatasetWriter(
        out,
        config=cfg,
        label_taxonomy=_LABEL_TAXONOMY,
        label_names=CLASS_NAMES,
        created_by_version="test",
    )
    for i, positive_class in enumerate(_POSITIVE_CLASS_BY_EXAM):
        exam_id = f"E{i:03d}"
        fold = 9 if i in VAL_IDX else 1
        graphs = [
            _make_graph(exam_id=exam_id, lead=lead, beat_index=beat)
            for lead in (0, 1)
            for beat in (0, 1, 2)
        ]
        metadata = _make_exam_metadata(
            exam_id, positive_class, fold, label_names=CLASS_NAMES, label_taxonomy=_LABEL_TAXONOMY
        )
        write_exam(writer, metadata, graphs)
    writer.finalize()
    return out


def test_synthetic_dataset_builds_and_iterates(tmp_path):
    """Sanity check on the fixture itself before it's used by the loop test."""
    h5_path = _build_synthetic_h5(tmp_path)
    n_exams = len(_POSITIVE_CLASS_BY_EXAM)
    ds = ExamGraphDataset(h5_path, exam_indices=list(range(n_exams)), num_leads=2)
    for i, positive_class in enumerate(_POSITIVE_CLASS_BY_EXAM):
        expected = torch.zeros(2)
        if positive_class is not None:
            expected[positive_class] = 1.0
        assert torch.equal(ds[i].y, expected)


def _build_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return ModelConfig(
        hidden_channels=16,
        beat_encoder={"layers": 1, "mlp_hidden": 16},
        gatv2_heads=2,
        gatv2_layers=1,
    ).build(num_classes=2, num_leads=2)


def _build_loaders(
    h5_path: Path,
    meta: HDF5Meta,
    representation: SubgraphRepresentationConfig | None = None,
):
    return build_loaders(
        dataset_path=h5_path,
        num_leads=meta.num_leads,
        train_idx=TRAIN_IDX,
        val_idx=VAL_IDX,
        test_idx=VAL_IDX,
        exam_labels=meta.exam_labels,
        sampling=SamplingConfig(n_neg=2),
        batch_size=2,
        num_workers=0,
        distributed=False,
        seed=42,
        device=torch.device("cpu"),
        subgraph_representation=representation,
    )


def _run_training(
    h5_path: Path,
    meta: HDF5Meta,
    ckpt_dir: Path,
    *,
    model_seed: int,
    representation: SubgraphRepresentationConfig | None = None,
) -> TrainResult:
    representation = representation or SubgraphRepresentationConfig()
    model = _build_model(model_seed)
    train_loader, val_loader, _test_loader = _build_loaders(h5_path, meta, representation)
    optim = OptimConfig(
        epochs=2,
        batch_size=2,
        num_workers=0,
        compile=False,
        lr=1e-3,
        eval_val_every_epochs=1,
        log_every_steps=1000,
        early_stop_patience=10,
        seed=42,
    )
    return train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        threshold=ThresholdConfig(),
        class_names=meta.class_names,
        dataset_fingerprint=meta.fingerprint,
        device=torch.device("cpu"),
        ckpt_dir=ckpt_dir,
        subgraph_representation=representation,
    )


def test_train_two_epochs_synthetic(tmp_path):
    h5_path = _build_synthetic_h5(tmp_path)
    meta = read_hdf5_meta(h5_path)
    assert meta.num_classes == 2
    assert meta.num_leads == 2
    representation = SubgraphRepresentationConfig(
        adjacency="temporal_chain",
        edge_attributes="zero",
    )

    ckpt_dir = tmp_path / "ckpt"
    model = _build_model(seed=42)
    train_loader, val_loader, _test_loader = _build_loaders(h5_path, meta, representation)
    optim = OptimConfig(
        epochs=2,
        batch_size=2,
        num_workers=0,
        compile=False,
        lr=1e-3,
        eval_val_every_epochs=1,
        log_every_steps=1000,
        early_stop_patience=10,
        seed=42,
    )
    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim,
        threshold=ThresholdConfig(),
        class_names=meta.class_names,
        dataset_fingerprint=meta.fingerprint,
        device=torch.device("cpu"),
        ckpt_dir=ckpt_dir,
        subgraph_representation=representation,
    )

    assert isinstance(result, TrainResult)
    assert math.isfinite(result.best_f1_val)
    assert 0.0 <= result.best_f1_val <= 1.0
    assert result.epochs_completed in (1, 2)

    assert math.isfinite(result.best_pr_auc_val)
    assert 0.0 <= result.best_pr_auc_val <= 1.0

    ckpt_path = ckpt_dir / "best_f1_val.pt"
    assert ckpt_path.exists()
    assert (ckpt_dir / "best_pr_auc_val.pt").exists()

    reconstructed_model, state = reconstruct_model_from_ckpt(ckpt_path)
    assert reconstructed_model.hparams == model.hparams

    ckpt_fingerprint = DatasetFingerprint(**state["dataset_fingerprint"])
    assert ckpt_fingerprint == meta.fingerprint
    assert state["subgraph_representation"] == representation.model_dump()
    _, pr_auc_state = reconstruct_model_from_ckpt(ckpt_dir / "best_pr_auc_val.pt")
    assert pr_auc_state["subgraph_representation"] == representation.model_dump()


def test_train_two_epochs_synthetic_is_deterministic(tmp_path):
    """Same seeds, same synthetic data -> same best_f1_val (CPU, no compile).

    Documents the seed contract of :func:`training.train.train`: reseeding
    ``torch``/``numpy`` from ``optim.seed`` at the top of the loop, combined
    with seeding before model construction, makes a from-scratch rerun
    reproduce the same result bit-for-bit on CPU.
    """
    h5_path = _build_synthetic_h5(tmp_path)
    meta = read_hdf5_meta(h5_path)

    result1 = _run_training(h5_path, meta, tmp_path / "ckpt1", model_seed=42)
    result2 = _run_training(h5_path, meta, tmp_path / "ckpt2", model_seed=42)

    assert math.isfinite(result1.best_f1_val)
    assert math.isfinite(result2.best_f1_val)
    assert result1.best_f1_val == result2.best_f1_val
    assert result1.best_pr_auc_val == result2.best_pr_auc_val
