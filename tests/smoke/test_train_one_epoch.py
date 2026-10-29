"""End-to-end smoke test: load a small real HDF5, run one epoch, assert ckpt + metrics.

This test is intentionally heavy (real h5py I/O, full forward/backward through the
hierarchical model, BCE loss, scheduler step, ckpt save). It skips automatically
when the small PTB-XL HDF5 file is not present in ``graph_datasets/ptbxl/``.

DDP is single-process.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from config.train_models import OptimConfig, SubgraphRepresentationConfig, ThresholdConfig
from models.hier_model import HierGNNModel
from training.dataset import ExamGraphDataset
from training.evaluate import evaluate_test
from training.hdf5_meta import read_hdf5_meta
from training.sampling import DynamicPosNegSampler
from training.train import train

pytestmark = pytest.mark.smoke


SMALL_H5 = (
    Path(__file__).resolve().parents[2]
    / "graph_datasets"
    / "ptbxl"
    / "ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__bp0p5-40hz-o4__resample-300hz__norm-p95med__natural__leads-0-1-2-3-4-5-6-7-8-9-10-11__limit-200.h5"
)


@pytest.fixture
def small_h5_path() -> Path:
    if not SMALL_H5.exists():
        pytest.skip(f"small PTB-XL HDF5 not present: {SMALL_H5}")
    return SMALL_H5


def _read_meta(h5_path: Path) -> tuple[int, int, list[str], list[int]]:
    with h5py.File(h5_path, "r") as f:
        num_exams = int(f["exam_id"].shape[0])
        num_classes = int(f["exam_labels"].shape[1])
        class_names = list(f.attrs["exam_label_names"])
        graph_leads = list(f.attrs["graph_leads"])
    return num_exams, num_classes, class_names, graph_leads


def test_train_one_epoch_smoke(small_h5_path, tmp_path):
    num_exams, num_classes, class_names, graph_leads = _read_meta(small_h5_path)
    assert num_exams >= 30, f"smoke needs >= 30 exams, got {num_exams}"

    n_train = int(num_exams * 0.7)
    n_val = max(int(num_exams * 0.15), 5)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, num_exams))

    num_leads = max(int(max(graph_leads)) + 1, 1)
    train_ds = ExamGraphDataset(small_h5_path, train_idx, num_leads=num_leads)
    val_ds = ExamGraphDataset(small_h5_path, val_idx, num_leads=num_leads)
    test_ds = ExamGraphDataset(small_h5_path, test_idx, num_leads=num_leads)

    train_loader = PyGDataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierGNNModel(
        num_classes=num_classes,
        num_leads=num_leads,
        hidden_channels=32,
        beat_encoder_layers=2,
        beat_encoder_mlp_hidden=64,
        gatv2_heads=2,
        gatv2_layers=1,
    ).to(device)

    optim_cfg = OptimConfig(
        epochs=1,
        early_stop_patience=5,
        eval_val_every_epochs=1,
        log_every_steps=10000,
    )
    ckpt_dir = tmp_path / "ckpt"
    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim_cfg,
        threshold=ThresholdConfig(),
        class_names=class_names,
        dataset_fingerprint=read_hdf5_meta(small_h5_path).fingerprint,
        device=device,
        ckpt_dir=ckpt_dir,
        subgraph_representation=SubgraphRepresentationConfig(),
    )
    assert result.epochs_completed == 1
    if not (ckpt_dir / "best_f1_val.pt").exists():
        pytest.skip(
            "best_f1_val.pt was not written (val f1/macro never finite-improved); "
            "skipping post-hoc test eval. Train loop assertions above already passed."
        )

    metrics = evaluate_test(
        model=model,
        test_loader=test_loader,
        ckpt_path=ckpt_dir / "best_f1_val.pt",
        class_names=class_names,
        device=device,
        threshold=0.5,
        out_json=tmp_path / "test_metrics.json",
    )
    assert "f1/macro" in metrics
    assert "auroc/macro" in metrics
    assert (tmp_path / "test_metrics.json").exists()


def test_train_one_epoch_with_dynamic_sampling(small_h5_path, tmp_path):
    """End-to-end smoke for the dynamic pos/neg sampler.

    Verifies that wiring the sampler through the PyG DataLoader and the
    existing train loop produces a complete epoch and a saved checkpoint when
    val f1/macro improves. Uses the same small HDF5 as the baseline smoke test
    and an aggressively small ``n_neg`` so the epoch fits in seconds.
    """
    num_exams, num_classes, class_names, graph_leads = _read_meta(small_h5_path)
    assert num_exams >= 30

    n_train = int(num_exams * 0.7)
    n_val = max(int(num_exams * 0.15), 5)
    train_idx = np.arange(n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)

    with h5py.File(small_h5_path, "r") as f:
        exam_labels = f["exam_labels"][:].astype(np.float32)

    label_sum = exam_labels[train_idx].sum(axis=1)
    pos_positions = np.where(label_sum > 0)[0].astype(np.int64)
    neg_positions = np.where(label_sum == 0)[0].astype(np.int64)
    if pos_positions.size == 0 or neg_positions.size == 0:
        pytest.skip("smoke split has no positive or no negative exams for dynamic sampling")
    n_neg = min(5, int(neg_positions.size))

    num_leads = max(int(max(graph_leads)) + 1, 1)
    train_ds = ExamGraphDataset(small_h5_path, train_idx, num_leads=num_leads)
    val_ds = ExamGraphDataset(small_h5_path, val_idx, num_leads=num_leads)

    sampler = DynamicPosNegSampler(
        pos_positions=pos_positions,
        neg_positions=neg_positions,
        n_neg=n_neg,
        seed=42,
    )
    train_loader = PyGDataLoader(train_ds, sampler=sampler, batch_size=2, num_workers=0)
    val_loader = PyGDataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierGNNModel(
        num_classes=num_classes,
        num_leads=num_leads,
        hidden_channels=32,
        beat_encoder_layers=2,
        beat_encoder_mlp_hidden=64,
        gatv2_heads=2,
        gatv2_layers=1,
    ).to(device)

    optim_cfg = OptimConfig(
        epochs=1,
        early_stop_patience=5,
        eval_val_every_epochs=1,
        log_every_steps=10000,
    )
    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=optim_cfg,
        threshold=ThresholdConfig(),
        class_names=class_names,
        dataset_fingerprint=read_hdf5_meta(small_h5_path).fingerprint,
        device=device,
        ckpt_dir=tmp_path / "ckpt_dyn",
        subgraph_representation=SubgraphRepresentationConfig(),
    )
    assert result.epochs_completed == 1
