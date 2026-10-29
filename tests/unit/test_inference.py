"""Tests for training.inference, shared inference utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.hier_model import ModelOutput
from tests.unit.fixtures import MinimalBatch, MinimalModel

pytestmark = [pytest.mark.unit]


def test_forward_batch_returns_model_output():
    from training.inference import forward_batch

    model = MinimalModel()
    batch = MinimalBatch(y=torch.tensor([1.0, 0.0]))
    out = forward_batch(model, batch, torch.device("cpu"))

    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (1, 2)


def test_gather_predictions_shapes(monkeypatch):
    from training.inference import GatheredPredictions, gather_predictions

    monkeypatch.setattr("training.inference.ddp.all_gather_concat", lambda t: t.detach().cpu())
    monkeypatch.setattr("training.inference.ddp.all_reduce_sum", lambda t: t)
    monkeypatch.setattr("training.inference.ddp.is_main_process", lambda: True)

    loader = [
        MinimalBatch(y=torch.tensor([1.0, 0.0])),
        MinimalBatch(y=torch.tensor([0.0, 1.0])),
    ]
    preds = gather_predictions(MinimalModel(), loader, torch.device("cpu"), num_classes=2)

    assert isinstance(preds, GatheredPredictions)
    assert preds.logits.shape == (2, 2)
    assert preds.labels.shape == (2, 2)
    assert preds.logits.dtype.kind == "f"
    assert preds.labels.dtype == __import__("numpy").int64
    assert __import__("math").isfinite(preds.avg_loss)
    assert preds.avg_loss > 0.0


def test_gather_predictions_empty_loader(monkeypatch):
    from training.inference import gather_predictions

    monkeypatch.setattr("training.inference.ddp.all_gather_concat", lambda t: t.detach().cpu())
    monkeypatch.setattr("training.inference.ddp.all_reduce_sum", lambda t: t)
    monkeypatch.setattr("training.inference.ddp.is_main_process", lambda: True)

    preds = gather_predictions(MinimalModel(), [], torch.device("cpu"), num_classes=3)

    assert preds.logits.shape == (0, 3)
    assert preds.labels.shape == (0, 3)
    assert preds.avg_loss == 0.0


def test_gather_predictions_loss_matches_bce(monkeypatch):
    """avg_loss is unweighted BCEWithLogitsLoss reduced PER ELEMENT.

    Per-element (sum / numel) keeps it on the same scale as the training loop's
    mean-reduced train/loss_epoch; dividing by num samples would inflate it by a
    factor of num_classes.
    """
    import numpy as np

    from training.inference import gather_predictions

    monkeypatch.setattr("training.inference.ddp.all_gather_concat", lambda t: t.detach().cpu())
    monkeypatch.setattr("training.inference.ddp.all_reduce_sum", lambda t: t)
    monkeypatch.setattr("training.inference.ddp.is_main_process", lambda: True)

    batches = [
        MinimalBatch(y=torch.tensor([1.0, 0.0])),
        MinimalBatch(y=torch.tensor([0.0, 1.0])),
    ]
    preds = gather_predictions(MinimalModel(), batches, torch.device("cpu"), num_classes=2)

    logits_t = torch.tensor(preds.logits)
    labels_t = torch.tensor(preds.labels, dtype=torch.float32)
    expected = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits_t, labels_t, reduction="sum"
        ).item()
        / logits_t.numel()
    )
    np.testing.assert_allclose(preds.avg_loss, expected, rtol=1e-5)


def test_gather_predictions_restores_training_mode(monkeypatch):
    """Regression: gather_predictions must leave the model in its prior
    train/eval mode, not silently in eval mode."""
    from training.inference import gather_predictions

    monkeypatch.setattr("training.inference.ddp.all_gather_concat", lambda t: t.detach().cpu())
    monkeypatch.setattr("training.inference.ddp.all_reduce_sum", lambda t: t)
    monkeypatch.setattr("training.inference.ddp.is_main_process", lambda: True)

    batches = [MinimalBatch(y=torch.tensor([1.0, 0.0]))]

    model = MinimalModel()
    model.train()
    gather_predictions(model, batches, torch.device("cpu"), num_classes=2)
    assert model.training is True

    model.eval()
    gather_predictions(model, batches, torch.device("cpu"), num_classes=2)
    assert model.training is False


def test_gather_predictions_failure_gates_before_prediction_collectives(monkeypatch):
    from training.inference import gather_predictions

    reductions: list[torch.Tensor] = []

    def record_reduction(tensor):
        reductions.append(tensor.detach().clone())
        return tensor

    monkeypatch.setattr("training.inference.ddp.all_reduce_sum", record_reduction)
    monkeypatch.setattr(
        "training.inference.ddp.all_gather_concat",
        lambda tensor: pytest.fail("prediction gather followed failed local inference"),
    )
    monkeypatch.setattr("training.inference.ddp.is_main_process", lambda: True)

    class FailingLoader:
        def __len__(self):
            return 1

        def __iter__(self):
            raise RuntimeError("loader-sentinel")

    model = MinimalModel()
    model.train()
    with pytest.raises(RuntimeError, match="rank-local inference failed") as raised:
        gather_predictions(
            model,
            FailingLoader(),
            torch.device("cpu"),
            num_classes=2,
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "loader-sentinel"
    assert model.training is True
    assert len(reductions) == 1
    assert reductions[0].shape == ()
    assert reductions[0].item() == 1


def test_resolve_device_explicit_cpu():
    from training.inference import resolve_device

    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_never_returns_literal_auto():
    from training.inference import resolve_device

    dev = resolve_device("auto")
    assert dev.type in ("cpu", "cuda")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev.type == expected


class _FakeModel:
    def __init__(self, num_leads=2, num_classes=3):
        self.num_leads = num_leads
        self.num_classes = num_classes
        self.eval_called = False

    def eval(self):
        self.eval_called = True


class _FakeDataset:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


class _FakeMeta:
    def __init__(self, total_exams=5, class_names=("A", "B", "C")):
        self.total_exams = total_exams
        self.class_names = class_names


def _patch_predict_collaborators(
    monkeypatch,
    *,
    model=None,
    state=None,
    meta=None,
    preds="SENTINEL_PREDS",
    dataset_cls=_FakeDataset,
    gather_predictions_fn=None,
    require_compatible_dataset_fn=None,
):
    """Fake every collaborator predict_on_dataset / load_checkpoint_and_predict
    call, and return the recorder dicts so tests can assert on call args."""
    import training.inference as inf

    model = model if model is not None else _FakeModel()
    state = (
        state
        if state is not None
        else {
            "symmetrize_edges": False,
            "subgraph_representation": {
                "adjacency": "stored_visibility",
                "edge_attributes": "derived",
            },
        }
    )
    meta = meta if meta is not None else _FakeMeta()
    checkpoint_sha256 = "a" * 64

    reconstruct_calls: list[dict] = []
    require_calls: list[dict] = []
    dataset_calls: list[_FakeDataset] = []
    loader_calls: list[dict] = []
    gather_calls: list[dict] = []

    def fake_reconstruct(ckpt_path, *, device=None):
        reconstruct_calls.append({"ckpt_path": ckpt_path, "device": device})
        return model, state, checkpoint_sha256

    def fake_require_compatible(ckpt_state, dataset_path, *, allow_symmetrized=False):
        require_calls.append(
            {
                "ckpt_state": ckpt_state,
                "dataset_path": dataset_path,
                "allow_symmetrized": allow_symmetrized,
            }
        )
        return meta

    def fake_dataset_ctor(
        dataset_path,
        indices,
        *,
        num_leads,
        symmetrize_edges,
        subgraph_representation=None,
    ):
        ds = dataset_cls(
            dataset_path,
            indices,
            num_leads=num_leads,
            symmetrize_edges=symmetrize_edges,
            subgraph_representation=subgraph_representation,
        )
        dataset_calls.append(ds)
        return ds

    def fake_loader_ctor(dataset, *, batch_size, shuffle, num_workers):
        loader_calls.append(
            {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
            }
        )
        return "SENTINEL_LOADER"

    def fake_gather(model_arg, loader, device, num_classes, *, desc):
        gather_calls.append(
            {
                "model": model_arg,
                "loader": loader,
                "device": device,
                "num_classes": num_classes,
                "desc": desc,
            }
        )
        if gather_predictions_fn is not None:
            return gather_predictions_fn()
        return preds

    monkeypatch.setattr(inf, "reconstruct_model_from_ckpt_with_sha256", fake_reconstruct)
    monkeypatch.setattr(
        inf, "require_compatible_dataset", require_compatible_dataset_fn or fake_require_compatible
    )
    monkeypatch.setattr(inf, "ExamGraphDataset", fake_dataset_ctor)
    monkeypatch.setattr(inf, "PyGDataLoader", fake_loader_ctor)
    monkeypatch.setattr(inf, "gather_predictions", fake_gather)

    return {
        "model": model,
        "state": state,
        "meta": meta,
        "checkpoint_sha256": checkpoint_sha256,
        "reconstruct_calls": reconstruct_calls,
        "require_calls": require_calls,
        "dataset_calls": dataset_calls,
        "loader_calls": loader_calls,
        "gather_calls": gather_calls,
    }


def test_predict_on_dataset_closes_dataset_even_when_gather_raises(monkeypatch):
    """The dataset must be closed via try/finally, not only on the happy path.
    a leaked HDF5 handle on an exception is exactly the bug this helper must
    not reintroduce relative to the hand-written per-script try/finally blocks."""
    from training.inference import predict_on_dataset

    def _raise():
        raise RuntimeError("boom")

    rec = _patch_predict_collaborators(monkeypatch, gather_predictions_fn=_raise)

    with pytest.raises(RuntimeError, match="boom"):
        predict_on_dataset(rec["model"], rec["state"], Path("/fake.h5"), torch.device("cpu"))

    assert rec["dataset_calls"][0].closed is True


def test_checkpoint_dataset_bundle_closes_dataset_even_when_body_raises(monkeypatch):
    """The dataset must be closed even when the `with` block raises, the
    same leaked-handle-on-exception bug the hand-rolled try/finally blocks
    had to avoid."""
    from training.inference import load_checkpoint_and_dataset

    rec = _patch_predict_collaborators(monkeypatch)
    bundle = load_checkpoint_and_dataset(
        Path("/fake/ckpt.pt"), Path("/fake.h5"), torch.device("cpu")
    )
    ds = rec["dataset_calls"][0]

    with pytest.raises(RuntimeError, match="boom"), bundle:
        raise RuntimeError("boom")

    assert ds.closed is True
