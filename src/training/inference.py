"""Shared inference utilities for training and evaluation.

Public surface:
- :func:`forward_batch`. Moves a PyG batch to device and calls the model.
- :func:`bf16_autocast`. BFloat16 autocast context, enabled only on CUDA.
- :func:`gather_predictions`. Runs inference on a loader, DDP all-gather,
  returns collected predictions.
- :class:`GatheredPredictions`. Return type of :func:`gather_predictions`.
- :func:`predict_on_dataset`. Validates + scores one HDF5 dataset against an
  already-loaded checkpoint's model.
- :func:`load_checkpoint_and_predict`. Loads a checkpoint and scores one
  dataset in a single call; composes :func:`training.checkpoint.reconstruct_model_from_ckpt`
  and :func:`predict_on_dataset`.
- :class:`CheckpointPredictions`. Return type of :func:`load_checkpoint_and_predict`
  and (as a two-tuple) :func:`predict_on_dataset`.
- :func:`load_checkpoint_and_dataset`. Loads a checkpoint and opens an
  ``ExamGraphDataset`` over one HDF5 dataset, for callers (per-exam
  explanation/eval scripts) that need the dataset itself rather than just
  scored predictions. Unlike :func:`predict_on_dataset`, the dataset is
  handed to the caller OPEN; see :class:`CheckpointDatasetBundle`.
- :class:`CheckpointDatasetBundle`. Return type of
  :func:`load_checkpoint_and_dataset`; also a context manager that closes
  its ``dataset`` on exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.auto import tqdm

from training import ddp
from training.checkpoint import (
    reconstruct_model_from_ckpt_with_sha256,
    resolve_subgraph_representation,
    resolve_symmetrize_edges,
)
from training.dataset import ExamGraphDataset
from training.hdf5_meta import require_compatible_dataset

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from models.hier_model import ModelOutput
    from training.hdf5_meta import HDF5Meta


@dataclass(frozen=True)
class GatheredPredictions:
    """Collected predictions after DDP all-gather."""

    logits: np.ndarray
    labels: np.ndarray
    avg_loss: float


@dataclass(frozen=True)
class LocalPredictions:
    """Rank-local inference outputs before any distributed aggregation."""

    logits: torch.Tensor
    labels: torch.Tensor
    loss_total: torch.Tensor
    element_count: int


def bf16_autocast(device: torch.device):
    return torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a ``--device`` CLI argument to a :class:`torch.device`.

    ``"auto"`` selects CUDA when available, else CPU. Any other value is passed
    to ``torch.device`` verbatim (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``).
    The naive ``"cuda" if auto and is_available() else spec`` idiom is wrong: on
    a CPU-only host it yields ``torch.device("auto")``, which raises.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def batch_to_forward_kwargs(batch: Any) -> dict[str, Any]:
    """Extract forward-pass keyword arguments from a PyG ``Batch`` object.

    This is the single source of truth for the field → kwarg mapping. Any code
    that needs to call the model on a collated batch must go through here rather
    than re-listing the fields inline, so that future schema changes only need
    one edit.

    The batch must already be on the desired device before calling this function.
    ``subgraph_beat`` is passed as ``None`` when the batch lacks it; the model
    forward accepts that.
    """
    return {
        "x": batch.x,
        "edge_index": batch.edge_index,
        "edge_attr": batch.edge_attr,
        "subgraph_node_batch": batch.subgraph_node_batch,
        "super_edge_index": batch.super_edge_index,
        "super_edge_step": batch.super_edge_step,
        "subgraph_lead": batch.subgraph_lead,
        "subgraph_exam": batch.subgraph_exam,
        "batch_size": int(batch.num_graphs),
        "subgraph_beat": getattr(batch, "subgraph_beat", None),
    }


def single_exam_inputs(
    dataset: ExamGraphDataset, exam_idx: int, device: torch.device
) -> dict[str, Any]:
    """Forward-kwargs dict for one exam, moved to ``device``.

    The returned ``x`` rows match ``dataset[exam_idx].x`` (HDF5 node order), so
    Integrated-Gradients importances, timing masks, and attention expansion all
    align row-for-row with the dataset item.  Only the keys the model's
    ``forward`` accepts are returned; ``y`` and PyG bookkeeping fields are
    excluded so strict forward-signature validation does not reject the call.
    """
    return batch_to_forward_kwargs(Batch.from_data_list([dataset[exam_idx]]).to(device))


def forward_batch(model: nn.Module, batch: Any, device: torch.device) -> ModelOutput:
    batch = batch.to(device, non_blocking=True)
    return model(**batch_to_forward_kwargs(batch))


def collect_local_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    desc: str = "eval",
) -> LocalPredictions:
    """Run rank-local inference without entering a distributed collective."""
    was_training = model.training
    try:
        model.eval()
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        total_loss = torch.zeros((), dtype=torch.float64, device=device)
        total_elements = 0
        logits_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []

        bar = tqdm(
            loader,
            desc=desc,
            leave=False,
            disable=not ddp.is_main_process(),
            total=len(loader) if hasattr(loader, "__len__") else None,
        )
        with torch.inference_mode():
            for batch in bar:
                with bf16_autocast(device):
                    out = forward_batch(model, batch, device)
                    logits = out.logits
                    target = batch.y.view(logits.shape).float().to(device)
                    loss = loss_fn(logits, target)
                total_loss += loss.detach().double()
                total_elements += logits.numel()
                logits_parts.append(logits.detach().float())
                target_parts.append(target.detach())

        if logits_parts:
            local_logits = torch.cat(logits_parts, dim=0)
            local_labels = torch.cat(target_parts, dim=0)
        else:
            local_logits = torch.empty((0, num_classes), dtype=torch.float32, device=device)
            local_labels = torch.empty((0, num_classes), dtype=torch.float32, device=device)
        return LocalPredictions(
            logits=local_logits,
            labels=local_labels,
            loss_total=total_loss,
            element_count=total_elements,
        )
    finally:
        model.train(was_training)


def gather_local_predictions(
    local: LocalPredictions,
    device: torch.device,
) -> GatheredPredictions:
    """Aggregate already-collected rank-local outputs in a fixed collective sequence."""
    loss_totals = torch.stack(
        [
            local.loss_total,
            torch.tensor(float(local.element_count), dtype=torch.float64, device=device),
        ]
    )
    ddp.all_reduce_sum(loss_totals)
    avg_loss = float(loss_totals[0].item() / max(loss_totals[1].item(), 1.0))
    logits_all = ddp.all_gather_concat(local.logits).numpy()
    labels_all = ddp.all_gather_concat(local.labels).numpy().astype(np.int64)
    return GatheredPredictions(logits=logits_all, labels=labels_all, avg_loss=avg_loss)


def gather_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    desc: str = "eval",
) -> GatheredPredictions:
    """Run local inference, failure-gate every rank, then aggregate predictions."""
    local: LocalPredictions | None = None
    local_error: Exception | None = None
    try:
        local = collect_local_predictions(
            model,
            loader,
            device,
            num_classes,
            desc=desc,
        )
    except Exception as exc:
        local_error = exc

    failed = torch.tensor(int(local_error is not None), dtype=torch.int32, device=device)
    ddp.all_reduce_sum(failed)
    if failed.item():
        if local_error is not None:
            raise RuntimeError("rank-local inference failed") from local_error
        raise RuntimeError("inference failed on another distributed rank")

    assert local is not None
    return gather_local_predictions(local, device)


@dataclass(frozen=True)
class CheckpointPredictions:
    """Result of :func:`load_checkpoint_and_predict` (and, as a two-tuple
    ``(meta, predictions)``, of :func:`predict_on_dataset`).

    ``state`` is the full raw checkpoint payload (see
    :mod:`training.checkpoint`), with ``class_names``, ``epoch``,
    and ``dataset_fingerprint``, so callers do not need a second checkpoint read.
    ``meta`` is the validated HDF5 metadata of the dataset that was scored
    (``meta.class_names`` is the dataset's class order, which
    :func:`training.hdf5_meta.require_compatible_dataset` has already proven
    matches ``state["class_names"]``).
    """

    model: nn.Module
    state: dict[str, Any]
    checkpoint_sha256: str
    meta: HDF5Meta
    predictions: GatheredPredictions


def _validate_and_open_dataset(
    model: nn.Module,
    ckpt_state: dict[str, Any],
    dataset_path: Path,
    *,
    allow_symmetrized: bool,
) -> tuple[HDF5Meta, ExamGraphDataset]:
    """Validate ``dataset_path`` against ``ckpt_state``, then open it.

    Shared construction logic for :func:`predict_on_dataset` (which closes the
    returned dataset itself before returning) and
    :func:`load_checkpoint_and_dataset` (which hands the open dataset to the
    caller). Kept in one place so the two never drift on how the dataset is
    built from a validated checkpoint/dataset pair.
    """
    meta = require_compatible_dataset(ckpt_state, dataset_path, allow_symmetrized=allow_symmetrized)
    dataset = ExamGraphDataset(
        dataset_path,
        np.arange(meta.total_exams),
        num_leads=model.num_leads,
        symmetrize_edges=resolve_symmetrize_edges(ckpt_state),
        subgraph_representation=resolve_subgraph_representation(ckpt_state),
    )
    return meta, dataset


def predict_on_dataset(
    model: nn.Module,
    ckpt_state: dict[str, Any],
    dataset_path: Path,
    device: torch.device,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    allow_symmetrized: bool = False,
    desc: str = "eval",
) -> tuple[HDF5Meta, GatheredPredictions]:
    """Validate ``dataset_path`` against ``ckpt_state``, then score it.

    Builds a fresh :class:`~training.dataset.ExamGraphDataset` over every exam
    in the file, wraps it in a non-shuffling ``PyGDataLoader``, and runs
    :func:`gather_predictions`. The dataset is opened and closed within this
    call (``finally: dataset.close()``); callers that need the dataset itself
    afterward (e.g. per-exam explanation scripts that re-run single exams
    through the model) should call :func:`load_checkpoint_and_dataset` instead,
    which hands back the same kind of dataset left open.

    Use this directly, passing an already-loaded ``model``/``ckpt_state``,
    when the same checkpoint is scored against multiple datasets (e.g. an
    in-domain vs. external ROC overlay); reloading the checkpoint per dataset
    would re-read and re-construct the model weights for no reason. For the
    common case of exactly one (checkpoint, dataset) pair, use
    :func:`load_checkpoint_and_predict` instead.

    ``allow_symmetrized`` is forwarded to
    :func:`~training.hdf5_meta.require_compatible_dataset` and also decides
    whether the dataset is built with edge symmetrization: passing ``True``
    without also handling a ``symmetrize_edges=True`` checkpoint the way the
    ``evaluate`` CLI does is a caller bug, since the resulting metrics would
    silently be computed on directed graphs the model was not trained with.
    """
    meta, dataset = _validate_and_open_dataset(
        model, ckpt_state, dataset_path, allow_symmetrized=allow_symmetrized
    )
    try:
        loader = PyGDataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        preds = gather_predictions(model, loader, device, num_classes=model.num_classes, desc=desc)
    finally:
        dataset.close()
    return meta, preds


def load_checkpoint_and_predict(
    ckpt_path: Path,
    dataset_path: Path,
    device: torch.device,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    allow_symmetrized: bool = False,
    desc: str = "eval",
) -> CheckpointPredictions:
    """Load a checkpoint and score one dataset, the common one-shot case.

    Composes :func:`~training.checkpoint.reconstruct_model_from_ckpt` and
    :func:`predict_on_dataset` for scripts that evaluate exactly one
    (checkpoint, dataset) pair (bootstrap CIs, error analyses). Scripts that reuse one loaded model
    across several datasets should call ``reconstruct_model_from_ckpt`` once
    and :func:`predict_on_dataset` per dataset instead, to avoid reloading the
    checkpoint's weights redundantly.
    """
    model, state, checkpoint_sha256 = reconstruct_model_from_ckpt_with_sha256(
        ckpt_path, device=device
    )
    meta, preds = predict_on_dataset(
        model,
        state,
        dataset_path,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        allow_symmetrized=allow_symmetrized,
        desc=desc,
    )
    return CheckpointPredictions(
        model=model,
        state=state,
        checkpoint_sha256=checkpoint_sha256,
        meta=meta,
        predictions=preds,
    )


@dataclass(frozen=True)
class CheckpointDatasetBundle:
    """Result of :func:`load_checkpoint_and_dataset`.

    Unlike :class:`CheckpointPredictions`, ``dataset`` is left OPEN. This
    bundle exists for callers that need direct per-exam access to the model
    and dataset (Integrated Gradients, region occlusion, latency measurement,
    beat-panel extraction for figures), not just a scored prediction matrix.

    The caller owns ``dataset``'s lifecycle from here on. This class is a
    context manager purely as a convenience for that ownership: entering
    returns ``self`` unchanged; exiting (normally or via exception) calls
    ``dataset.close()``, exactly like a ``try/finally: dataset.close()``
    block::

        with load_checkpoint_and_dataset(ckpt_path, dataset_path, device) as bundle:
            ...  # use bundle.model, bundle.dataset, ...
        # bundle.dataset is closed here, even if the block raised.

    Using the bundle this way is optional, since ``bundle.dataset.close()`` may
    also be called directly in a caller-written ``try/finally``, but the
    context-manager form is preferred for new call sites since it cannot
    forget the ``finally``.
    """

    model: nn.Module
    state: dict[str, Any]
    checkpoint_sha256: str
    meta: HDF5Meta
    dataset: ExamGraphDataset

    def __enter__(self) -> CheckpointDatasetBundle:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.dataset.close()


def load_checkpoint_and_dataset(
    ckpt_path: Path,
    dataset_path: Path,
    device: torch.device,
    *,
    allow_symmetrized: bool = False,
) -> CheckpointDatasetBundle:
    """Load a checkpoint and open an ``ExamGraphDataset`` over one HDF5 file.

    Per-exam explanation and evaluation scripts (the clinical-alignment gate,
    the Prominence workflows, the rhythm-lead audit, the efficiency report) use
    this instead of :func:`predict_on_dataset` /
    :func:`load_checkpoint_and_predict`, because those close the dataset before
    returning, exactly the opposite of what per-exam re-inference needs.

    Reconstructs the model, puts it in eval mode (single-exam inference,
    meaning IG, occlusion, and latency timing, must not run through the model
    in train mode), validates the dataset against the checkpoint via
    :func:`~training.hdf5_meta.require_compatible_dataset` (raises
    ``ValueError``, NOT caught here, on any incompatibility), then opens an
    ``ExamGraphDataset`` over every exam in the file.

    The returned dataset is left OPEN. See :class:`CheckpointDatasetBundle`
    for the lifecycle contract. ``allow_symmetrized`` is forwarded to
    ``require_compatible_dataset`` and also decides whether the dataset is
    built with edge symmetrization, exactly as in :func:`predict_on_dataset`.
    """
    model, state, checkpoint_sha256 = reconstruct_model_from_ckpt_with_sha256(
        ckpt_path, device=device
    )
    model.eval()
    meta, dataset = _validate_and_open_dataset(
        model, state, dataset_path, allow_symmetrized=allow_symmetrized
    )
    return CheckpointDatasetBundle(
        model=model,
        state=state,
        checkpoint_sha256=checkpoint_sha256,
        meta=meta,
        dataset=dataset,
    )
