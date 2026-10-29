"""Training loop with DDP support and checkpoint selection.

Public entry: :func:`train` runs the full schedule given a resolved config and
already-built data loaders. CLI orchestration (path resolution, launcher
dispatch) lives in ``src/cli/train.py`` (commit 4); this module is
launcher-agnostic so it can be unit-tested in isolation.

Checkpointing rule:
- ``best_f1_val.pt``: saved whenever ``f1/macro`` on val improves.
- ``best_pr_auc_val.pt``: saved whenever ``pr_auc/macro`` on val improves.
  This is the natural recalibration candidate when the best-F1 checkpoint
  turns out miscalibrated at the fixed 0.5 threshold.
- Final test evaluation (in :mod:`training.evaluate`) loads ``best_f1_val.pt``.

Early stop is triggered when ``f1/macro`` on val has not improved for
``early_stop_patience`` consecutive eval rounds. The LR scheduler is
``ReduceLROnPlateau`` driven by the same metric.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config.train_models import OptimConfig, SubgraphRepresentationConfig, ThresholdConfig
from training import ddp
from training.checkpoint import save_checkpoint
from training.evaluate import evaluate_loader
from training.inference import bf16_autocast, forward_batch

if TYPE_CHECKING:
    from models.hier_model import ModelOutput
    from training.hdf5_meta import DatasetFingerprint


@dataclass
class TrainResult:
    best_f1_val: float
    best_pr_auc_val: float
    epochs_completed: int
    history: list[dict[str, float]]


@dataclass
class MicroBatchStats:
    """Detached per-micro-batch scalars the epoch loop folds into its running
    training statistics, the same quantities the pre-accumulation loop read off
    each :class:`ModelOutput`. Kept detached and scalar so a whole window's
    worth never pins autograd graphs (or logits) in memory.
    """

    loss: torch.Tensor
    batch_size: int


@dataclass
class EpochTotals:
    """Rank-local running sums returned by :func:`run_train_epoch`, ready for the
    single cross-rank all-reduce the epoch summary performs. ``example_count``
    weights the loss means.
    """

    loss_weighted_sum: torch.Tensor
    example_count: int


def configure_matmul_precision() -> None:
    """Enable TF32 tensor cores for float32 matmuls.

    Covers the fp32 matmuls that run outside the bf16 autocast regions;
    optimizer state math stays fp32 either way. Process-global and
    idempotent. Called by :func:`train` and by the training benchmark so
    measured step times match real runs.
    """
    torch.set_float32_matmul_precision("high")


def build_optimizer(
    model: nn.Module, optim: OptimConfig, device: torch.device
) -> torch.optim.AdamW:
    """Construct the training AdamW exactly as the training loop uses it.

    On CUDA the fused implementation is used: it applies the whole update in
    one multi-tensor kernel instead of dozens of small launches, which matters
    for this model (many small parameter tensors, launch-overhead-bound steps).
    The fused kernel requires all parameters on a CUDA device, so CPU runs fall
    back to the default (foreach) implementation.

    Single source of truth: ``scripts/training/benchmark_training.py`` imports this so
    benchmark step timings always measure the real optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=optim.lr,
        weight_decay=optim.weight_decay,
        fused=device.type == "cuda",
    )


def run_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    grad_clip_norm: float | None,
) -> tuple[ModelOutput, torch.Tensor]:
    """Run one optimizer step on ``batch``: the single source of truth for the
    training loop's per-batch body, shared with ``scripts/training/benchmark_training.py``
    so measured step times exercise the exact same op sequence as a real run.

    Order (bit-identical to what a caller must not reorder): zero the
    gradients -> forward the batch under :func:`bf16_autocast` -> compute
    ``loss_fn(logits, target)`` (still inside the autocast
    scope) -> backward -> optional grad-norm clipping -> optimizer step.

    Returns ``(out, loss)``, the raw model output and the scalar training
    loss, so the caller can accumulate the example-weighted loss.
    """
    optimizer.zero_grad(set_to_none=True)
    with bf16_autocast(device):
        out = forward_batch(model, batch, device)
        logits = out.logits
        target = batch.y.view(logits.shape).float().to(device)
        loss = loss_fn(logits, target)
    loss.backward()
    if grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, foreach=True)
    optimizer.step()
    return out, loss


def _iter_windows[T](iterable: Iterable[T], window_size: int) -> Iterator[list[T]]:
    """Yield consecutive windows of up to ``window_size`` items from ``iterable``.

    The final window is short when the item count is not a multiple of
    ``window_size``, the leftover partial window that still earns its own
    optimizer step. Items are pulled lazily (only one window is held at a time),
    so wrapping a streaming ``DataLoader`` never materializes a whole epoch.
    """
    iterator = iter(iterable)
    while True:
        window = list(itertools.islice(iterator, window_size))
        if not window:
            return
        yield window


def _normalize_gradients_(model: nn.Module, denom: int) -> None:
    """In place, divide every accumulated gradient by ``denom`` so a window's
    summed-loss gradients become the gradient of the mean loss over its examples.
    """
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(denom)


def run_accumulated_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    window: Sequence[Any],
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    grad_clip_norm: float | None,
) -> list[MicroBatchStats]:
    """Run exactly ONE optimizer step over ``window`` (one to
    ``gradient_accumulation_steps`` micro-batches) with example-weighted
    gradient accumulation.

    Each micro-batch contributes ``loss_i * n_i`` to the backward pass (``n_i`` =
    its example count), so once summed across the window the accumulated
    ``.grad`` is the gradient of the *summed* per-example loss; dividing once by
    the window's total example count ``N`` turns it into the gradient of the
    *mean* loss over the concatenated examples, identical to a single
    forward/backward on one batch of all ``N`` examples for a mean-reduction
    ``loss_fn``. Gradients are zeroed once at the window start, clipped once, and
    the optimizer steps once.

    A single-micro-batch window (the default ``gradient_accumulation_steps == 1``
    case, and any leftover window of one) delegates to :func:`run_training_step`
    so that path stays bit-for-bit identical to non-accumulated training, with no
    ``*n_i`` / ``÷N`` round-trip on the gradients.

    Under DDP the all-reduce is suppressed on every micro-batch except the last
    via :func:`ddp.no_sync_context`; the final micro-batch's backward reduces the
    whole accumulated gradient, which DDP averages across ranks, so with equal
    per-rank example counts the result matches the mean over the global effective
    batch. Returns one :class:`MicroBatchStats` per micro-batch for the caller's
    running epoch statistics.
    """
    if len(window) == 1:
        out, loss = run_training_step(model, optimizer, window[0], device, loss_fn, grad_clip_norm)
        return [
            MicroBatchStats(
                loss=loss.detach(),
                batch_size=out.logits.size(0),
            )
        ]

    optimizer.zero_grad(set_to_none=True)
    stats: list[MicroBatchStats] = []
    total_examples = 0
    last_index = len(window) - 1
    for index, batch in enumerate(window):
        with ddp.no_sync_context(model, sync=index == last_index):
            with bf16_autocast(device):
                out = forward_batch(model, batch, device)
                logits = out.logits
                target = batch.y.view(logits.shape).float().to(device)
                loss = loss_fn(logits, target)
            batch_size = logits.size(0)
            (loss * batch_size).backward()
        total_examples += batch_size
        stats.append(
            MicroBatchStats(
                loss=loss.detach(),
                batch_size=batch_size,
            )
        )
    _normalize_gradients_(model, total_examples)
    if grad_clip_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, foreach=True)
    optimizer.step()
    return stats


def run_train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optim: OptimConfig,
    *,
    epoch: int,
) -> EpochTotals:
    """Train one epoch, grouping the loader's micro-batches into windows of
    ``optim.gradient_accumulation_steps`` and taking one optimizer step per
    window (:func:`run_accumulated_training_step`). Returns the rank-local running sums for the epoch summary. With the
    default ``gradient_accumulation_steps == 1`` every window is a single
    micro-batch and this is exactly the pre-accumulation per-batch loop.
    """
    running_loss = torch.zeros((), dtype=torch.float64, device=device)
    running_count = 0

    train_bar = tqdm(
        train_loader,
        desc=f"epoch {epoch}",
        leave=False,
        disable=not ddp.is_main_process(),
        total=len(train_loader) if hasattr(train_loader, "__len__") else None,
    )
    for optim_step, window in enumerate(
        _iter_windows(train_bar, optim.gradient_accumulation_steps), start=1
    ):
        stats = run_accumulated_training_step(
            model, optimizer, window, device, loss_fn, optim.grad_clip_norm
        )
        for micro in stats:
            running_loss += micro.loss.double() * micro.batch_size
            running_count += micro.batch_size
        if ddp.is_main_process() and optim_step % optim.log_every_steps == 0:
            train_bar.set_postfix(loss=f"{float(stats[-1].loss.item()):.4f}", refresh=False)

    return EpochTotals(
        loss_weighted_sum=running_loss,
        example_count=running_count,
    )


def train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optim: OptimConfig,
    threshold: ThresholdConfig,
    class_names: Sequence[str],
    dataset_fingerprint: DatasetFingerprint,
    device: torch.device,
    ckpt_dir: Path,
    subgraph_representation: SubgraphRepresentationConfig,
    symmetrize_edges: bool = False,
) -> TrainResult:
    """Run the full training schedule.

    Returns the run history (per-eval metric dicts) and best val metrics.
    Both checkpoints (``best_f1_val.pt``, ``best_pr_auc_val.pt``) are written
    by rank 0 only when their respective metric improves. The training loss
    is the unweighted mean ``BCEWithLogitsLoss``.
    """
    torch.manual_seed(optim.seed)
    np.random.seed(optim.seed)

    configure_matmul_precision()

    optimizer = build_optimizer(model, optim, device)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=optim.scheduler_factor, patience=optim.scheduler_patience
    )
    loss_fn = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_pr_auc = -1.0
    patience_left = optim.early_stop_patience

    epoch_iter = tqdm(
        range(1, optim.epochs + 1),
        desc="epochs",
        disable=not ddp.is_main_process(),
        total=optim.epochs,
    )
    for epoch in epoch_iter:
        model.train()
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        totals = run_train_epoch(
            model,
            optimizer,
            train_loader,
            device,
            loss_fn,
            optim,
            epoch=epoch,
        )
        train_totals = torch.stack(
            [
                totals.loss_weighted_sum,
                torch.tensor(float(totals.example_count), dtype=torch.float64, device=device),
            ]
        )
        ddp.all_reduce_sum(train_totals)
        r_loss, r_count = train_totals
        train_loss = float(r_loss.item() / max(r_count.item(), 1.0))
        epoch_time = time.time() - epoch_start

        base_metrics = {
            "train/loss_epoch": train_loss,
            "epoch": epoch,
            "epoch_time_s": epoch_time,
        }

        do_eval = (epoch % max(1, optim.eval_val_every_epochs)) == 0
        if not do_eval:
            continue

        val_report = evaluate_loader(
            model, val_loader, device, class_names=class_names, threshold=threshold.value
        )
        val_metrics, val_loss = val_report.metrics, val_report.avg_loss
        f1_macro = val_metrics.get("f1/macro", float("nan"))
        auroc_macro = val_metrics.get("auroc/macro", float("nan"))
        pr_auc_macro = val_metrics.get("pr_auc/macro", float("nan"))

        improved_f1 = np.isfinite(f1_macro) and f1_macro > best_f1
        improved_pr_auc = np.isfinite(pr_auc_macro) and pr_auc_macro > best_pr_auc

        if improved_f1:
            best_f1 = float(f1_macro)
            patience_left = optim.early_stop_patience
        else:
            patience_left -= 1
        if improved_pr_auc:
            best_pr_auc = float(pr_auc_macro)

        if ddp.is_main_process():
            if improved_f1:
                parts = [
                    f"f1={best_f1:.4f}",
                    f"pr_auc={pr_auc_macro:.4f}" if np.isfinite(pr_auc_macro) else "pr_auc=nan",
                    f"auroc={auroc_macro:.4f}" if np.isfinite(auroc_macro) else "auroc=nan",
                    f"val_loss={val_loss:.4f}",
                ]
                tqdm.write(f"best f1 val - epoch {epoch}: {', '.join(parts)}")
                save_checkpoint(
                    ckpt_dir / "best_f1_val.pt",
                    model=model,
                    epoch=epoch,
                    metric_dict=val_metrics,
                    class_names=class_names,
                    dataset_fingerprint=dataset_fingerprint,
                    symmetrize_edges=symmetrize_edges,
                    subgraph_representation=subgraph_representation,
                )
            if improved_pr_auc:
                save_checkpoint(
                    ckpt_dir / "best_pr_auc_val.pt",
                    model=model,
                    epoch=epoch,
                    metric_dict=val_metrics,
                    class_names=class_names,
                    dataset_fingerprint=dataset_fingerprint,
                    symmetrize_edges=symmetrize_edges,
                    subgraph_representation=subgraph_representation,
                )
            log_payload = {
                **base_metrics,
                "val/loss": val_loss,
                "patience_left": patience_left,
            }
            log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
            history.append(log_payload)

        scheduler.step(f1_macro if np.isfinite(f1_macro) else 0.0)

        postfix = {
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "f1": f"{f1_macro:.4f}" if np.isfinite(f1_macro) else "nan",
            "pr_auc": f"{pr_auc_macro:.4f}" if np.isfinite(pr_auc_macro) else "nan",
            "patience": patience_left,
        }
        epoch_iter.set_postfix(refresh=False, **postfix)

        if patience_left <= 0:
            if ddp.is_main_process():
                tqdm.write(
                    f"[train] early stop at epoch {epoch}; "
                    f"best f1/macro={best_f1:.4f}, best pr_auc/macro={best_pr_auc:.4f}"
                )
            break

    return TrainResult(
        best_f1_val=float(best_f1),
        best_pr_auc_val=float(best_pr_auc),
        epochs_completed=epoch,
        history=history,
    )


def write_history_json(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(history, f, indent=2)
