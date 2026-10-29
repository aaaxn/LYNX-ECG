"""Data-loader construction for the training CLI."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader as PyGDataLoader

from training import ddp
from training.dataset import ExamGraphDataset, GraphIndexArrays
from training.sampling import build_train_sampler

if TYPE_CHECKING:
    from config.train_models import SamplingConfig, SubgraphRepresentationConfig


class EvalShardSampler(Sampler[int]):
    """Partition dataset indices across DDP ranks for evaluation WITHOUT padding.

    PyTorch's ``DistributedSampler(drop_last=False)`` repeats samples so every
    rank receives an equal count; :func:`training.inference.gather_predictions`
    then all-gathers those repeats, inflating val/test metrics (and the gathered
    loss denominator) with up to ``world_size - 1`` duplicates. Evaluation does
    not shuffle and the metrics are order-independent, so each rank instead takes
    a strided, non-overlapping slice of ``range(len(dataset))``. The union across
    ranks is exactly the dataset, with every sample scored exactly once, and
    :func:`training.ddp.all_gather_concat` already handles the resulting unequal
    per-rank counts.
    """

    def __init__(self, dataset_len: int, num_replicas: int, rank: int) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not (0 <= rank < num_replicas):
            raise ValueError(f"rank {rank} out of range for num_replicas {num_replicas}")
        self._indices = list(range(rank, dataset_len, num_replicas))

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def select_device(choice: str) -> torch.device:
    """Return a ``torch.device`` from the user's CLI choice (``"auto"``/``"cuda"``/``"cpu"``)."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def build_datasets(
    *,
    dataset_path: Path,
    num_leads: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    test_dataset_path: Path | None = None,
    symmetrize_edges: bool = False,
    subgraph_representation: SubgraphRepresentationConfig | None = None,
) -> tuple[ExamGraphDataset, ExamGraphDataset, ExamGraphDataset]:
    """Construct the train/val/test datasets once.

    Each ``ExamGraphDataset`` loads the per-graph index arrays into RAM, so for
    large datasets this is expensive; build them once and pass the result to
    :func:`build_loaders_from_datasets`.

    ``symmetrize_edges`` is forwarded identically to all three datasets. Train
    and eval graphs must match, or metrics would be computed on a differently
    shaped receptive field than the model was trained with.

    ``subgraph_representation`` is likewise forwarded as one immutable value
    to train, validation, and test, including a separately stored test HDF5.
    """
    arrays = GraphIndexArrays.from_hdf5(dataset_path)
    train_ds = ExamGraphDataset(
        dataset_path,
        np.asarray(train_idx),
        num_leads=num_leads,
        arrays=arrays,
        symmetrize_edges=symmetrize_edges,
        subgraph_representation=subgraph_representation,
    )
    val_ds = ExamGraphDataset(
        dataset_path,
        np.asarray(val_idx),
        num_leads=num_leads,
        arrays=arrays,
        symmetrize_edges=symmetrize_edges,
        subgraph_representation=subgraph_representation,
    )
    test_path = test_dataset_path if test_dataset_path is not None else dataset_path
    test_arrays = arrays if test_path == dataset_path else GraphIndexArrays.from_hdf5(test_path)
    test_ds = ExamGraphDataset(
        test_path,
        np.asarray(test_idx),
        num_leads=num_leads,
        arrays=test_arrays,
        symmetrize_edges=symmetrize_edges,
        subgraph_representation=subgraph_representation,
    )
    return train_ds, val_ds, test_ds


def build_loaders_from_datasets(
    *,
    train_ds: ExamGraphDataset,
    val_ds: ExamGraphDataset,
    test_ds: ExamGraphDataset,
    train_idx: np.ndarray,
    exam_labels: np.ndarray,
    sampling: SamplingConfig,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    seed: int,
    device: torch.device,
    shuffle_block_exams: int | None = None,
    eval_batch_size: int | None = None,
) -> tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """Build train/val/test ``PyGDataLoader``s over pre-built datasets.

    Handles the dynamic positive/negative train sampler, DDP sharding, and
    CUDA input-pipeline options (pinned memory, persistent workers).
    ``shuffle_block_exams`` sets the locality-preserving read window of the
    train sampler's visit order (``None`` = fully random order).

    ``eval_batch_size`` sizes the val/test loaders independently of the train
    ``batch_size``. Evaluation runs without backward, so a larger batch fits
    the same VRAM and cuts eval wall-clock. ``None`` = same as ``batch_size``.

    ``train_idx`` is the global exam-index array for the train split; it is used
    only to derive per-exam sampler weights from ``exam_labels`` (the dataset
    itself indexes by position). See :func:`build_loaders` for the all-in-one
    entry point.
    """
    train_idx = np.asarray(train_idx)

    common_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        common_kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    train_kwargs = {"batch_size": batch_size, **common_kwargs}
    eval_kwargs = {
        "batch_size": eval_batch_size if eval_batch_size is not None else batch_size,
        **common_kwargs,
    }

    train_sampler = build_train_sampler(
        train_idx=train_idx,
        exam_labels=exam_labels,
        sampling=sampling,
        seed=seed,
        num_replicas=ddp.world_size() if distributed else 1,
        rank=ddp.rank() if distributed else 0,
        block_size=shuffle_block_exams,
    )
    train_loader = PyGDataLoader(train_ds, sampler=train_sampler, **train_kwargs)

    if distributed:
        ws, rk = ddp.world_size(), ddp.rank()
        val_sampler = EvalShardSampler(len(val_ds), ws, rk)
        test_sampler = EvalShardSampler(len(test_ds), ws, rk)
        val_loader = PyGDataLoader(val_ds, sampler=val_sampler, **eval_kwargs)
        test_loader = PyGDataLoader(test_ds, sampler=test_sampler, **eval_kwargs)
    else:
        val_loader = PyGDataLoader(val_ds, shuffle=False, **eval_kwargs)
        test_loader = PyGDataLoader(test_ds, shuffle=False, **eval_kwargs)
    return train_loader, val_loader, test_loader


def build_loaders(
    *,
    dataset_path: Path,
    num_leads: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    exam_labels: np.ndarray,
    sampling: SamplingConfig,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    seed: int,
    device: torch.device,
    test_dataset_path: Path | None = None,
    shuffle_block_exams: int | None = None,
    eval_batch_size: int | None = None,
    symmetrize_edges: bool = False,
    subgraph_representation: SubgraphRepresentationConfig | None = None,
) -> tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
    """Build datasets and their loaders in one call (datasets built fresh).

    Convenience wrapper over :func:`build_datasets` + :func:`build_loaders_from_datasets`
    for callers that build loaders exactly once. Callers that rebuild loaders
    repeatedly over the same data should call the two pieces directly so the
    datasets (which load index arrays into RAM) are built only once.
    """
    train_ds, val_ds, test_ds = build_datasets(
        dataset_path=dataset_path,
        num_leads=num_leads,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        test_dataset_path=test_dataset_path,
        symmetrize_edges=symmetrize_edges,
        subgraph_representation=subgraph_representation,
    )
    return build_loaders_from_datasets(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_idx=train_idx,
        exam_labels=exam_labels,
        sampling=sampling,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        seed=seed,
        device=device,
        shuffle_block_exams=shuffle_block_exams,
        eval_batch_size=eval_batch_size,
    )
