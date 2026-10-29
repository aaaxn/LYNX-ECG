"""Per-epoch positive/negative sampling for ECG training.

:class:`DynamicPosNegSampler` follows the protocol from *An Empirical
Comparison of ECG Foundation Models Across Modalities* (§3.3): every epoch
combines all positive exams with a fresh **without-replacement** sample of
``n_neg`` negatives. Epoch size = ``len(pos) + n_neg``. The sampler redraws on
``set_epoch``, so each epoch sees a fresh composition.

It is DDP-correct by "generate-then-shard": all ranks derive an identical
full-epoch index list from ``(seed, epoch)``, then slice
``[rank::num_replicas]``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from config.train_models import SamplingConfig


def _block_local_shuffle(
    indices: np.ndarray, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Locality-preserving shuffle of an arbitrary dataset-position list.

    Positions are grouped by ``position // block_size`` (a window of ``block_size`` dataset positions
    ≈ file locality, since split indices are sorted ascending), the group
    order and the order within each group are shuffled. Reads then stay
    inside a sliding ``block_size``-exam window the page cache can serve,
    instead of striding randomly across an HDF5 file that exceeds it, while
    SGD still sees a random order within each window.

    Returns a new array containing exactly the input positions.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    idx = np.sort(indices, kind="stable")
    if idx.size == 0:
        return idx
    bucket = idx // block_size
    starts = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    ends = np.r_[starts[1:], idx.size]
    out = np.empty_like(idx)
    write_pos = 0
    for b in rng.permutation(starts.size):
        seg = idx[starts[b] : ends[b]]
        out[write_pos : write_pos + seg.size] = rng.permutation(seg)
        write_pos += seg.size
    return out


def _pad_and_shard(epoch_indices: np.ndarray, num_replicas: int, rank: int) -> np.ndarray:
    """Pad ``epoch_indices`` to a multiple of ``num_replicas`` and return this rank's slice."""
    per_rank = int(np.ceil(epoch_indices.size / num_replicas))
    padded_size = per_rank * num_replicas
    padding_size = padded_size - epoch_indices.size
    if padding_size > 0:
        if padding_size <= epoch_indices.size:
            pad = epoch_indices[:padding_size]
        else:
            repeats = int(np.ceil(padding_size / epoch_indices.size))
            pad = np.tile(epoch_indices, repeats)[:padding_size]
        epoch_indices = np.concatenate([epoch_indices, pad])
    return epoch_indices[rank::num_replicas]


class DynamicPosNegSampler(Sampler[int]):
    """Yield positions in a training index array, resampling negatives per epoch.

    Args:
        pos_positions: Positions (into the Dataset's index space) of exams with
            at least one positive target label. Included in full every epoch.
        neg_positions: Positions of exams with no positive target labels (the
            "Undiagnosed" pool). A fresh ``n_neg``-sized subset is drawn each
            epoch without replacement.
        n_neg: Number of negatives to sample per epoch. Must satisfy
            ``n_neg <= len(neg_positions)``.
        seed: Base seed; combined with the epoch number to derive the per-epoch
            RNG. Same ``(seed, epoch)`` ⇒ same shuffled epoch index list on
            every process (DDP correctness).
        num_replicas: World size for DDP sharding (default 1 ⇒ no sharding).
        rank: Rank for DDP sharding (default 0).
        block_size: When set, the epoch index list is ordered by
            :func:`_block_local_shuffle` (locality-preserving windows of
            ``block_size`` dataset positions) instead of a fully random
            shuffle. Epoch *composition* (all positives + the fresh negative
            sample) is unchanged, only the visit order changes. ``None`` = fully
            random order.
    """

    def __init__(
        self,
        *,
        pos_positions: np.ndarray,
        neg_positions: np.ndarray,
        n_neg: int,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        block_size: int | None = None,
    ) -> None:
        self.pos_positions = np.ascontiguousarray(pos_positions, dtype=np.int64)
        self.neg_positions = np.ascontiguousarray(neg_positions, dtype=np.int64)
        self.n_neg = int(n_neg)
        if self.n_neg <= 0:
            raise ValueError(f"n_neg must be > 0, got {self.n_neg}")
        if self.n_neg > self.neg_positions.size:
            raise ValueError(
                f"n_neg ({self.n_neg}) exceeds negative pool size "
                f"({self.neg_positions.size}); cannot sample without replacement."
            )
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative for DynamicPosNegSampler, got {self.seed}")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {self.num_replicas}")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"rank must satisfy 0 <= rank < num_replicas, "
                f"got rank={self.rank}, num_replicas={self.num_replicas}"
            )
        self.block_size = int(block_size) if block_size is not None else None
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError(f"block_size must be > 0 or None, got {self.block_size}")
        self._epoch = 0
        self._epoch_indices: np.ndarray | None = None

    def set_epoch(self, epoch: int) -> None:
        """Recompute the epoch index list for ``epoch`` (called once per epoch)."""
        self._epoch = int(epoch)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self._epoch]))
        chosen_neg = rng.choice(self.neg_positions, size=self.n_neg, replace=False)
        epoch_indices = np.concatenate([self.pos_positions, chosen_neg])
        if self.block_size is None:
            rng.shuffle(epoch_indices)
        else:
            epoch_indices = _block_local_shuffle(epoch_indices, self.block_size, rng)

        self._epoch_indices = _pad_and_shard(epoch_indices, self.num_replicas, self.rank)

    def _ensure_built(self) -> np.ndarray:
        if self._epoch_indices is None:
            self.set_epoch(self._epoch)
        assert self._epoch_indices is not None
        return self._epoch_indices

    def __iter__(self) -> Iterator[int]:
        indices = self._ensure_built()
        return iter(int(i) for i in indices)

    def __len__(self) -> int:
        return int(np.ceil((self.pos_positions.size + self.n_neg) / self.num_replicas))


def build_train_sampler(
    *,
    train_idx: np.ndarray,
    exam_labels: np.ndarray,
    sampling: SamplingConfig,
    seed: int,
    num_replicas: int = 1,
    rank: int = 0,
    block_size: int | None = None,
) -> DynamicPosNegSampler:
    """Build the train sampler for the configured negative count.

    ``exam_labels`` is the full ``[E, K]`` multi-hot label array. ``train_idx``
    selects the training split, and the returned sampler yields positions
    relative to ``ExamGraphDataset(train_idx)``.

    ``block_size`` (``optim.shuffle_block_exams`` in presets) enables the
    locality-preserving visit order inside the returned sampler; ``None``
    keeps the fully random order.
    """
    train_idx = np.asarray(train_idx)
    label_sum = exam_labels[train_idx].sum(axis=1)
    pos_positions = np.where(label_sum > 0)[0].astype(np.int64)
    neg_positions = np.where(label_sum == 0)[0].astype(np.int64)

    return DynamicPosNegSampler(
        pos_positions=pos_positions,
        neg_positions=neg_positions,
        n_neg=sampling.n_neg,
        seed=seed,
        num_replicas=num_replicas,
        rank=rank,
        block_size=block_size,
    )
