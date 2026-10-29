"""Unit tests for the locality-preserving visit order in the pos/neg samplers.

``_block_local_shuffle`` reorders an epoch index list so reads stay inside a
sliding window of ``block_size`` dataset positions (page-cache friendly on
HDF5 files larger than RAM) without changing the epoch *composition*. The
dynamic sampler exposes it via ``block_size``; ``None`` keeps the fully
random order.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.train_models import SamplingConfig
from training.sampling import (
    DynamicPosNegSampler,
    _block_local_shuffle,
    build_train_sampler,
)

pytestmark = pytest.mark.unit


def test_block_local_shuffle_preserves_multiset():
    rng = np.random.default_rng(0)
    indices = np.array([5, 5, 900, 3, 120, 120, 77], dtype=np.int64)

    out = _block_local_shuffle(indices, block_size=100, rng=rng)

    assert sorted(out.tolist()) == sorted(indices.tolist())


def test_block_local_shuffle_groups_each_block_contiguously():
    """Every ``position // block_size`` bucket must occupy one contiguous run."""
    rng = np.random.default_rng(1)
    indices = np.arange(0, 1000, 3, dtype=np.int64)
    block_size = 100

    out = _block_local_shuffle(indices, block_size=block_size, rng=rng)

    buckets = out // block_size
    change_points = np.flatnonzero(np.r_[True, buckets[1:] != buckets[:-1]])
    assert len(change_points) == np.unique(buckets).size


def test_block_local_shuffle_shuffles_within_and_across_blocks():
    rng = np.random.default_rng(2)
    indices = np.arange(0, 500, dtype=np.int64)

    out = _block_local_shuffle(indices, block_size=100, rng=rng)

    assert not np.array_equal(out, indices)
    first_block = out[:100]
    assert not np.array_equal(np.sort(first_block), first_block)


def test_block_local_shuffle_empty_and_validation():
    rng = np.random.default_rng(3)
    empty = np.empty(0, dtype=np.int64)

    assert _block_local_shuffle(empty, block_size=10, rng=rng).size == 0
    with pytest.raises(ValueError, match="block_size"):
        _block_local_shuffle(np.array([1]), block_size=0, rng=rng)


def _dyn(block_size: int | None, seed: int = 42) -> DynamicPosNegSampler:
    pos = np.arange(0, 50, dtype=np.int64)
    neg = np.arange(50, 2050, dtype=np.int64)
    return DynamicPosNegSampler(
        pos_positions=pos, neg_positions=neg, n_neg=400, seed=seed, block_size=block_size
    )


def test_dynamic_block_size_changes_order_not_composition():
    """Same (seed, epoch): identical epoch *set*, different visit order."""
    plain = _dyn(block_size=None)
    local = _dyn(block_size=128)
    plain.set_epoch(5)
    local.set_epoch(5)

    plain_indices = np.asarray(list(plain))
    local_indices = np.asarray(list(local))

    assert sorted(plain_indices.tolist()) == sorted(local_indices.tolist())
    assert not np.array_equal(plain_indices, local_indices)


def test_dynamic_block_size_is_deterministic():
    a, b = _dyn(block_size=128), _dyn(block_size=128)
    a.set_epoch(3)
    b.set_epoch(3)

    assert list(a) == list(b)


def test_dynamic_block_size_bounds_position_jumps_within_blocks():
    """Within one contiguous block run, positions span less than block_size."""
    sampler = _dyn(block_size=128)
    sampler.set_epoch(0)
    out = np.asarray(list(sampler))

    buckets = out // 128
    starts = np.flatnonzero(np.r_[True, buckets[1:] != buckets[:-1]])
    ends = np.r_[starts[1:], out.size]
    for lo, hi in zip(starts, ends, strict=True):
        seg = out[lo:hi]
        assert seg.max() - seg.min() < 128


def test_dynamic_block_size_ddp_shards_stay_disjoint_and_complete():
    pos = np.arange(0, 50, dtype=np.int64)
    neg = np.arange(50, 2050, dtype=np.int64)
    shards = []
    for rank in range(2):
        s = DynamicPosNegSampler(
            pos_positions=pos,
            neg_positions=neg,
            n_neg=400,
            seed=7,
            num_replicas=2,
            rank=rank,
            block_size=128,
        )
        s.set_epoch(1)
        shards.append(list(s))

    assert len(shards[0]) == len(shards[1]) == 225
    combined = sorted(shards[0] + shards[1])
    assert len(set(combined)) == 450


def test_dynamic_rejects_non_positive_block_size():
    with pytest.raises(ValueError, match="block_size"):
        _dyn(block_size=-1)


def test_build_train_sampler_forwards_block_size():
    exam_labels = np.zeros((300, 2), dtype=np.uint8)
    exam_labels[:30, 0] = 1
    train_idx = np.arange(300)

    dyn = build_train_sampler(
        train_idx=train_idx,
        exam_labels=exam_labels,
        sampling=SamplingConfig(n_neg=100),
        seed=1,
        block_size=64,
    )
    assert dyn.block_size == 64
