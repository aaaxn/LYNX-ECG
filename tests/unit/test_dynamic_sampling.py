"""Unit tests for ``training.sampling.DynamicPosNegSampler``.

The sampler implements the paper-spec dynamic-sampling protocol: every epoch
yields all positives plus ``n_neg`` negatives sampled without replacement from
the negative pool, with a per-epoch RNG that is identical across DDP ranks for
the same ``(seed, epoch)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.sampling import DynamicPosNegSampler


@pytest.mark.unit
def test_same_seed_and_epoch_yield_identical_indices():
    """Cross-rank determinism: two samplers with same (seed, epoch) match exactly."""
    pos = np.arange(0, 20, dtype=np.int64)
    neg = np.arange(20, 220, dtype=np.int64)

    sampler_a = DynamicPosNegSampler(pos_positions=pos, neg_positions=neg, n_neg=50, seed=42)
    sampler_b = DynamicPosNegSampler(pos_positions=pos, neg_positions=neg, n_neg=50, seed=42)

    sampler_a.set_epoch(3)
    sampler_b.set_epoch(3)

    indices_a = list(sampler_a)
    indices_b = list(sampler_b)

    assert indices_a == indices_b
    assert len(indices_a) == len(sampler_a) == 70


@pytest.mark.unit
def test_all_positives_included_and_negatives_without_replacement():
    """Every positive position appears each epoch; negatives have no duplicates."""
    pos = np.arange(0, 20, dtype=np.int64)
    neg = np.arange(20, 220, dtype=np.int64)

    sampler = DynamicPosNegSampler(pos_positions=pos, neg_positions=neg, n_neg=50, seed=42)
    sampler.set_epoch(0)
    indices = np.asarray(list(sampler), dtype=np.int64)

    assert indices.size == 70
    assert set(pos.tolist()).issubset(indices.tolist())
    chosen_neg = indices[~np.isin(indices, pos)]
    assert chosen_neg.size == 50
    assert np.unique(chosen_neg).size == 50
    assert np.all(np.isin(chosen_neg, neg))


@pytest.mark.unit
def test_different_epochs_draw_different_negative_subsets():
    """``set_epoch(0)`` vs ``set_epoch(1)`` must change the negative composition."""
    pos = np.arange(0, 20, dtype=np.int64)
    neg = np.arange(20, 1020, dtype=np.int64)

    sampler = DynamicPosNegSampler(pos_positions=pos, neg_positions=neg, n_neg=100, seed=42)

    sampler.set_epoch(0)
    negs_e0 = set(int(i) for i in sampler if i >= 20)

    sampler.set_epoch(1)
    negs_e1 = set(int(i) for i in sampler if i >= 20)

    assert negs_e0 != negs_e1
    assert len(negs_e0) == 100 and len(negs_e1) == 100


@pytest.mark.unit
def test_ddp_sharding_partitions_epoch_indices_across_ranks():
    """With world_size=2, rank-0 and rank-1 slices together cover the epoch set.

    The full epoch index list is built identically on every rank from
    ``(seed, epoch)``; each rank then takes its rank-strided slice. Padding (if
    any) replays elements from the head so per-rank length is exactly
    ``ceil(N / num_replicas)``. After de-duplicating any padding, the union of
    the two ranks' slices must equal the full epoch list.
    """
    pos = np.arange(0, 20, dtype=np.int64)
    neg = np.arange(20, 220, dtype=np.int64)
    n_neg = 51

    rank0 = DynamicPosNegSampler(
        pos_positions=pos,
        neg_positions=neg,
        n_neg=n_neg,
        seed=42,
        num_replicas=2,
        rank=0,
    )
    rank1 = DynamicPosNegSampler(
        pos_positions=pos,
        neg_positions=neg,
        n_neg=n_neg,
        seed=42,
        num_replicas=2,
        rank=1,
    )
    full = DynamicPosNegSampler(pos_positions=pos, neg_positions=neg, n_neg=n_neg, seed=42)
    rank0.set_epoch(5)
    rank1.set_epoch(5)
    full.set_epoch(5)

    s0 = list(rank0)
    s1 = list(rank1)
    full_indices = list(full)

    assert len(s0) == len(s1) == 36
    assert len(rank0) == len(rank1) == 36
    assert set(s0).issubset(set(full_indices))
    assert set(s1).issubset(set(full_indices))
    assert set(s0) | set(s1) == set(full_indices)


@pytest.mark.unit
def test_ddp_padding_tiles_when_epoch_smaller_than_world_size():
    """When the sampled epoch is smaller than ``num_replicas``, padding must tile.

    Otherwise some ranks receive zero indices and hang on the next DDP collective
    while other ranks attempt a backward pass. Mirrors
    ``torch.utils.data.DistributedSampler``'s repeat-padding behavior.
    """
    pos = np.array([], dtype=np.int64)
    neg = np.arange(0, 100, dtype=np.int64)

    samplers = [
        DynamicPosNegSampler(
            pos_positions=pos,
            neg_positions=neg,
            n_neg=1,
            seed=42,
            num_replicas=4,
            rank=r,
        )
        for r in range(4)
    ]
    for s in samplers:
        s.set_epoch(0)

    slices = [list(s) for s in samplers]
    assert all(len(sl) == 1 for sl in slices), f"per-rank lengths: {[len(sl) for sl in slices]}"
    assert all(len(s) == 1 for s in samplers)
    assert all(sl[0] == slices[0][0] for sl in slices)


@pytest.mark.unit
def test_constructor_rejects_n_neg_above_pool_size():
    """Misconfiguration must fail loudly at sampler construction, not mid-epoch."""
    pos = np.arange(0, 10, dtype=np.int64)
    neg = np.arange(10, 60, dtype=np.int64)

    with pytest.raises(ValueError, match="n_neg"):
        DynamicPosNegSampler(pos_positions=pos, neg_positions=neg, n_neg=51, seed=42)


@pytest.mark.unit
def test_constructor_rejects_n_neg_zero_and_negative():
    """Sampler must enforce ``n_neg > 0`` to match the config's ``Field(gt=0)``."""
    pos = np.arange(0, 10, dtype=np.int64)
    neg = np.arange(10, 60, dtype=np.int64)

    for bad in (0, -1, -100):
        with pytest.raises(ValueError, match="n_neg"):
            DynamicPosNegSampler(pos_positions=pos, neg_positions=neg, n_neg=bad, seed=42)


@pytest.mark.unit
def test_constructor_rejects_negative_seed():
    """``np.random.SeedSequence`` requires non-negative entropy, reject early."""
    pos = np.arange(0, 10, dtype=np.int64)
    neg = np.arange(10, 60, dtype=np.int64)

    with pytest.raises(ValueError, match="seed"):
        DynamicPosNegSampler(pos_positions=pos, neg_positions=neg, n_neg=5, seed=-1)
