"""Unit tests for EvalShardSampler, the non-padding DDP eval shard.

Regression: DistributedSampler(drop_last=False) repeats
samples to equalise per-rank counts, which inflates all-gathered eval metrics
with duplicates. EvalShardSampler must partition the dataset exactly: every
index appears on exactly one rank, none are dropped, none duplicated.
"""

from __future__ import annotations

import pytest

from training.loaders import EvalShardSampler

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("dataset_len", "world_size"),
    [(0, 1), (1, 1), (5, 1), (7, 3), (8, 4), (10, 3), (100, 7), (2, 3)],
)
def test_partition_is_exact_and_duplicate_free(dataset_len, world_size):
    shards = [list(EvalShardSampler(dataset_len, world_size, rank)) for rank in range(world_size)]
    flat = [i for shard in shards for i in shard]

    assert sorted(flat) == list(range(dataset_len))
    assert len(flat) == len(set(flat)) == dataset_len

    seen: set[int] = set()
    for shard in shards:
        assert not (set(shard) & seen)
        seen.update(shard)

    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1

    for rank in range(world_size):
        assert len(EvalShardSampler(dataset_len, world_size, rank)) == sizes[rank]


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="num_replicas"):
        EvalShardSampler(10, 0, 0)
    with pytest.raises(ValueError, match="rank"):
        EvalShardSampler(10, 2, 2)
    with pytest.raises(ValueError, match="rank"):
        EvalShardSampler(10, 2, -1)
