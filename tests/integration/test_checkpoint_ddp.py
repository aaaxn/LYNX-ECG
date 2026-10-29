"""Checkpoint behavior that requires a real distributed process group."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from training.checkpoint import unwrap_model

pytestmark = pytest.mark.integration


def test_unwrap_model_peels_compile_over_ddp():
    store = dist.TCPStore("127.0.0.1", 0, 1, is_master=True)
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        base = torch.nn.Linear(4, 2)
        wrapped = torch.compile(torch.nn.parallel.DistributedDataParallel(base))
        assert unwrap_model(wrapped) is base
    finally:
        dist.destroy_process_group()
