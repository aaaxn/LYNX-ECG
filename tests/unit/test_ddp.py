"""Process-free unit tests for DDP helper fallbacks and wrappers."""

from __future__ import annotations

import contextlib

import pytest
import torch

from training import ddp

pytestmark = pytest.mark.unit


def test_all_reduce_sum_noops_when_not_distributed(monkeypatch):
    monkeypatch.setattr(ddp, "is_distributed", lambda: False)
    values = torch.tensor([2.0, 3.0])

    out = ddp.all_reduce_sum(values)

    assert out is values
    assert torch.equal(out, torch.tensor([2.0, 3.0]))


def test_all_reduce_sum_calls_distributed_sum(monkeypatch):
    monkeypatch.setattr(ddp, "is_distributed", lambda: True)
    calls = {}

    def fake_all_reduce(tensor, op):
        calls["op"] = op
        tensor.add_(torch.tensor([5.0, 7.0]))

    monkeypatch.setattr(ddp.dist, "all_reduce", fake_all_reduce)
    values = torch.tensor([2.0, 3.0])

    out = ddp.all_reduce_sum(values)

    assert out is values
    assert torch.equal(out, torch.tensor([7.0, 10.0]))
    assert calls["op"] == ddp.dist.ReduceOp.SUM


def test_wrap_model_noops_when_not_distributed(monkeypatch):
    monkeypatch.setattr(ddp, "is_distributed", lambda: False)
    model = torch.nn.Linear(2, 2)

    assert ddp.wrap_model(model) is model


def test_find_ddp_module_returns_none_for_plain_module():
    assert ddp.find_ddp_module(torch.nn.Linear(2, 2)) is None


def test_no_sync_context_is_nullcontext_without_ddp():
    model = torch.nn.Linear(2, 2)
    for sync in (True, False):
        ctx = ddp.no_sync_context(model, sync=sync)
        assert isinstance(ctx, contextlib.nullcontext)
