"""DDP contracts that require real Gloo process groups."""

from __future__ import annotations

import multiprocessing
import queue
from datetime import timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest
import torch
import torch.distributed as dist

from training import ddp

pytestmark = pytest.mark.integration

_PROCESS_TIMEOUT = 15


def _real_gloo_scenario(rank, events):
    assert ddp.is_distributed(), f"rank {rank}: process group is not initialized"
    assert ddp.rank() == rank, f"rank {rank}: helper returned rank {ddp.rank()}"
    assert ddp.world_size() == 2, f"rank {rank}: helper returned world size {ddp.world_size()}"

    uneven_input = (
        torch.tensor([[10, 11], [12, 13]], dtype=torch.int64)
        if rank == 0
        else torch.tensor([[20, 21]], dtype=torch.int64)
    )
    uneven = ddp.all_gather_concat(uneven_input)
    expected_uneven = torch.tensor([[10, 11], [12, 13], [20, 21]], dtype=torch.int64)
    assert uneven.device.type == "cpu", f"rank {rank}: uneven gather did not return on CPU"
    assert uneven.shape == (3, 2), f"rank {rank}: uneven gather shape was {uneven.shape}"
    assert torch.equal(uneven, expected_uneven), (
        f"rank {rank}: uneven gather was not rank ordered or retained padding"
    )

    empty_input = (
        torch.empty((0, 2), dtype=torch.int64)
        if rank == 0
        else torch.tensor([[30, 31], [32, 33]], dtype=torch.int64)
    )
    one_empty = ddp.all_gather_concat(empty_input)
    expected_nonempty = torch.tensor([[30, 31], [32, 33]], dtype=torch.int64)
    assert one_empty.device.type == "cpu", f"rank {rank}: zero-row gather did not return on CPU"
    assert one_empty.shape == (2, 2), f"rank {rank}: zero-row gather shape was {one_empty.shape}"
    assert torch.equal(one_empty, expected_nonempty), (
        f"rank {rank}: zero-row gather introduced synthetic values"
    )

    release_rank_one, rank_zero_entered, rank_zero_returned = events
    if rank == 0:
        rank_zero_entered.set()
        ddp.barrier()
        rank_zero_returned.set()
    else:
        assert release_rank_one.wait(timeout=_PROCESS_TIMEOUT), (
            "rank 1: parent did not release barrier participation"
        )
        ddp.barrier()

    return {
        "rank": ddp.rank(),
        "world_size": ddp.world_size(),
        "uneven": uneven.tolist(),
        "one_empty": one_empty.tolist(),
        "barrier_completed": True,
    }


def _distributed_child(rank, rendezvous_uri, scenario, result_queue, events):
    initialized = False
    try:
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=rendezvous_uri,
                rank=rank,
                world_size=2,
                timeout=timedelta(seconds=10),
            )
            initialized = True
            with patch.object(torch.cuda, "is_available", return_value=False):
                payload = scenario(rank, events)
        finally:
            if initialized:
                ddp.shutdown_distributed()

        lifecycle = {
            "is_distributed": ddp.is_distributed(),
            "rank": ddp.rank(),
            "world_size": ddp.world_size(),
        }
        assert lifecycle == {"is_distributed": False, "rank": 0, "world_size": 1}, (
            f"rank {rank}: fallback state after shutdown was {lifecycle}"
        )
        result_queue.put(("success", rank, payload, lifecycle))
    except BaseException as error:
        result_queue.put(("error", rank, type(error).__name__, str(error)))
        raise


def _run_two_rank_gloo(tmp_path, scenario, events=(), during_run=None):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    rendezvous_path = tmp_path / f"gloo-{uuid4().hex}"
    assert not rendezvous_path.exists()
    rendezvous_uri = rendezvous_path.resolve().as_uri()
    processes = [
        context.Process(
            target=_distributed_child,
            args=(rank, rendezvous_uri, scenario, result_queue, events),
        )
        for rank in range(2)
    ]
    reports = []
    try:
        for process in processes:
            process.start()

        if during_run is not None:
            during_run()
        for _ in processes:
            try:
                reports.append(result_queue.get(timeout=_PROCESS_TIMEOUT))
            except queue.Empty:
                pytest.fail("timed out waiting for a Gloo child report")
        for process in processes:
            process.join(timeout=_PROCESS_TIMEOUT)

        errors = [report for report in reports if report[0] == "error"]
        assert not errors, f"Gloo child failures: {errors}"
        assert {report[1] for report in reports} == {0, 1}
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
        return {report[1]: report[2:] for report in reports}
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=_PROCESS_TIMEOUT)
                assert not process.is_alive(), f"Gloo child {process.pid} survived cleanup"
        result_queue.cancel_join_thread()
        result_queue.close()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_real_gloo_gathers_barrier_and_lifecycle(tmp_path):
    context = multiprocessing.get_context("spawn")
    release_rank_one = context.Event()
    rank_zero_entered = context.Event()
    rank_zero_returned = context.Event()

    def observe_barrier():
        assert rank_zero_entered.wait(timeout=_PROCESS_TIMEOUT), "rank 0 did not enter barrier"
        assert not rank_zero_returned.wait(timeout=0.5), (
            "rank 0 returned before rank 1 participated"
        )
        release_rank_one.set()
        assert rank_zero_returned.wait(timeout=_PROCESS_TIMEOUT), (
            "rank 0 did not return after rank 1 participated"
        )

    reports = _run_two_rank_gloo(
        tmp_path,
        _real_gloo_scenario,
        events=(release_rank_one, rank_zero_entered, rank_zero_returned),
        during_run=observe_barrier,
    )

    rank_zero, rank_one = reports[0][0], reports[1][0]
    assert rank_zero["rank"] == 0, f"rank 0 payload reported rank {rank_zero['rank']}"
    assert rank_one["rank"] == 1, f"rank 1 payload reported rank {rank_one['rank']}"
    assert rank_zero["world_size"] == rank_one["world_size"] == 2, (
        "real Gloo payloads did not report a two-rank world"
    )
    assert (
        rank_zero["uneven"]
        == rank_one["uneven"]
        == [
            [10, 11],
            [12, 13],
            [20, 21],
        ]
    ), "uneven gather payloads were not rank ordered"
    assert (
        rank_zero["one_empty"]
        == rank_one["one_empty"]
        == [
            [30, 31],
            [32, 33],
        ]
    ), "zero-row gather payloads did not retain only real rows"
    assert rank_zero["barrier_completed"] is True, "rank 0 did not complete the barrier"
    assert rank_one["barrier_completed"] is True, "rank 1 did not complete the barrier"

    expected_lifecycle = {"is_distributed": False, "rank": 0, "world_size": 1}
    for rank, (_, lifecycle) in reports.items():
        assert lifecycle == expected_lifecycle, (
            f"rank {rank}: fallback state after shutdown was {lifecycle}"
        )


class _FakeCompiled:
    """Stand-in for ``torch.compile``'s ``OptimizedModule``: the compiled wrapper
    exposes its wrapped (inner) module as ``_orig_mod``. This repo compiles
    OUTSIDE DDP, so under compilation the DDP instance lives here."""

    def __init__(self, inner):
        self._orig_mod = inner


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_find_ddp_module_unwraps_ddp_and_compile(tmp_path):
    """DDP discovery must see through the compile-outside-DDP wrapper ordering,
    and no_sync_context must gate the reducer only on non-final micro-batches."""
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        base = torch.nn.Linear(2, 2)
        ddp_model = torch.nn.parallel.DistributedDataParallel(base)

        assert ddp.find_ddp_module(ddp_model) is ddp_model
        assert ddp.find_ddp_module(_FakeCompiled(ddp_model)) is ddp_model
        assert ddp.find_ddp_module(_FakeCompiled(base)) is None

        assert ddp_model.require_backward_grad_sync is True
        with ddp.no_sync_context(ddp_model, sync=True):
            assert ddp_model.require_backward_grad_sync is True

        for candidate in (ddp_model, _FakeCompiled(ddp_model)):
            with ddp.no_sync_context(candidate, sync=False):
                assert ddp_model.require_backward_grad_sync is False
            assert ddp_model.require_backward_grad_sync is True
    finally:
        dist.destroy_process_group()
