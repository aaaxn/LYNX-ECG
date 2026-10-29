"""Lightweight DDP utilities.

Wraps ``torch.distributed`` so the train / evaluate scripts can be DDP-aware
without scattering ``init_process_group`` calls everywhere. Single-process
fallbacks make the same code path work for ``--launcher single``.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return rank() == 0


def init_distributed(backend: str | None = None) -> bool:
    """Initialize the default process group from torchrun env vars.

    Returns ``True`` if a distributed group was initialized, ``False`` if the
    env vars indicate single-process execution. Safe to call when not under
    torchrun, where it will short-circuit.

    Picks ``nccl`` when CUDA is available and the user did not override,
    otherwise ``gloo`` (CPU).
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    if dist.is_available() and dist.is_initialized():
        return True
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=10))
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    return True


def shutdown_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def wrap_model(model: torch.nn.Module, *, static_graph: bool = True) -> torch.nn.Module:
    """Wrap ``model`` in DDP on the current CUDA device; no-op single-process.

    The model must already be on its target device (call ``.to(device)``
    first). DDP broadcasts the wrapped parameters from rank 0 as-is.
    ``find_unused_parameters`` stays ``False``: the model uses every parameter
    on every step.

    Reducer tuning applied here:

    - ``gradient_as_bucket_view=True``: gradients alias the communication
      buckets directly instead of being copied into them each backward.
      Semantics are unchanged for our optimizer path (fused/foreach AdamW and
      ``clip_grad_norm_`` both operate on ``param.grad`` in place).
    - ``static_graph=True`` (default): a static reducer requires the set of
      parameters receiving gradients to be identical on every iteration, which
      is exactly the contract ``find_unused_parameters=False`` already imposes
      (a parameter that skips one backward errors out either way), so this
      adds no new failure mode while letting DDP skip per-step graph
      bookkeeping. Callers that ``torch.compile`` the wrapped model pass
      ``static_graph=False``: dynamo's DDPOptimizer path is developed and
      tested against the default (non-static) reducer, and the static
      reducer's bookkeeping wins are subsumed by compilation anyway.
    """
    if not is_distributed():
        return model
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if local_rank is not None else None,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=static_graph,
    )


def wrap_and_compile_model(
    base_model: torch.nn.Module,
    *,
    use_compile: bool,
) -> torch.nn.Module:
    """Wrap ``base_model`` in DDP via :func:`wrap_model`, then optionally
    ``torch.compile`` it, so the invariant below holds in one place.

    Ordering invariant (do not reorder): compile wraps OUTSIDE DDP. Dynamo's
    ``DDPOptimizer`` splits the compiled graph at gradient-bucket boundaries
    to keep allreduce overlapped with backward, which requires DDP to be the
    innermost wrapper compile sees. Accordingly ``static_graph`` is forced
    off (via :func:`wrap_model`'s ``static_graph=False`` when
    ``use_compile=True``): dynamo's ``DDPOptimizer`` path is developed and
    tested against the default (non-static) reducer, and the static reducer's
    bookkeeping wins are subsumed by compilation anyway.
    """
    model = wrap_model(base_model, static_graph=not use_compile)
    if use_compile:
        model = torch.compile(model, dynamic=True)
    return model


def find_ddp_module(
    model: torch.nn.Module,
) -> torch.nn.parallel.DistributedDataParallel | None:
    """Return the :class:`DistributedDataParallel` wrapper governing ``model``'s
    gradient synchronization, or ``None`` when ``model`` is not DDP-wrapped.

    This repo compiles OUTSIDE DDP, as ``torch.compile(DistributedDataParallel(m))``
    (see :func:`wrap_and_compile_model`), so under compilation the DDP instance
    is not ``model`` itself but the ``OptimizedModule``'s wrapped original,
    exposed as ``model._orig_mod``. Checking both covers the compiled and
    uncompiled cases; a plain single-process module matches neither and yields
    ``None`` (so callers fall back to a no-op sync context).
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model
    orig = getattr(model, "_orig_mod", None)
    if isinstance(orig, torch.nn.parallel.DistributedDataParallel):
        return orig
    return None


def no_sync_context(model: torch.nn.Module, *, sync: bool) -> AbstractContextManager[None]:
    """Context manager selecting whether a micro-batch backward synchronizes
    gradients across ranks.

    During example-weighted gradient accumulation only the FINAL micro-batch of
    a window should trigger the DDP all-reduce, which reduces the whole
    accumulated ``.grad``, not just that micro-batch's contribution, while the
    earlier micro-batches accumulate gradients locally without communicating.
    Pass ``sync=False`` for every non-final micro-batch to get
    :meth:`DistributedDataParallel.no_sync`, and ``sync=True`` for the final one
    (an all-reduce-enabling :func:`~contextlib.nullcontext`).

    On a single-process run, or any time ``model`` is not DDP-wrapped, this is
    always a no-op ``nullcontext`` so the accumulation loop needs no
    ``if distributed`` branch. The DDP instance is located with
    :func:`find_ddp_module`, which unwraps the compile-outside-DDP ordering.
    """
    if sync:
        return nullcontext()
    ddp_module = find_ddp_module(model)
    if ddp_module is None:
        return nullcontext()
    return ddp_module.no_sync()


def barrier() -> None:
    """Block until all ranks reach this point; no-op when single-process.

    Use before reading artifacts that another rank wrote (e.g. ``best_f1_val.pt``
    written by rank 0 only). Without it the non-writing ranks may race past
    the read and trigger a ``FileNotFoundError`` or, worse, hang on the next
    collective because rank 0 has exited.
    """
    if is_distributed():
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
        dist.barrier(device_ids=[device_id] if device_id is not None else None)


def all_gather_concat(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a tensor across ranks and concatenate along dim 0.

    On single-process runs returns the input unchanged. The contract is that
    all ranks pass tensors with identical shapes EXCEPT possibly dim 0.
    """
    if not is_distributed():
        return tensor.detach().cpu()

    ws = world_size()
    sizes = torch.zeros(ws, dtype=torch.long, device=tensor.device)
    sizes[rank()] = tensor.size(0)
    dist.all_reduce(sizes, op=dist.ReduceOp.SUM)
    max_size = int(sizes.max().item())

    pad = max_size - tensor.size(0)
    if pad > 0:
        padding_shape = (pad, *tensor.shape[1:])
        tensor = torch.cat([tensor, tensor.new_zeros(padding_shape)], dim=0)

    gathered = [torch.empty_like(tensor) for _ in range(ws)]
    dist.all_gather(gathered, tensor)

    chunks: list[torch.Tensor] = []
    for r in range(ws):
        n = int(sizes[r].item())
        chunks.append(gathered[r][:n])
    return torch.cat(chunks, dim=0).detach().cpu()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum ``tensor`` across ranks in-place; no-op for single-process runs."""
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
