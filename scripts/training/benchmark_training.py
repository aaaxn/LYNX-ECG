"""Benchmark the training stack on a real dataset: step, loader, e2e, eval.

Measures, for a given training preset (all sections optional):

- ``loader``. DataLoader-only throughput with the preset's real train
  sampler (exams/s, time-to-first-batch). This is the section sensitive to
  HDF5 read locality (page cache vs random IOPS).
- ``step``. Pure-GPU training-step time (forward + backward + optimizer)
  over a small pool of pre-fetched batches cycled in place, per batch size.
  Isolates compute from IO. Reports ms/step, exams/s, peak VRAM.
- ``e2e``. End-to-end steps consuming the real loader (data wait
  included), per batch size. This is the number that predicts epoch time.
- ``eval``. Inference-only throughput over the val split per eval batch
  size (no backward, ``inference_mode`` + bf16 autocast like the real eval).
- ``parity``. Eager vs ``torch.compile`` logits comparison (fp32 eval mode,
  and bf16-autocast) on a few val batches. Only with ``--compile``.

Results go to ``--out`` as JSON (env + git metadata included) and a summary
table is printed. Sections run in a fixed order with fixed sampler epochs so
two invocations of the same script version are directly comparable.

Usage:
    uv run python scripts/training/benchmark_training.py \
        --config configs/train/code100_tp.json \
        --batch-sizes 128 --out reports/bench_baseline.json

Page-cache caveat: the ``loader`` section depends on how much of the HDF5 is
already resident in the OS page cache. The script records ``Cached:`` from
``/proc/meminfo`` before and after so before/after comparisons can be judged
honestly; run back-to-back under similar system load.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader

from config.train_models import TrainPreset, load_train_preset
from training.hdf5_meta import read_hdf5_meta
from training.inference import batch_to_forward_kwargs, bf16_autocast, forward_batch
from training.loaders import build_datasets, build_loaders_from_datasets
from training.splits import resolve_splits
from training.train import build_optimizer, configure_matmul_precision, run_training_step

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="Training preset JSON.")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="32",
        help="Comma-separated train batch sizes for the step/e2e sections.",
    )
    parser.add_argument(
        "--eval-batch-sizes",
        type=str,
        default="32,128,256,512",
        help="Comma-separated batch sizes for the eval section.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps before timing.")
    parser.add_argument("--steps", type=int, default=60, help="Timed steps (step section).")
    parser.add_argument("--e2e-steps", type=int, default=100, help="Timed steps (e2e section).")
    parser.add_argument(
        "--loader-batches", type=int, default=300, help="Batches timed in the loader section."
    )
    parser.add_argument(
        "--eval-batches", type=int, default=40, help="Batches timed per eval batch size."
    )
    parser.add_argument(
        "--sections",
        type=str,
        default="loader,step,e2e,eval",
        help="Comma-separated sections to run (loader,step,e2e,eval,parity).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(dynamic=True) the model for step/e2e (and enable parity).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Override preset.optim.num_workers."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "<unknown>"


def _meminfo_cached_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("Cached:"):
                return float(line.split()[1]) / 1024**2
    except OSError:
        pass
    return float("nan")


def _env_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "git_sha": _git_sha(),
        "device": str(device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def _build_model(preset: TrainPreset, num_classes: int, num_leads: int, device: torch.device):
    torch.manual_seed(preset.optim.seed)
    return preset.model.build(num_classes=num_classes, num_leads=num_leads).to(device)


def _build_optimizer(
    model: nn.Module, preset: TrainPreset, device: torch.device
) -> torch.optim.Optimizer:
    return build_optimizer(model, preset.optim, device)


def _train_step(
    model: nn.Module,
    batch: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    grad_clip_norm: float | None,
    device: torch.device,
) -> None:
    """One training step. Delegates to :func:`training.train.run_training_step`,
    the single source of truth for the real training loop's per-batch body, so
    benchmark step timings always measure exactly what a real run does."""
    run_training_step(model, optimizer, batch, device, loss_fn, grad_clip_norm)


def _fetch_batch_pool(loader: PyGDataLoader, pool_size: int) -> list[Any]:
    pool = list(itertools.islice(iter(loader), pool_size))
    if len(pool) < pool_size:
        raise RuntimeError(
            f"loader yielded only {len(pool)} batches; need {pool_size} for the batch pool"
        )
    return pool


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mem_gb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


class _LoaderFactory:
    """Rebuild train/val loaders over shared datasets for a given batch size."""

    def __init__(
        self,
        preset: TrainPreset,
        meta: Any,
        device: torch.device,
        num_workers: int,
    ) -> None:
        self.preset = preset
        self.device = device
        self.num_workers = num_workers
        splits = resolve_splits(preset, meta)
        self.train_idx = splits.train_idx
        self.exam_labels = meta.exam_labels
        self.train_ds, self.val_ds, self.test_ds = build_datasets(
            dataset_path=Path(preset.dataset_path),
            num_leads=meta.num_leads,
            train_idx=splits.train_idx,
            val_idx=splits.val_idx,
            test_idx=splits.test_idx,
            test_dataset_path=(
                Path(preset.test_dataset_path) if preset.test_dataset_path else None
            ),
            symmetrize_edges=preset.symmetrize_edges,
        )

    def train_loader(self, batch_size: int, *, epoch: int) -> PyGDataLoader:
        train_loader, _, _ = build_loaders_from_datasets(
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            test_ds=self.test_ds,
            train_idx=self.train_idx,
            exam_labels=self.exam_labels,
            sampling=self.preset.sampling,
            batch_size=batch_size,
            num_workers=self.num_workers,
            distributed=False,
            seed=self.preset.optim.seed,
            device=self.device,
            shuffle_block_exams=self.preset.optim.shuffle_block_exams,
        )
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        return train_loader

    def val_loader(self, batch_size: int) -> PyGDataLoader:
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.device.type == "cuda",
            "shuffle": False,
        }
        if self.num_workers > 0:
            kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
        return PyGDataLoader(self.val_ds, **kwargs)


def bench_loader(factory: _LoaderFactory, batch_size: int, n_batches: int) -> dict[str, Any]:
    loader = factory.train_loader(batch_size, epoch=0)
    cached_before = _meminfo_cached_gb()
    t0 = time.perf_counter()
    it = iter(loader)
    first = next(it)
    t_first = time.perf_counter() - t0
    exams = int(first.num_graphs)
    nodes = int(first.x.size(0))
    t1 = time.perf_counter()
    for batch in itertools.islice(it, n_batches - 1):
        exams += int(batch.num_graphs)
        nodes += int(batch.x.size(0))
    elapsed = time.perf_counter() - t1 + t_first
    del it, loader
    return {
        "batch_size": batch_size,
        "batches": n_batches,
        "exams": exams,
        "time_to_first_batch_s": t_first,
        "elapsed_s": elapsed,
        "exams_per_s": exams / elapsed,
        "nodes_per_s": nodes / elapsed,
        "page_cached_gb_before": cached_before,
        "page_cached_gb_after": _meminfo_cached_gb(),
    }


def bench_step(
    factory: _LoaderFactory,
    preset: TrainPreset,
    meta: Any,
    device: torch.device,
    batch_size: int,
    warmup: int,
    steps: int,
    use_compile: bool,
    pool_size: int = 8,
) -> dict[str, Any]:
    loader = factory.train_loader(batch_size, epoch=1)
    pool = _fetch_batch_pool(loader, pool_size)
    del loader
    exams_per_cycle = sum(int(b.num_graphs) for b in pool)

    model = _build_model(preset, meta.num_classes, meta.num_leads, device)
    if use_compile:
        model = torch.compile(model, dynamic=True)
    optimizer = _build_optimizer(model, preset, device)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()

    cycle = itertools.cycle(pool)
    for _ in range(warmup):
        _train_step(model, next(cycle), optimizer, loss_fn, preset.optim.grad_clip_norm, device)
    _cuda_sync(device)
    _reset_peak_mem(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        _train_step(model, next(cycle), optimizer, loss_fn, preset.optim.grad_clip_norm, device)
    _cuda_sync(device)
    elapsed = time.perf_counter() - t0

    exams_per_step = exams_per_cycle / len(pool)
    return {
        "batch_size": batch_size,
        "steps": steps,
        "ms_per_step": elapsed / steps * 1e3,
        "exams_per_s": exams_per_step * steps / elapsed,
        **_peak_mem_gb(device),
    }


def bench_e2e(
    factory: _LoaderFactory,
    preset: TrainPreset,
    meta: Any,
    device: torch.device,
    batch_size: int,
    warmup: int,
    steps: int,
    use_compile: bool,
) -> dict[str, Any]:
    model = _build_model(preset, meta.num_classes, meta.num_leads, device)
    if use_compile:
        model = torch.compile(model, dynamic=True)
    optimizer = _build_optimizer(model, preset, device)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()

    loader = factory.train_loader(batch_size, epoch=2)
    it = iter(loader)
    for batch in itertools.islice(it, warmup):
        _train_step(model, batch, optimizer, loss_fn, preset.optim.grad_clip_norm, device)
    _cuda_sync(device)
    _reset_peak_mem(device)

    exams = 0
    data_wait_s = 0.0
    t0 = time.perf_counter()
    for _ in range(steps):
        t_fetch = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        data_wait_s += time.perf_counter() - t_fetch
        exams += int(batch.num_graphs)
        _train_step(model, batch, optimizer, loss_fn, preset.optim.grad_clip_norm, device)
    _cuda_sync(device)
    elapsed = time.perf_counter() - t0
    del it, loader
    return {
        "batch_size": batch_size,
        "steps": steps,
        "exams": exams,
        "ms_per_step": elapsed / steps * 1e3,
        "exams_per_s": exams / elapsed,
        "data_wait_frac": data_wait_s / elapsed,
        **_peak_mem_gb(device),
    }


def bench_eval(
    factory: _LoaderFactory,
    preset: TrainPreset,
    meta: Any,
    device: torch.device,
    batch_size: int,
    n_batches: int,
) -> dict[str, Any]:
    model = _build_model(preset, meta.num_classes, meta.num_leads, device)
    model.eval()
    loader = factory.val_loader(batch_size)

    exams = 0
    with torch.inference_mode():
        it = iter(loader)
        for batch in itertools.islice(it, 3):
            with bf16_autocast(device):
                forward_batch(model, batch, device)
        _cuda_sync(device)
        _reset_peak_mem(device)
        t0 = time.perf_counter()
        for batch in itertools.islice(it, n_batches):
            with bf16_autocast(device):
                forward_batch(model, batch, device)
            exams += int(batch.num_graphs)
        _cuda_sync(device)
        elapsed = time.perf_counter() - t0
    del it, loader
    return {
        "batch_size": batch_size,
        "batches": n_batches,
        "exams": exams,
        "exams_per_s": exams / elapsed,
        **_peak_mem_gb(device),
    }


def bench_parity(
    factory: _LoaderFactory,
    preset: TrainPreset,
    meta: Any,
    device: torch.device,
    n_batches: int = 4,
) -> dict[str, Any]:
    """Compare eager vs compiled logits on identical weights and inputs."""
    eager = _build_model(preset, meta.num_classes, meta.num_leads, device)
    compiled_base = _build_model(preset, meta.num_classes, meta.num_leads, device)
    compiled_base.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_base, dynamic=True)
    eager.eval()
    compiled.eval()

    loader = factory.val_loader(batch_size=32)
    batches = _fetch_batch_pool(loader, n_batches)
    del loader

    max_diff_fp32 = 0.0
    max_diff_bf16 = 0.0
    with torch.inference_mode():
        for batch in batches:
            batch = batch.to(device)
            kwargs = batch_to_forward_kwargs(batch)
            diff = (eager(**kwargs).logits - compiled(**kwargs).logits).abs().max().item()
            max_diff_fp32 = max(max_diff_fp32, diff)
            with bf16_autocast(device):
                diff = (eager(**kwargs).logits - compiled(**kwargs).logits).abs().max().item()
            max_diff_bf16 = max(max_diff_bf16, diff)
    return {
        "batches": n_batches,
        "max_abs_logit_diff_fp32": max_diff_fp32,
        "max_abs_logit_diff_bf16_autocast": max_diff_bf16,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    if args.compile and "parity" not in sections:
        sections.append("parity")
    known = {"loader", "step", "e2e", "eval", "parity"}
    unknown = set(sections) - known
    if unknown:
        raise SystemExit(f"unknown sections: {sorted(unknown)}; valid: {sorted(known)}")
    if "parity" in sections and not args.compile:
        raise SystemExit("the parity section requires --compile")

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    eval_batch_sizes = [int(b) for b in args.eval_batch_sizes.split(",")]

    preset = load_train_preset(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_matmul_precision()
    num_workers = args.num_workers if args.num_workers is not None else preset.optim.num_workers

    print(f"[bench] config={args.config} device={device} num_workers={num_workers}", flush=True)
    meta = read_hdf5_meta(Path(preset.dataset_path))
    t0 = time.perf_counter()
    factory = _LoaderFactory(preset, meta, device, num_workers)
    startup_s = time.perf_counter() - t0
    print(f"[bench] datasets built in {startup_s:.1f}s", flush=True)

    results: dict[str, Any] = {
        "config": str(args.config),
        "dataset_path": preset.dataset_path,
        "num_workers": num_workers,
        "compile": bool(args.compile),
        "env": _env_info(device),
        "dataset_startup_s": startup_s,
        "sections": {},
    }

    def write_out() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(results, f, indent=2)

    def cuda_oom_guard(fn, *fn_args) -> dict[str, Any]:
        """Run one measurement; on CUDA OOM record the error instead of dying.

        Larger batch sizes are *expected* to eventually OOM during sweeps; the
        earlier measurements must survive. Anything other than OOM re-raises.
        """
        try:
            return fn(*fn_args)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            return {"error": "cuda_oom", "detail": str(exc).split("\n")[0]}

    if "loader" in sections:
        print("[bench] loader section...", flush=True)
        r = bench_loader(factory, preset.optim.batch_size, args.loader_batches)
        results["sections"]["loader"] = r
        print(
            f"  loader bs={r['batch_size']}: {r['exams_per_s']:.0f} exams/s, "
            f"first batch {r['time_to_first_batch_s']:.1f}s",
            flush=True,
        )
        write_out()

    if "step" in sections:
        results["sections"]["step"] = []
        for bs in batch_sizes:
            r = cuda_oom_guard(
                bench_step, factory, preset, meta, device, bs, args.warmup, args.steps, args.compile
            )
            r.setdefault("batch_size", bs)
            results["sections"]["step"].append(r)
            if "error" in r:
                print(f"  step bs={bs}: {r['error']}", flush=True)
            else:
                print(
                    f"  step bs={bs}: {r['ms_per_step']:.1f} ms/step, "
                    f"{r['exams_per_s']:.0f} exams/s, "
                    f"peak {r.get('peak_alloc_gb', float('nan')):.2f} GiB",
                    flush=True,
                )
            write_out()

    if "e2e" in sections:
        results["sections"]["e2e"] = []
        for bs in batch_sizes:
            r = cuda_oom_guard(
                bench_e2e,
                factory,
                preset,
                meta,
                device,
                bs,
                args.warmup,
                args.e2e_steps,
                args.compile,
            )
            r.setdefault("batch_size", bs)
            results["sections"]["e2e"].append(r)
            if "error" in r:
                print(f"  e2e bs={bs}: {r['error']}", flush=True)
            else:
                print(
                    f"  e2e bs={bs}: {r['exams_per_s']:.0f} exams/s, "
                    f"data wait {r['data_wait_frac'] * 100:.0f}%, "
                    f"peak {r.get('peak_alloc_gb', float('nan')):.2f} GiB",
                    flush=True,
                )
            write_out()

    if "eval" in sections:
        results["sections"]["eval"] = []
        for bs in eval_batch_sizes:
            r = cuda_oom_guard(bench_eval, factory, preset, meta, device, bs, args.eval_batches)
            r.setdefault("batch_size", bs)
            results["sections"]["eval"].append(r)
            if "error" in r:
                print(f"  eval bs={bs}: {r['error']}", flush=True)
            else:
                print(f"  eval bs={bs}: {r['exams_per_s']:.0f} exams/s", flush=True)
            write_out()

    if "parity" in sections:
        print("[bench] parity section...", flush=True)
        r = bench_parity(factory, preset, meta, device)
        results["sections"]["parity"] = r
        print(
            f"  parity: max |Δlogit| fp32={r['max_abs_logit_diff_fp32']:.3e}, "
            f"bf16={r['max_abs_logit_diff_bf16_autocast']:.3e}",
            flush=True,
        )
        write_out()

    write_out()
    print(f"[bench] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
