"""Compute-efficiency metrics for the HierGNNModel and its dataset.

This module provides two categories of functions:

**PURE** (no I/O, no side-effects, safe to call anywhere):
- :func:`compute_reduction_stats_from_arrays`: reduction-ratio math from numpy arrays
- :func:`_summary_stats`: mean/median/std of a 1-D array
- :func:`_beats_to_exam_sums`: segment-sum beats into per-exam totals
- :func:`count_parameters`: parameter counts from an nn.Module
- :func:`format_params_human`: human-readable string (e.g. "1.23M")

**I/O ADAPTER** (reads the HDF5 file, then delegates to a pure function):
- :func:`graph_reduction_stats`: reads graph_original/reduced_num_nodes and
  graph_edge_ptr from the dataset HDF5, then calls
  :func:`compute_reduction_stats_from_arrays`

**MEASUREMENT / STATEFUL** (wall-clock, device-dependent, or temporarily
mutates model device):
- :func:`measure_latency`: per-exam forward-pass wall time; depends on the
  current GPU/CPU state and is non-deterministic across runs
- :func:`measure_peak_memory`: resets CUDA peak-memory stats and runs
  multiple forwards; no-op on CPU
- :func:`measure_flops`: temporarily moves the model to CPU (restored in a
  ``finally`` block), runs forward passes under FlopCounterMode, then returns
  the model to its original device

FLOPs coverage note
-------------------
:func:`measure_flops` uses ``torch.utils.flop_counter.FlopCounterMode``,
which counts ATen linear operations (``addmm``, ``mm``).

**Convention (empirically verified):** ``FlopCounterMode.get_total_flops()``
returns *FLOPs* (i.e. multiply-add = 2 ops per MAC). Verified by checking a
single ``nn.Linear(16, 8, bias=False)`` on a batch-4 input: the count is
1024 == 2 * 4 * 16 * 8 (FLOPs), not 512 (MACs).

The PyG GATv2/GIN scatter-gather aggregation ops are **not** instrumented by
FlopCounterMode, so the reported FLOPs are a *lower bound*. The number is
still useful for cross-checkpoint comparisons (all checkpoints miss the same
ops), but should not be presented as a complete FLOP count in a paper.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
import torch.nn as nn

from training.inference import single_exam_inputs

if TYPE_CHECKING:
    from training.dataset import ExamGraphDataset


def count_parameters(model: nn.Module) -> dict[str, object]:
    """Return total and per-named-child parameter counts.

    Only direct children are broken out; grandchildren are rolled up into
    their parent. Buffers (non-trainable) are not counted.

    Returns
    -------
    dict with keys ``"total"`` (int) and ``"submodules"`` (dict[str, int]).
    """
    submodule_counts: dict[str, int] = {}
    for name, child in model.named_children():
        submodule_counts[name] = sum(p.numel() for p in child.parameters())

    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "submodules": submodule_counts}


def _summary_stats(arr: np.ndarray) -> dict[str, float]:
    """Return mean/median/std for a 1-D float array."""
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
    }


def _beats_to_exam_sums(per_beat: np.ndarray, beats_per_exam: np.ndarray) -> np.ndarray:
    """Segment-sum per-beat values into per-exam totals.

    ``per_beat`` must be ordered so that exam i's beats occupy indices
    ``[cumsum[i], cumsum[i+1])`` where ``cumsum = [0] + list(cumsum(beats_per_exam))``.
    """
    beats_per_exam = np.asarray(beats_per_exam, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(beats_per_exam)])
    result = np.zeros(len(beats_per_exam), dtype=np.float64)
    for i in range(len(beats_per_exam)):
        result[i] = per_beat[int(offsets[i]) : int(offsets[i + 1])].sum()
    return result


def compute_reduction_stats_from_arrays(
    original_num_nodes: np.ndarray,
    reduced_num_nodes: np.ndarray,
    edges_per_beat: np.ndarray,
    beats_per_exam: np.ndarray,
    source_sampling_rate: int,
    target_sampling_rate: int,
) -> dict[str, object]:
    """Pure computation of graph-reduction statistics from pre-filtered arrays.

    All three per-beat arrays must be ordered in exam order so that
    :func:`_beats_to_exam_sums` can aggregate them correctly.

    For each baseline the function reports two complementary per-beat ratios.
    Keys are rate-agnostic (no Hz literals) so they remain correct regardless
    of the dataset's sampling rates; the actual Hz values appear in each
    entry's ``"label"`` sub-field.

    * ``retained_fraction_resampled``: nodes KEPT vs post-resample baseline
      (reduced / original at ``target_sampling_rate`` Hz).
    * ``reduction_fraction_resampled``: nodes REMOVED vs post-resample
      baseline (1 - retained); the intuitive "sparsity" figure.
    * ``retained_fraction_native``: nodes KEPT vs implied native source-rate
      baseline (reduced / (original * source_sr / target_sr)).
    * ``reduction_fraction_native``: nodes REMOVED vs native baseline.

    Both retained and reduction fractions are reported explicitly so readers
    cannot confuse "12.8% retained" with "12.8% removed" in a paper table.

    Parameters
    ----------
    original_num_nodes:
        Node count *before* turning-point reduction, per beat. These nodes
        live in the ``target_sampling_rate`` Hz space (post-resample,
        pre-TP-reduction).
    reduced_num_nodes:
        Node count *after* turning-point reduction, per beat.
    edges_per_beat:
        Edge count per beat (``np.diff(graph_edge_ptr)``).
    beats_per_exam:
        Number of graphs/beats belonging to each exam, in dataset order.
    source_sampling_rate:
        Sampling rate of the raw source signal (e.g. 400 Hz).
    target_sampling_rate:
        Sampling rate after resampling, i.e. the rate at which
        ``original_num_nodes`` was computed (e.g. 300 Hz).
    """
    orig = np.asarray(original_num_nodes, dtype=np.float64)
    reduced = np.asarray(reduced_num_nodes, dtype=np.float64)
    epb = np.asarray(edges_per_beat, dtype=np.float64)

    safe_orig = np.where(orig > 0, orig, 1.0)

    retained_resampled = reduced / safe_orig
    reduction_resampled = 1.0 - retained_resampled

    implied_orig_native = safe_orig * source_sampling_rate / target_sampling_rate
    retained_native = reduced / implied_orig_native
    reduction_native = 1.0 - retained_native

    nodes_per_exam = _beats_to_exam_sums(reduced, beats_per_exam)
    edges_per_exam = _beats_to_exam_sums(epb, beats_per_exam)

    return {
        "n_graphs": len(original_num_nodes),
        "n_exams": len(beats_per_exam),
        "target_sampling_rate_hz": int(target_sampling_rate),
        "source_sampling_rate_hz": int(source_sampling_rate),
        "original_nodes_per_beat": _summary_stats(orig),
        "reduced_nodes_per_beat": _summary_stats(reduced),
        "retained_fraction_resampled": {
            **_summary_stats(retained_resampled),
            "label": (
                f"fraction of nodes KEPT after TP reduction"
                f" (vs {target_sampling_rate} Hz post-resample baseline)"
            ),
        },
        "reduction_fraction_resampled": {
            **_summary_stats(reduction_resampled),
            "label": (
                f"fraction of nodes REMOVED after TP reduction"
                f" (sparsity vs {target_sampling_rate} Hz post-resample baseline)"
            ),
        },
        "retained_fraction_native": {
            **_summary_stats(retained_native),
            "label": (
                f"fraction of nodes KEPT after TP reduction"
                f" (vs implied {source_sampling_rate} Hz native-rate baseline)"
            ),
        },
        "reduction_fraction_native": {
            **_summary_stats(reduction_native),
            "label": (
                f"fraction of nodes REMOVED after TP reduction"
                f" (sparsity vs implied {source_sampling_rate} Hz native-rate baseline)"
            ),
        },
        "edges_per_beat": _summary_stats(epb),
        "nodes_per_exam": _summary_stats(nodes_per_exam),
        "edges_per_exam": _summary_stats(edges_per_exam),
    }


def graph_reduction_stats(ds: ExamGraphDataset) -> dict[str, object]:
    """Read graph-reduction statistics for the exams in ``ds``.

    Reads ``graph_original_num_nodes``, ``graph_reduced_num_nodes``, and
    ``graph_edge_ptr`` from the HDF5 file, filters to the dataset's
    ``exam_indices``, and delegates to
    :func:`compute_reduction_stats_from_arrays`.

    Returns a nested dict of scalars suitable for JSON serialisation.
    """
    with h5py.File(ds.h5_path, "r") as f:
        all_orig = f["graph_original_num_nodes"][:].astype(np.int64)
        all_reduced = f["graph_reduced_num_nodes"][:].astype(np.int64)
        all_edge_ptr = f["graph_edge_ptr"][:].astype(np.int64)
        missing_attrs = [
            name for name in ("source_sampling_rate", "sampling_rate") if name not in f.attrs
        ]
        if missing_attrs:
            raise ValueError(
                f"{ds.h5_path}: missing required sampling-rate HDF5 attributes {missing_attrs}"
            )
        rates: dict[str, int] = {}
        for name in ("source_sampling_rate", "sampling_rate"):
            try:
                raw = float(f.attrs[name])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{ds.h5_path}: sampling-rate HDF5 attributes must be positive integers"
                ) from exc
            if not np.isfinite(raw) or raw <= 0 or not raw.is_integer():
                raise ValueError(
                    f"{ds.h5_path}: sampling-rate HDF5 attributes must be positive integers"
                )
            rates[name] = int(raw)
        source_sr = rates["source_sampling_rate"]
        target_sr = rates["sampling_rate"]

    exam_ids = ds.exam_indices
    g_starts = ds.exam_to_graph_start[exam_ids]
    g_ends = ds.exam_to_graph_end[exam_ids]
    beats_per_exam = (g_ends - g_starts).astype(np.int64)

    orig_parts: list[np.ndarray] = []
    reduced_parts: list[np.ndarray] = []
    epb_parts: list[np.ndarray] = []

    all_edges_per_beat = np.diff(all_edge_ptr).astype(np.int64)
    for s, e in zip(g_starts, g_ends, strict=True):
        orig_parts.append(all_orig[s:e])
        reduced_parts.append(all_reduced[s:e])
        epb_parts.append(all_edges_per_beat[s:e])

    return compute_reduction_stats_from_arrays(
        np.concatenate(orig_parts) if orig_parts else np.empty(0, dtype=np.int64),
        np.concatenate(reduced_parts) if reduced_parts else np.empty(0, dtype=np.int64),
        np.concatenate(epb_parts) if epb_parts else np.empty(0, dtype=np.int64),
        beats_per_exam,
        source_sr,
        target_sr,
    )


def measure_latency(
    model: nn.Module,
    dataset: ExamGraphDataset,
    device: torch.device,
    n_exams: int = 200,
    warmup: int = 10,
) -> dict[str, object]:
    """Measure per-exam forward-pass latency.

    Each exam is batched individually (batch_size=1) to measure single-exam
    latency as reported in a paper's efficiency section. Warmup iterations
    are discarded before measurement begins.

    GPU timing uses CUDA events (``cudaEvent_t``) to respect the async
    execution model; CPU timing uses ``time.perf_counter``.

    Forward-kwargs construction (dataset[i] → Batch → kwargs) is performed
    **before** the timed region so only the model call is measured.

    Returns mean, median, std (all in ms) and throughput (exams/s).
    """
    model.eval()
    n_available = len(dataset)
    n_measure = min(n_exams, n_available)
    n_total = warmup + n_measure

    times_ms: list[float] = []

    with torch.no_grad():
        for i in range(n_total):
            idx = i % n_available
            inputs = single_exam_inputs(dataset, idx, device)

            if device.type == "cuda":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                model(**inputs)
                end_ev.record()
                torch.cuda.synchronize()
                elapsed_ms = start_ev.elapsed_time(end_ev)
            else:
                t0 = time.perf_counter()
                model(**inputs)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if i >= warmup:
                times_ms.append(elapsed_ms)

    arr = np.array(times_ms, dtype=np.float64)
    return {
        "n_exams_measured": len(arr),
        "warmup": int(warmup),
        "device": str(device),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr)),
        "throughput_exams_per_s": float(1000.0 / np.mean(arr)),
    }


def measure_peak_memory(
    model: nn.Module,
    dataset: ExamGraphDataset,
    device: torch.device,
    n_exams: int = 50,
) -> dict[str, object]:
    """Measure worst-case peak GPU memory over a sample of exams.

    Peak memory is graph-size-dependent, so sampling multiple exams and
    reporting the maximum provides a more representative bound than a single
    exam. For each exam the peak-memory stats are reset before the forward
    so only that exam's footprint is captured; the maximum across all sampled
    exams is then reported along with the graph size that produced it.

    Only meaningful on CUDA; returns ``None`` for the byte fields on CPU.
    """
    if device.type != "cuda":
        return {
            "device": str(device),
            "max_memory_allocated_mb": None,
            "note": "Peak-memory measurement requires a CUDA device.",
        }

    model.eval()
    n = min(n_exams, len(dataset))

    max_peak_bytes = 0
    max_peak_idx = 0
    max_peak_n_nodes = 0
    max_peak_n_edges = 0

    with torch.no_grad():
        for i in range(n):
            inputs = single_exam_inputs(dataset, i, device)
            torch.cuda.reset_peak_memory_stats(device)
            model(**inputs)
            peak = torch.cuda.max_memory_allocated(device)
            if peak > max_peak_bytes:
                max_peak_bytes = peak
                max_peak_idx = i
                max_peak_n_nodes = int(inputs["x"].shape[0])
                max_peak_n_edges = int(inputs["edge_index"].shape[1])

    return {
        "device": str(device),
        "n_exams_sampled": int(n),
        "max_memory_allocated_mb": float(max_peak_bytes) / (1024**2),
        "worst_case_exam_idx": int(max_peak_idx),
        "worst_case_n_nodes": int(max_peak_n_nodes),
        "worst_case_n_edges": int(max_peak_n_edges),
    }


_FLOPS_COVERAGE_NOTE = (
    "FlopCounterMode convention (verified): get_total_flops() returns FLOPs "
    "(2 ops per multiply-add; MACs = FLOPs / 2). "
    "Ops counted: ATen linear operations (addmm, mm). "
    "NOT counted: PyG GATv2 and GIN sparse scatter-gather aggregations. "
    "Reported FLOPs are therefore a lower bound on total ops, valid for "
    "cross-checkpoint comparison but must not be cited as a complete operation count."
)


def measure_flops(
    model: nn.Module,
    dataset: ExamGraphDataset,
    n_exams: int = 50,
) -> dict[str, object]:
    """Estimate per-exam FLOPs using ``FlopCounterMode``.

    Always runs on CPU so FlopCounterMode does not conflict with CUDA streams.
    The model is temporarily moved to CPU and **restored to its original
    device in a** ``finally`` **block**, so the caller's model is not left on CPU.

    Input preparation (dataset[i] → Batch → kwargs) is excluded from the
    counted region. Only the model forward is instrumented.

    **Convention:** ``get_total_flops()`` returns *FLOPs* (2 per MAC); see
    the module-level docstring and ``_FLOPS_COVERAGE_NOTE`` for full details.
    """
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    cpu_device = torch.device("cpu")

    first_param = next(model.parameters(), None)
    original_device = first_param.device if first_param is not None else cpu_device

    n = min(n_exams, len(dataset))
    inputs_list = [single_exam_inputs(dataset, i, cpu_device) for i in range(n)]

    flops_list: list[int] = []
    try:
        model.to(cpu_device)
        for inputs in inputs_list:
            with FlopCounterMode(display=False) as fc, torch.no_grad():
                model(**inputs)
            flops_list.append(int(fc.get_total_flops()))
    finally:
        model.to(original_device)

    arr = np.array(flops_list, dtype=np.float64)
    return {
        "n_exams_measured": int(n),
        "mean_flops": float(np.mean(arr)),
        "median_flops": float(np.median(arr)),
        "std_flops": float(np.std(arr)),
        "unit": "FLOPs (multiply-add = 2 ops; MACs = FLOPs / 2)",
        "coverage_note": _FLOPS_COVERAGE_NOTE,
    }


def format_params_human(n: int) -> str:
    """Format a parameter count as a human-readable string.

    Examples
    --------
    >>> format_params_human(1234567)
    '1.23M'
    >>> format_params_human(999)
    '999'
    >>> format_params_human(1_000_000_000)
    '1.00B'
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)
