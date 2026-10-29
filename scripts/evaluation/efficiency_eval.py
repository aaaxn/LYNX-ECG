"""Efficiency quantification for a HierGNNModel checkpoint.

Measures and reports five groups of metrics:
  1. Parameter count:  total and per-submodule
  2. Graph reduction:  how aggressively turning-point reduction compresses beats
  3. Latency:          per-exam forward-pass wall time (mean/median/std, throughput)
  4. Peak memory:      worst-case GPU memory across a sample of exams (batch_size=1)
  5. FLOPs:            ATen linear-op count via FlopCounterMode (lower bound only)

Usage:
    uv run python scripts/evaluation/efficiency_eval.py \\
        --ckpt results/code100_tp_s55555/best_f1_val.pt \\
        --dataset graph_datasets/codetest/codetest__turning_points__40hz-original-mgap80ms__raw_400hz__bp0p5-40hz-o4__resample-300hz__norm-p95med__natural__leads-0-1-2-3-4-5-6-7-8-9-10-11.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from explain.report import (
    build_analysis_report,
    capture_report_provenance,
    write_analysis_report,
)
from training.efficiency import (
    count_parameters,
    format_params_human,
    graph_reduction_stats,
    measure_flops,
    measure_latency,
    measure_peak_memory,
)
from training.inference import load_checkpoint_and_dataset, resolve_device

SCRIPT_NAME = "efficiency_eval"


def _divider(title: str = "", width: int = 72) -> str:
    if title:
        pad = width - len(title) - 2
        return f"{'─' * (pad // 2)} {title} {'─' * (pad - pad // 2)}"
    return "─" * width


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Efficiency quantification for a HierGNNModel checkpoint."
    )
    ap.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint .pt file.")
    ap.add_argument(
        "--dataset", type=Path, required=True, help="Path to current-contract HDF5 dataset."
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="Device for latency/memory ('auto', 'cpu', 'cuda', 'cuda:1', …). "
        "FLOPs always run on CPU regardless of this flag.",
    )
    ap.add_argument(
        "--n-exams",
        type=_positive_int,
        default=200,
        help="Number of exams to time for latency (after warmup). Must be >= 1.",
    )
    ap.add_argument(
        "--warmup",
        type=_non_negative_int,
        default=10,
        help="Warmup iterations discarded before latency measurement. Must be >= 0.",
    )
    ap.add_argument(
        "--n-flops-exams",
        type=_positive_int,
        default=50,
        help="Number of exams to profile for FLOPs estimation. Must be >= 1.",
    )
    ap.add_argument(
        "--n-memory-exams",
        type=_positive_int,
        default=50,
        help="Number of exams to sample for worst-case peak-memory measurement. Must be >= 1.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to reports/efficiency/<ckpt_stem>.json.",
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    try:
        bundle = load_checkpoint_and_dataset(args.ckpt, args.dataset, device)
    except ValueError as exc:
        raise SystemExit(f"incompatible checkpoint/dataset: {exc}") from exc
    model, meta = bundle.model, bundle.meta
    provenance = capture_report_provenance(
        args.ckpt,
        args.dataset,
        checkpoint_state=bundle.state,
        checkpoint_sha256=bundle.checkpoint_sha256,
        dataset_meta=meta,
    )
    class_names = list(meta.class_names)
    n_exams_total = meta.total_exams

    print(f"\n{'═' * 72}")
    print(f"EFFICIENCY REPORT: {args.ckpt.name}")
    print(f"{'═' * 72}")
    print(f"checkpoint : {args.ckpt}")
    print(f"dataset    : {args.dataset.name}  ({n_exams_total} exams)")
    print(f"device     : {device}")
    print(f"classes    : {class_names}")

    results: dict[str, object] = {
        "n_exams_total": n_exams_total,
        "device": str(device),
        "class_names": class_names,
    }

    with bundle:
        dataset = bundle.dataset
        print(f"\n{_divider('1. PARAMETERS')}")
        param_stats = count_parameters(model)
        total = int(param_stats["total"])
        submodules: dict[str, int] = param_stats["submodules"]  # type: ignore[assignment]
        print(f"  {'total':<28}  {format_params_human(total):>8}  ({total:,})")
        for name, count in submodules.items():
            pct = 100.0 * count / total if total > 0 else 0.0
            print(f"  {'  ' + name:<28}  {format_params_human(count):>8}  ({pct:.1f}%)")
        results["parameters"] = param_stats

        print(f"\n{_divider('2. GRAPH REDUCTION')}")
        red_stats = graph_reduction_stats(dataset)
        n_graphs = int(red_stats["n_graphs"])  # type: ignore[arg-type]
        target_sr = int(red_stats["target_sampling_rate_hz"])  # type: ignore[arg-type]
        source_sr = int(red_stats["source_sampling_rate_hz"])  # type: ignore[arg-type]
        print(f"  graphs (beats) in dataset  : {n_graphs:,}")
        print(f"  exams  in dataset          : {int(red_stats['n_exams']):,}")  # type: ignore[arg-type]
        print(f"  post-resample rate         : {target_sr} Hz")
        print(f"  native source rate         : {source_sr} Hz")

        orig = red_stats["original_nodes_per_beat"]  # type: ignore[index]
        reduced = red_stats["reduced_nodes_per_beat"]  # type: ignore[index]
        print(f"\n  {'metric':<48}  {'mean':>8}  {'median':>8}  {'std':>8}")
        print(f"  {'─' * 78}")
        print(
            f"  {f'original nodes/beat ({target_sr} Hz, post-resample, pre-TP)':<48}"
            f"  {orig['mean']:>8.1f}  {orig['median']:>8.1f}  {orig['std']:>8.1f}"
        )
        print(
            f"  {'reduced nodes/beat (post-TP)':<48}"
            f"  {reduced['mean']:>8.1f}  {reduced['median']:>8.1f}  {reduced['std']:>8.1f}"
        )

        rd_r = red_stats["reduction_fraction_resampled"]  # type: ignore[index]
        rt_r = red_stats["retained_fraction_resampled"]  # type: ignore[index]
        rd_n = red_stats["reduction_fraction_native"]  # type: ignore[index]
        rt_n = red_stats["retained_fraction_native"]  # type: ignore[index]

        print(
            f"  {f'nodes REMOVED vs {target_sr} Hz post-resample (sparsity)':<48}"
            f"  {rd_r['mean']:>8.4f}  {rd_r['median']:>8.4f}  {rd_r['std']:>8.4f}"
        )
        print(
            f"  {f'  nodes KEPT vs {target_sr} Hz post-resample (retained)':<48}"
            f"  {rt_r['mean']:>8.4f}  {rt_r['median']:>8.4f}  {rt_r['std']:>8.4f}"
        )
        print(
            f"  {f'nodes REMOVED vs native {source_sr} Hz (sparsity)':<48}"
            f"  {rd_n['mean']:>8.4f}  {rd_n['median']:>8.4f}  {rd_n['std']:>8.4f}"
        )
        print(
            f"  {f'  nodes KEPT vs native {source_sr} Hz (retained)':<48}"
            f"  {rt_n['mean']:>8.4f}  {rt_n['median']:>8.4f}  {rt_n['std']:>8.4f}"
        )
        print(
            f"\n  C1 headline: {rd_r['mean'] * 100:.1f}% removed vs {target_sr} Hz post-resample,"
            f" {rd_n['mean'] * 100:.1f}% removed vs native {source_sr} Hz"
        )

        epb = red_stats["edges_per_beat"]  # type: ignore[index]
        npe = red_stats["nodes_per_exam"]  # type: ignore[index]
        epe = red_stats["edges_per_exam"]  # type: ignore[index]
        print(
            f"  {'edges/beat':<48}  {epb['mean']:>8.1f}  {epb['median']:>8.1f}  {epb['std']:>8.1f}"
        )
        print(
            f"  {'nodes/exam (reduced)':<48}"
            f"  {npe['mean']:>8.1f}  {npe['median']:>8.1f}  {npe['std']:>8.1f}"
        )
        print(
            f"  {'edges/exam':<48}  {epe['mean']:>8.1f}  {epe['median']:>8.1f}  {epe['std']:>8.1f}"
        )
        results["graph_reduction"] = red_stats

        print(f"\n{_divider('3. LATENCY (batch_size=1)')}")
        lat = measure_latency(model, dataset, device, n_exams=args.n_exams, warmup=args.warmup)
        print(f"  device          : {lat['device']}")
        print(f"  exams timed     : {lat['n_exams_measured']}  (warmup={lat['warmup']})")
        print(f"  mean latency    : {lat['mean_ms']:.2f} ms")
        print(f"  median latency  : {lat['median_ms']:.2f} ms")
        print(f"  std latency     : {lat['std_ms']:.2f} ms")
        print(f"  throughput      : {lat['throughput_exams_per_s']:.1f} exams/s")
        results["latency"] = lat

        print(f"\n{_divider('4. PEAK MEMORY (worst case, batch_size=1)')}")
        mem = measure_peak_memory(model, dataset, device, n_exams=args.n_memory_exams)
        if mem["max_memory_allocated_mb"] is None:
            print(f"  {mem.get('note', 'N/A, CUDA not in use')}")
        else:
            print(f"  device                     : {mem['device']}")
            print(f"  exams sampled              : {mem['n_exams_sampled']}")
            print(f"  max_memory_allocated (peak): {mem['max_memory_allocated_mb']:.2f} MB")
            print(
                f"  worst-case exam            : idx={mem['worst_case_exam_idx']},"
                f" nodes={mem['worst_case_n_nodes']}, edges={mem['worst_case_n_edges']}"
            )
        results["peak_memory"] = mem

        print(f"\n{_divider('5. FLOPs (ATen linear ops, lower bound)')}")
        flop_stats = measure_flops(model, dataset, n_exams=args.n_flops_exams)
        unit_label = flop_stats.get("unit", "FLOPs")
        print(f"  exams profiled  : {flop_stats['n_exams_measured']}")
        print(f"  unit            : {unit_label}")
        print(f"  mean            : {flop_stats['mean_flops'] / 1e9:.3f} GFLOPs")
        print(f"  median          : {flop_stats['median_flops'] / 1e9:.3f} GFLOPs")
        print(f"  std             : {flop_stats['std_flops'] / 1e9:.3f} GFLOPs")
        print(f"  (MACs = FLOPs / 2 = {flop_stats['mean_flops'] / 2 / 1e9:.3f} GMACs mean)")
        print(f"\n  CAVEAT: {flop_stats['coverage_note']}")
        results["flops"] = flop_stats

    print(f"\n{'═' * 72}\n")

    out_path = args.output or (Path("reports/efficiency") / f"{args.ckpt.stem}.json")
    parameters = {
        "device": args.device,
        "n_exams": args.n_exams,
        "warmup": args.warmup,
        "n_flops_exams": args.n_flops_exams,
        "n_memory_exams": args.n_memory_exams,
    }
    report = build_analysis_report(
        script=SCRIPT_NAME, provenance=provenance, parameters=parameters, payload=results
    )
    write_analysis_report(out_path, report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
