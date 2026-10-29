"""Go/no-go gate: does the model attend to the clinically correct region?

For every positive exam of each morphology class, this computes node importance
two ways, Integrated Gradients (:func:`explain.integrated_gradients`) and the
intra-lead attention gate expanded to nodes, and measures how concentrated that
importance is in the class's fiducial region (:func:`explain.region_alignment`).

A lift > 1 means importance lands in the clinically expected region more than a
uniform spread would. Three references make the signal interpretable:
  - in-region lift (the claim),
  - decoy-region lift (a different fiducial window, tests *specificity*),
  - random-importance lift (≈ 1, the null).

Rhythm classes (AF, SB, ST) are diagnosed from inter-beat rhythm, not a
within-beat window, so they are excluded (see explain.clinical_alignment).

Usage:
    uv run python scripts/explain/clinical_alignment_gate.py \
        --ckpt results/code100_tp_s55555/best_f1_val.pt \
        --dataset graph_datasets/codetest/codetest__turning_points__40hz-original-mgap80ms__raw_400hz__bp0p5-40hz-o4__resample-300hz__norm-p95med__natural__leads-0-1-2-3-4-5-6-7-8-9-10-11.h5
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from explain.attribution import integrated_gradients
from explain.clinical_alignment import (
    decoy_region_window,
    expand_subgraph_scores_to_nodes,
    region_alignment,
    region_node_mask,
    region_window,
)
from explain.ecg_timing import exam_timing
from explain.report import (
    build_analysis_report,
    capture_report_provenance,
    write_analysis_report,
)
from explain.report import sanitize_stats as _sanitize_stats
from training.bootstrap import median_iqr_ci
from training.inference import load_checkpoint_and_dataset, resolve_device, single_exam_inputs

SCRIPT_NAME = "clinical_alignment_gate"


def empirical_feature_median(
    dataset_path: Path,
    *,
    sample_nodes: int,
    seed: int,
    sample_windows: int = 256,
) -> np.ndarray:
    """Estimate the training-set node-feature median from deterministic windows.

    Full CODE contains billions of nodes, so an exact scan would turn a baseline
    sensitivity check into a second corpus build. We instead read evenly sized,
    reproducibly sampled contiguous windows. Contiguous reads respect HDF5 chunk
    locality; the fixed seed and reported sample size make the estimate
    reproducible. Small datasets are read exactly.
    """
    if sample_nodes < 1:
        raise ValueError(f"sample_nodes must be >= 1, got {sample_nodes}")
    if sample_windows < 1:
        raise ValueError(f"sample_windows must be >= 1, got {sample_windows}")

    with h5py.File(dataset_path, "r") as h5:
        node_x = h5["node_x"]
        if node_x.ndim != 2:
            raise ValueError(f"node_x must be 2-D, got shape {node_x.shape}")
        total_nodes, num_features = map(int, node_x.shape)
        if total_nodes == 0:
            raise ValueError(f"{dataset_path}: node_x is empty")

        target = min(sample_nodes, total_nodes)
        if target == total_nodes:
            sample = np.asarray(node_x[:], dtype=np.float64)
        else:
            windows = min(sample_windows, target)
            window_size = math.ceil(target / windows)
            max_start = max(total_nodes - window_size, 0)
            rng = np.random.default_rng(seed)
            starts = np.sort(rng.integers(0, max_start + 1, size=windows))
            chunks = [
                np.asarray(node_x[int(start) : int(start) + window_size], dtype=np.float64)
                for start in starts
            ]
            sample = np.concatenate(chunks, axis=0)[:target]

    median = np.median(sample, axis=0)
    if median.shape != (num_features,) or not np.isfinite(median).all():
        raise ValueError(f"non-finite or malformed feature median: {median!r}")
    return median.astype(np.float32, copy=False)


def make_ig_baseline(
    inputs_x: torch.Tensor,
    *,
    mode: str,
    empirical_median: np.ndarray | None,
) -> torch.Tensor | None:
    """Construct an exam-shaped IG baseline, or ``None`` for the zero default."""
    if mode == "zero":
        if empirical_median is not None:
            raise ValueError("empirical_median must be None for zero baseline")
        return None
    if mode != "empirical_feature_median":
        raise ValueError(f"unknown IG baseline mode: {mode!r}")
    if empirical_median is None:
        raise ValueError("empirical_feature_median mode requires empirical_median")
    if empirical_median.shape != (inputs_x.shape[1],):
        raise ValueError(
            f"baseline feature shape {empirical_median.shape} != ({inputs_x.shape[1]},)"
        )
    vector = torch.as_tensor(empirical_median, dtype=inputs_x.dtype, device=inputs_x.device)
    return vector.unsqueeze(0).expand_as(inputs_x)


def class_alignment_ok(ig_median: float, decoy_median: float) -> bool:
    """Gate rule for one class: in-region IG lift must beat 1.0 AND the decoy."""
    return ig_median > 1.0 and ig_median > decoy_median


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ig-steps", type=int, default=64)
    ap.add_argument(
        "--ig-baseline",
        choices=("zero", "empirical_feature_median"),
        default="zero",
    )
    ap.add_argument(
        "--baseline-dataset",
        type=Path,
        default=None,
        help="Training HDF5 used only to estimate empirical feature medians.",
    )
    ap.add_argument("--baseline-sample-nodes", type=int, default=1_000_000)
    ap.add_argument("--baseline-seed", type=int, default=0)
    ap.add_argument("--max-exams-per-class", type=int, default=0, help="0 = all positives")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random-importance control")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for median CIs")
    ap.add_argument("--alpha", type=float, default=0.05, help="two-sided CI level")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if args.ig_baseline == "empirical_feature_median" and args.baseline_dataset is None:
        ap.error("--baseline-dataset is required for empirical_feature_median")
    if args.ig_baseline == "zero" and args.baseline_dataset is not None:
        ap.error("--baseline-dataset is only valid for empirical_feature_median")
    empirical_median = (
        empirical_feature_median(
            args.baseline_dataset,
            sample_nodes=args.baseline_sample_nodes,
            seed=args.baseline_seed,
        )
        if args.baseline_dataset is not None
        else None
    )

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
    ckpt_classes = list(meta.class_names)

    with bundle, h5py.File(args.dataset, "r") as h5:
        dataset = bundle.dataset
        node_feature_names = [str(c) for c in h5.attrs["node_feature_names"]]
        labels = h5["exam_labels"][:].astype(np.int64)
        graph_exam_index = h5["graph_exam_index"][:]

        morph_classes = [c for c in ckpt_classes if region_window(c) is not None]
        print(f"\n{'=' * 78}\nCLINICAL-ALIGNMENT GATE: {args.ckpt.name}\n{'=' * 78}")
        print(f"dataset={args.dataset.name}  classes(morphology)={morph_classes}")
        print(f"ig_steps={args.ig_steps}  baseline={args.ig_baseline}  device={device}\n")

        results: dict[str, dict] = {}
        for cls_name in morph_classes:
            cls = ckpt_classes.index(cls_name)
            window = region_window(cls_name)
            decoy = decoy_region_window(cls_name)
            assert window is not None and decoy is not None
            rng = torch.Generator().manual_seed(args.seed + cls)

            positives = np.flatnonzero(labels[:, cls] == 1)
            if args.max_exams_per_class > 0:
                positives = positives[: args.max_exams_per_class]

            ig_in, attn_in, ig_decoy, rand_in, region_nodes = [], [], [], [], []
            ig_in_by_ch: dict[str, list[float]] = {n: [] for n in node_feature_names}
            ig_decoy_by_ch: dict[str, list[float]] = {n: [] for n in node_feature_names}
            for exam_idx in positives:
                exam_idx = int(exam_idx)
                inputs = single_exam_inputs(dataset, exam_idx, device)
                n_nodes = int(inputs["x"].shape[0])

                baseline = make_ig_baseline(
                    inputs["x"], mode=args.ig_baseline, empirical_median=empirical_median
                )
                attr = integrated_gradients(
                    model,
                    inputs,
                    target_class=cls,
                    baseline=baseline,
                    steps=args.ig_steps,
                )
                ig_imp = attr.node_importance.detach().cpu()

                with torch.no_grad():
                    out = model(**inputs, return_attention=True)
                attn_imp = expand_subgraph_scores_to_nodes(
                    out.attention.intra_lead_gate.detach().cpu(),
                    inputs["subgraph_node_batch"].cpu(),
                )

                timing = exam_timing(h5, exam_idx, graph_exam_index=graph_exam_index)
                in_mask = region_node_mask(
                    timing.node_t, timing.node_beat, timing.beat_r_peak_t, window
                )
                decoy_mask = region_node_mask(
                    timing.node_t, timing.node_beat, timing.beat_r_peak_t, decoy
                )

                ig_score = region_alignment(ig_imp, in_mask)
                ig_in.append(ig_score.lift)
                region_nodes.append(float(ig_score.n_region_nodes))
                attn_in.append(region_alignment(attn_imp, in_mask).lift)
                ig_decoy.append(region_alignment(ig_imp, decoy_mask).lift)
                feat_imp = attr.feature_importance.detach().cpu()
                for ci, name in enumerate(node_feature_names):
                    ig_in_by_ch[name].append(region_alignment(feat_imp[:, ci], in_mask).lift)
                    ig_decoy_by_ch[name].append(region_alignment(feat_imp[:, ci], decoy_mask).lift)
                rand_in.append(region_alignment(torch.rand(n_nodes, generator=rng), in_mask).lift)

            ci_kw = {"n_boot": args.n_boot, "alpha": args.alpha, "seed": args.seed + cls}
            results[cls_name] = {
                "n_positives": int(positives.size),
                "ig_in_region": median_iqr_ci(ig_in, **ci_kw),
                "attn_in_region": median_iqr_ci(attn_in, **ci_kw),
                "ig_decoy_region": median_iqr_ci(ig_decoy, **ci_kw),
                "random_in_region": median_iqr_ci(rand_in, **ci_kw),
                "region_nodes": median_iqr_ci(region_nodes, **ci_kw),
                "ig_in_region_by_channel": {
                    n: median_iqr_ci(ig_in_by_ch[n], **ci_kw) for n in node_feature_names
                },
                "ig_decoy_region_by_channel": {
                    n: median_iqr_ci(ig_decoy_by_ch[n], **ci_kw) for n in node_feature_names
                },
            }

    print(
        f"{'class':<8}{'n':>4}{'reg_nodes':>10}  {'IG lift':>9}{'IG 95% CI':>16}"
        f"{'IG decoy':>10}{'rand':>7}{'attn*':>8}   verdict"
    )
    gate_pass = True
    for cls_name, r in results.items():
        ig = r["ig_in_region"]["median"]
        ci_lo, ci_hi = r["ig_in_region"]["ci_lo"], r["ig_in_region"]["ci_hi"]
        at = r["attn_in_region"]["median"]
        dc = r["ig_decoy_region"]["median"]
        rnd = r["random_in_region"]["median"]
        rn = r["region_nodes"]["median"]
        ok = class_alignment_ok(ig, dc)
        gate_pass = gate_pass and ok
        print(
            f"{cls_name:<8}{r['n_positives']:>4}{rn:>10.1f}  {ig:>9.2f}"
            f"{f'[{ci_lo:.2f},{ci_hi:.2f}]':>16}{dc:>10.2f}"
            f"{rnd:>7.2f}{at:>8.2f}   {'ALIGNED' if ok else 'WEAK'}"
        )
    print(
        f"\nGATE: {'PASS' if gate_pass else 'REVIEW'} "
        f"(in-region IG lift > 1 and > decoy for all morphology classes)"
    )
    print("* attn lift is beat-level and cannot resolve within-beat windows; IG is primary.")

    channel_hdr = "".join(f"{n[:10]:>12}" for n in node_feature_names)
    print(f"\n{'PER-CHANNEL in-region IG lift (median)':<40}")
    print(f"{'class':<8}{channel_hdr}   (decoy per channel)")
    for cls_name, r in results.items():
        in_ch = r["ig_in_region_by_channel"]
        dc_ch = r["ig_decoy_region_by_channel"]
        in_str = "".join(f"{in_ch[n]['median']:>12.2f}" for n in node_feature_names)
        dc_str = ", ".join(f"{n[:8]}={dc_ch[n]['median']:.2f}" for n in node_feature_names)
        print(f"{cls_name:<8}{in_str}   [{dc_str}]")

    baseline_suffix = "" if args.ig_baseline == "zero" else f"__{args.ig_baseline}"
    out_path = args.output or (
        Path("reports/clinical_alignment_gate") / f"{args.ckpt.stem}{baseline_suffix}.json"
    )
    parameters = {
        "device": args.device,
        "ig_steps": args.ig_steps,
        "ig_baseline": args.ig_baseline,
        "baseline_dataset": (
            str(args.baseline_dataset.resolve()) if args.baseline_dataset is not None else None
        ),
        "baseline_sample_nodes": args.baseline_sample_nodes,
        "baseline_seed": args.baseline_seed,
        "empirical_feature_median": (
            empirical_median.tolist() if empirical_median is not None else None
        ),
        "max_exams_per_class": args.max_exams_per_class,
        "seed": args.seed,
        "n_boot": args.n_boot,
        "alpha": args.alpha,
    }
    payload = {
        "gate_pass": gate_pass,
        "results": {
            cls_name: {
                "n_positives": r["n_positives"],
                "ig_in_region": _sanitize_stats(r["ig_in_region"]),
                "attn_in_region": _sanitize_stats(r["attn_in_region"]),
                "ig_decoy_region": _sanitize_stats(r["ig_decoy_region"]),
                "random_in_region": _sanitize_stats(r["random_in_region"]),
                "region_nodes": _sanitize_stats(r["region_nodes"]),
                "ig_in_region_by_channel": {
                    ch: _sanitize_stats(v) for ch, v in r["ig_in_region_by_channel"].items()
                },
                "ig_decoy_region_by_channel": {
                    ch: _sanitize_stats(v) for ch, v in r["ig_decoy_region_by_channel"].items()
                },
            }
            for cls_name, r in results.items()
        },
    }
    report = build_analysis_report(
        script=SCRIPT_NAME, provenance=provenance, parameters=parameters, payload=payload
    )
    write_analysis_report(out_path, report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
