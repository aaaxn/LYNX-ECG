"""Run the all-exam Prominence eight-state Shapley for one trained seed.

The report covers every CODE-Test exam under both availability rules
(``beat_complete`` and ``per_region``) and stores per-class raw states and
exact Shapley allocations. One invocation analyzes one checkpoint.

Usage:
    uv run python scripts/explain/prominence_availability_shapley.py \
        --run 55=results/code100_tp_s55/best_f1_val.pt \
        --dataset graph_datasets/codetest/DATASET.h5 \
        --delineation-qc reports/region_delineation_qc/codetest_region_delineation_qc.json \
        --device cuda \
        --output reports/prominence_availability_shapley/s55.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from cli.run_arguments import seed_path_type
from ecg_artifact_io import sha256_file
from explain.delineation_workflow import load_delineation_qc_report
from explain.prominence_availability_workflow import (
    analyze_prominence_availability_seed,
    write_prominence_availability_report,
)
from training.inference import load_checkpoint_and_dataset, resolve_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all eight PR-QRS-T deletion states for the selected Prominence "
            "candidate over every dataset exam, under both the beat-complete and the "
            "per-region availability rule, reading all three morphology-class outputs "
            "from each forward pass."
        )
    )
    parser.add_argument(
        "--run",
        required=True,
        type=seed_path_type("SEED=CHECKPOINT"),
        metavar="SEED=CHECKPOINT",
        help="one independently trained checkpoint",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--delineation-qc", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-symmetrized-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataset_path = args.dataset.resolve()
    delineation_qc_path = args.delineation_qc.resolve()
    delineation_qc_report_sha256 = sha256_file(delineation_qc_path)
    dataset_hdf5_sha256 = sha256_file(dataset_path)
    delineation = load_delineation_qc_report(delineation_qc_path)
    device = resolve_device(args.device)
    try:
        bundle = load_checkpoint_and_dataset(
            args.run.path.resolve(),
            dataset_path,
            device,
            allow_symmetrized=args.allow_symmetrized_checkpoint,
        )
    except ValueError as exc:
        raise SystemExit(
            f"incompatible checkpoint/dataset for seed {args.run.seed}: {exc}"
        ) from exc
    with bundle, h5py.File(dataset_path, "r") as h5:
        report = analyze_prominence_availability_seed(
            bundle.model,
            bundle.dataset,
            h5,
            delineation,
            delineation_qc_path=delineation_qc_path,
            delineation_qc_report_sha256=delineation_qc_report_sha256,
            checkpoint_path=args.run.path.resolve(),
            checkpoint_sha256=bundle.checkpoint_sha256,
            checkpoint_state=bundle.state,
            dataset_meta=bundle.meta,
            dataset_hdf5_sha256=dataset_hdf5_sha256,
            seed=args.run.seed,
            device=device,
            show_progress=args.progress,
        )
    write_prominence_availability_report(args.output, report, overwrite=args.overwrite)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
