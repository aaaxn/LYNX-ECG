"""Run exact PR-QRS-T 2^3 factorial analysis on Prominence-only support.

The selected Prominence candidate alone defines lead-beat availability. Fixed
legacy, fixed standardized, and Prominence are evaluated on that same mask;
DWT is not used.

Usage:
    uv run python scripts/explain/prominence_support_factorial.py \
        --run 55555=results/code100_tp_s55555/best_f1_val.pt \
        --run 55=results/code100_tp_s55/best_f1_val.pt \
        --run 1111=results/code100_tp_s1111/best_f1_val.pt \
        --dataset graph_datasets/codetest/DATASET.h5 \
        --delineation-qc reports/region_delineation_qc/codetest_region_delineation_qc.json \
        --device cuda \
        --output reports/prominence_support_factorial/codetest.json
"""

from __future__ import annotations

import argparse
import csv
import io
import tempfile
from pathlib import Path

import h5py

from cli.run_arguments import SeedPathArgument, seed_path_type
from ecg_artifact_io import (
    preflight_artifact_package,
    publish_artifact_package,
    sha256_file,
    staged_artifact,
)
from explain.delineation_workflow import load_delineation_qc_report
from explain.prominence_factorial_workflow import (
    ProminenceFactorialSeedRun,
    ProminenceSupportFactorialReport,
    analyze_prominence_factorial_seed,
    build_prominence_factorial_report,
    write_prominence_factorial_report,
)
from training.inference import load_checkpoint_and_dataset, resolve_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all eight PR-QRS-T states for fixed legacy, fixed standardized, "
            "and Prominence on support defined only by complete selected-Prominence "
            "triplets. DWT is not used. The report includes factorial effects, paired "
            "method contrasts, and paired segment main-effect contrasts with the "
            "minimum symmetric margin containing each pointwise 95% interval."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=seed_path_type("SEED=CHECKPOINT"),
        metavar="SEED=CHECKPOINT",
        help="repeat exactly three times for independently trained checkpoints",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--delineation-qc", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--allow-symmetrized-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _evaluate_run(
    argument: SeedPathArgument,
    *,
    dataset_path: Path,
    delineation_qc_path: Path,
    delineation_qc_report_sha256: str,
    dataset_hdf5_sha256: str,
    device_name: str,
    allow_symmetrized: bool,
) -> ProminenceFactorialSeedRun:
    checkpoint_path = argument.path.resolve()
    delineation = load_delineation_qc_report(delineation_qc_path)
    if sha256_file(delineation_qc_path) != delineation_qc_report_sha256:
        raise ValueError(f"delineation QC changed while loading seed {argument.seed}")
    device = resolve_device(device_name)
    try:
        bundle = load_checkpoint_and_dataset(
            checkpoint_path,
            dataset_path,
            device,
            allow_symmetrized=allow_symmetrized,
        )
    except ValueError as exc:
        raise SystemExit(
            f"incompatible checkpoint/dataset for seed {argument.seed}: {exc}"
        ) from exc
    with bundle, h5py.File(dataset_path, "r") as h5:
        return analyze_prominence_factorial_seed(
            bundle.model,
            bundle.dataset,
            h5,
            delineation,
            delineation_qc_path=delineation_qc_path,
            delineation_qc_report_sha256=delineation_qc_report_sha256,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=bundle.checkpoint_sha256,
            checkpoint_state=bundle.state,
            dataset_meta=bundle.meta,
            dataset_hdf5_sha256=dataset_hdf5_sha256,
            seed=argument.seed,
            device=device,
        )


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
    *,
    overwrite: bool,
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    with staged_artifact(path, overwrite=overwrite) as staged:
        staged.write_text(stream.getvalue())


def _write_tables(
    report: ProminenceSupportFactorialReport,
    *,
    summary_path: Path,
    method_contrast_path: Path,
    segment_contrast_path: Path,
) -> None:
    _write_csv(
        summary_path,
        [
            "class_name",
            "support_view",
            "method",
            "scale",
            "term",
            "n_exams",
            "mean",
            "median",
            "q1",
            "q3",
            "bootstrap_mean_ci_low",
            "bootstrap_mean_ci_high",
        ],
        [{**summary.model_dump(), "method": summary.method.value} for summary in report.summaries],
        overwrite=False,
    )
    _write_csv(
        method_contrast_path,
        [
            "class_name",
            "support_view",
            "method_a",
            "method_b",
            "scale",
            "term",
            "n_exams",
            "mean_difference_a_minus_b",
            "bootstrap_mean_difference_ci_low",
            "bootstrap_mean_difference_ci_high",
        ],
        [
            {
                **contrast.model_dump(),
                "method_a": contrast.method_a.value,
                "method_b": contrast.method_b.value,
            }
            for contrast in report.paired_method_contrasts
        ],
        overwrite=False,
    )
    _write_csv(
        segment_contrast_path,
        [
            "class_name",
            "support_view",
            "method",
            "scale",
            "segment_a",
            "segment_b",
            "n_exams",
            "mean_difference_a_minus_b",
            "bootstrap_mean_difference_ci_low",
            "bootstrap_mean_difference_ci_high",
            "minimum_symmetric_equivalence_margin",
        ],
        [
            {**contrast.model_dump(), "method": contrast.method.value}
            for contrast in report.segment_main_effect_contrasts
        ],
        overwrite=False,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.bootstrap_seed < 0:
        raise SystemExit("--bootstrap-seed must be non-negative")
    if not 100 <= args.bootstrap_resamples <= 100_000:
        raise SystemExit("--bootstrap-resamples must lie in [100, 100000]")
    if len(args.run) != 3 or len({argument.seed for argument in args.run}) != 3:
        raise SystemExit("exactly three --run values with unique seeds are required")

    output = args.output.resolve()
    if output.suffix != ".json":
        raise SystemExit("--output must have a .json suffix")
    dataset_path = args.dataset.resolve()
    delineation_qc_path = args.delineation_qc.resolve()
    summary_name = f"{output.stem}_summaries.csv"
    method_contrast_name = f"{output.stem}_method_contrasts.csv"
    segment_contrast_name = f"{output.stem}_segment_contrasts.csv"
    package_names = (
        output.name,
        summary_name,
        method_contrast_name,
        segment_contrast_name,
    )
    preflight_artifact_package(output.parent, package_names, overwrite=args.overwrite)

    delineation_qc_report_sha256 = sha256_file(delineation_qc_path)
    delineation = load_delineation_qc_report(delineation_qc_path)
    if delineation.dataset.path != str(dataset_path):
        raise SystemExit("--dataset differs from the dataset bound by --delineation-qc")
    dataset_hdf5_sha256 = sha256_file(dataset_path)
    runs = [
        _evaluate_run(
            argument,
            dataset_path=dataset_path,
            delineation_qc_path=delineation_qc_path,
            delineation_qc_report_sha256=delineation_qc_report_sha256,
            dataset_hdf5_sha256=dataset_hdf5_sha256,
            device_name=args.device,
            allow_symmetrized=args.allow_symmetrized_checkpoint,
        )
        for argument in args.run
    ]
    report = build_prominence_factorial_report(
        runs,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}-prominence-factorial-package-",
        dir=output.parent,
    ) as temporary_directory:
        staged_dir = Path(temporary_directory)
        staged_output = staged_dir / output.name
        staged_summary = staged_dir / summary_name
        staged_method_contrast = staged_dir / method_contrast_name
        staged_segment_contrast = staged_dir / segment_contrast_name
        write_prominence_factorial_report(staged_output, report)
        _write_tables(
            report,
            summary_path=staged_summary,
            method_contrast_path=staged_method_contrast,
            segment_contrast_path=staged_segment_contrast,
        )
        publish_artifact_package(
            staged_dir,
            output.parent,
            package_names,
            overwrite=args.overwrite,
        )

    print(f"wrote {output}")
    print(f"wrote {output.parent / summary_name}")
    print(f"wrote {output.parent / method_contrast_name}")
    print(f"wrote {output.parent / segment_contrast_name}")
    print(
        f"seeds={report.seed_order} summaries={len(report.summaries)} "
        f"method_contrasts={len(report.paired_method_contrasts)} "
        f"segment_contrasts={len(report.segment_main_effect_contrasts)}"
    )


if __name__ == "__main__":
    main()
