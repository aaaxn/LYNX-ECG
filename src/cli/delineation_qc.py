"""Standalone, model-output-blind PR-QRS-T delineation QC CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from explain.delineation_viz import write_delineation_review_pdf
from explain.delineation_workflow import (
    build_delineation_qc_report,
    load_delineation_qc_config,
    write_delineation_qc_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="delineation-qc",
        description=(
            "Compare NeuroKit DWT and prominence delineation on stored graph R anchors "
            "without reading model outputs. DWT keeps the original signal; prominence "
            "uses the configured negative-polarity inversion and preselected parameters."
        ),
        epilog=(
            "Common support is the set of lead-beat units with complete PR/QRS/T "
            "delineation from both DWT and the selected prominence method. Outputs are "
            "a strict QC JSON and a secondary review PDF."
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="validated QC JSON report")
    parser.add_argument(
        "--review-pdf",
        type=Path,
        required=True,
        help="labeled 12-lead protocol and prominence-candidate comparison package",
    )
    parser.add_argument(
        "--exam-scope",
        choices=("morphology_positive", "label_positive", "all_exams"),
        default="morphology_positive",
        help=(
            "analyze morphology-positive exams only (default), exams positive for any "
            "of the six labels, or every exam for label-defined positive and negative "
            "control cohorts"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.overwrite:
        existing = [path for path in (args.output, args.review_pdf) if path.exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite existing artifact(s): {existing}")

    config = load_delineation_qc_config(args.config)
    report = build_delineation_qc_report(
        args.dataset,
        config,
        exam_scope=args.exam_scope,
        show_progress=True,
    )
    pages = write_delineation_review_pdf(
        args.dataset,
        report,
        args.review_pdf,
        overwrite=args.overwrite,
    )
    write_delineation_qc_report(args.output, report, overwrite=args.overwrite)

    print(f"scope: {report.exam_scope}")
    print(f"report: {args.output}")
    print(f"review package: {args.review_pdf} ({pages} pages)")
    for summary in report.method_summaries:
        print(
            f"{summary.method.value}/{summary.candidate}: "
            f"{summary.overall.n_complete}/{summary.overall.n_units} "
            f"({summary.overall.completeness_rate:.2%})"
        )
    support = report.common_support.overall
    print(f"common support: {support.n_common}/{support.n_units} ({support.common_rate:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
