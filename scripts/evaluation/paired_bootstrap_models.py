"""Compare model families with shared-exam bootstrap resamples across seeds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ecg_artifact_io import sha256_file, staged_artifact
from ecg_paired_predictions import (
    combine_paired_prediction_cohorts,
    load_paired_prediction_cohort,
)
from training.metrics import PRIMARY_METRIC_KEYS
from training.paired_model_bootstrap import paired_model_bootstrap, summarize_model_metrics


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _alpha(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1)")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine exact prediction cohorts and bootstrap the difference "
            "between one reference model and every comparator."
        )
    )
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        type=Path,
        help="Paired-prediction NPZ; repeat to combine disjoint model families.",
    )
    parser.add_argument("--reference", required=True, help="Reference model name.")
    parser.add_argument("--n-bootstrap", type=_positive_int, default=2000)
    parser.add_argument("--alpha", type=_alpha, default=0.05)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    loaded = [load_paired_prediction_cohort(path) for path in args.cohort]
    cohort = combine_paired_prediction_cohorts(loaded)
    scores = cohort.scores_by_model()
    point_summary = summarize_model_metrics(cohort.y_true, scores, seeds=cohort.seeds)
    intervals = paired_model_bootstrap(
        cohort.y_true,
        scores,
        reference_model=args.reference,
        n_boot=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
    )

    report = {
        "dataset": cohort.dataset,
        "n_exams": len(cohort.ids),
        "labels": list(cohort.labels),
        "models": list(cohort.models),
        "seeds": list(cohort.seeds),
        "reference_model": args.reference,
        "estimand": "mean_seed(reference metric) - mean_seed(comparator metric)",
        "metric_keys": list(PRIMARY_METRIC_KEYS),
        "bootstrap": {
            "n_resamples": args.n_bootstrap,
            "alpha": args.alpha,
            "seed": args.seed,
            "resampling_unit": "exam",
            "seed_resampling": False,
        },
        "model_metrics": point_summary,
        "paired_differences": {
            comparator: {key: asdict(interval) for key, interval in comparator_intervals.items()}
            for comparator, comparator_intervals in intervals.items()
        },
        "source_cohorts": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "models": list(source.models),
                "sources": list(source.sources),
            }
            for path, source in zip(args.cohort, loaded, strict=True)
        ],
    }
    serialized = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with staged_artifact(args.output, overwrite=args.overwrite) as staged_path:
        staged_path.write_text(serialized)
        json.loads(staged_path.read_text())

    print(f"dataset={cohort.dataset} exams={len(cohort.ids)} seeds={list(cohort.seeds)}")
    for model in cohort.models:
        mean = point_summary[model]["mean"]
        print(
            f"{model}: F1={mean['f1/macro']:.6f} "
            f"AUROC={mean['auroc/macro']:.6f} AUPRC={mean['pr_auc/macro']:.6f}"
        )
    for comparator, comparator_intervals in intervals.items():
        values = " ".join(
            f"{key}={interval.point:+.6f}[{interval.lo:+.6f},{interval.hi:+.6f}]"
            for key, interval in comparator_intervals.items()
        )
        print(f"{args.reference} - {comparator}: {values}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
