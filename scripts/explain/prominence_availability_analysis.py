"""Aggregate the three-seed all-exam Prominence availability Shapley reports.

The analysis emits the four result cells (availability rule x cohort), the
1dAVb takeover pattern, and negative-cohort specificity summaries. It reads
only the per-seed reports written by ``prominence_availability_shapley.py``.

Usage:
    uv run python scripts/explain/prominence_availability_analysis.py \
        --report reports/prominence_availability_shapley/s55.json \
        --report reports/prominence_availability_shapley/s1111.json \
        --report reports/prominence_availability_shapley/s55555.json \
        --output reports/prominence_availability_shapley/analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from ecg_artifact_io import staged_artifact
from explain.common_support import AvailabilityRule
from explain.ecg_taxonomy import EXPECTED_REGION, MORPHOLOGY_CLASSES, ECGRegion
from explain.prominence_availability_workflow import (
    AVAILABILITY_RULE_ORDER,
    DECISION_THRESHOLD,
    ProminenceAvailabilityShapleyReport,
    class_verdict,
    load_prominence_availability_report,
)

COHORTS = ("morphology_isolated", "all_positive")

PR_CLASS = next(name for name, region in EXPECTED_REGION.items() if region is ECGRegion.PR)


def _positive_cohort(exam_positive: list[str], class_name: str, cohort: str) -> bool:
    if class_name not in exam_positive:
        return False
    if cohort == "morphology_isolated":
        return len(exam_positive) == 1
    return True


def _verdicts(
    reports: list[ProminenceAvailabilityShapleyReport],
    *,
    rule: AvailabilityRule,
    cohort: str,
) -> dict[str, object]:
    per_class: dict[str, object] = {}
    takeover: Counter[str] = Counter()
    total_correct = 0
    total_leads_given_correct = 0
    for class_name in MORPHOLOGY_CLASSES:
        expected = EXPECTED_REGION[class_name]
        n_verdicts = 0
        n_correct = 0
        n_expected_leads = 0
        n_leads_given_correct = 0
        winners_by_exam: dict[int, list[str | None]] = defaultdict(list)
        for report in reports:
            for exam in report.exams:
                if not _positive_cohort(exam.positive_classes, class_name, cohort):
                    continue
                verdict = class_verdict(exam.rule_result(rule), class_name)
                n_verdicts += 1
                winners_by_exam[exam.exam_index].append(verdict.winner)
                if verdict.correct:
                    n_correct += 1
                if verdict.winner == expected:
                    n_expected_leads += 1
                    if verdict.correct:
                        n_leads_given_correct += 1
                elif class_name == PR_CLASS and verdict.winner is not None:
                    takeover[verdict.winner] += 1
        strict = sum(
            1
            for winners in winners_by_exam.values()
            if len(winners) == len(reports) and all(value == expected for value in winners)
        )
        total_correct += n_correct
        total_leads_given_correct += n_leads_given_correct
        per_class[class_name] = {
            "n_exams": len(winners_by_exam),
            "n_verdicts": n_verdicts,
            "classified_correctly": n_correct,
            "expected_region_leads": n_expected_leads,
            "expected_leads_given_correct": n_leads_given_correct,
            "strict_all_seeds_exams": strict,
        }
    return {
        "per_class": per_class,
        "all_leads_given_correct": total_leads_given_correct,
        "all_correct": total_correct,
        "takeover_1davb_when_pr_loses": dict(takeover),
    }


def _specificity(
    reports: list[ProminenceAvailabilityShapleyReport],
    *,
    rule: AvailabilityRule,
) -> dict[str, object]:
    """Negative-cohort behavior: intact scores and expected-region deletions."""
    result: dict[str, object] = {}
    cohorts = reports[0].label_defined_cohorts
    for class_name in MORPHOLOGY_CLASSES:
        expected = EXPECTED_REGION[class_name]
        for cohort_name, indices in (
            ("target_negative", set(cohorts.target_negative_by_target[class_name])),
            ("six_class_negative", set(cohorts.six_class_negative)),
        ):
            intact_probabilities: list[float] = []
            expected_contributions: list[float] = []
            false_positive_verdicts = 0
            n_verdicts = 0
            for report in reports:
                for exam in report.exams:
                    if exam.exam_index not in indices:
                        continue
                    verdict = class_verdict(exam.rule_result(rule), class_name)
                    n_verdicts += 1
                    if verdict.intact_probability is not None:
                        intact_probabilities.append(verdict.intact_probability)
                        if verdict.correct:
                            false_positive_verdicts += 1
                    if verdict.contributions is not None:
                        expected_contributions.append(verdict.contributions[expected])
            result[f"{class_name}/{cohort_name}"] = {
                "n_verdicts": n_verdicts,
                "false_positive_verdicts": false_positive_verdicts,
                "median_intact_probability": (
                    median(intact_probabilities) if intact_probabilities else None
                ),
                "median_expected_region_logit_shapley": (
                    median(expected_contributions) if expected_contributions else None
                ),
            }
    return result


def _print_cell(title: str, cell: dict[str, object]) -> None:
    print(f"\n=== {title}")
    per_class = cell["per_class"]
    assert isinstance(per_class, dict)
    for class_name, row in per_class.items():
        assert isinstance(row, dict)
        n_verdicts = row["n_verdicts"]
        correct = row["classified_correctly"]
        leads_correct = row["expected_leads_given_correct"]
        strict = row["strict_all_seeds_exams"]
        n_exams = row["n_exams"]
        print(
            f"  {class_name}: exams={n_exams} verdicts={n_verdicts} "
            f"correct={correct} leads|correct={leads_correct}/{correct} "
            f"strict={strict}/{n_exams}"
        )
    print(
        f"  ALL leads|correct = {cell['all_leads_given_correct']}/{cell['all_correct']}"
        f" | 1dAVb takeover: {cell['takeover_1davb_when_pr_loses']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    reports = [load_prominence_availability_report(path) for path in args.report]
    seeds = [report.seed for report in reports]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("seed reports must come from distinct training seeds")
    digests = {report.delineation_qc_report_sha256 for report in reports}
    if len(digests) != 1:
        raise SystemExit("seed reports must consume the same delineation QC artifact")

    analysis: dict[str, object] = {
        "producer": "prominence_availability_analysis",
        "seeds": seeds,
        "decision_threshold": DECISION_THRESHOLD,
        "expected_regions": EXPECTED_REGION,
        "source_reports": [str(path.resolve()) for path in args.report],
        "cells": {},
        "specificity": {},
    }
    cells = analysis["cells"]
    assert isinstance(cells, dict)
    for rule in AVAILABILITY_RULE_ORDER:
        for cohort in COHORTS:
            cell = _verdicts(reports, rule=rule, cohort=cohort)
            cells[f"{rule.value}/{cohort}"] = cell
            _print_cell(f"{rule.value} x {cohort}", cell)
    specificity = analysis["specificity"]
    assert isinstance(specificity, dict)
    for rule in AVAILABILITY_RULE_ORDER:
        specificity[rule.value] = _specificity(reports, rule=rule)

    with staged_artifact(args.output, overwrite=args.overwrite) as staged:
        staged.write_text(json.dumps(analysis, indent=1))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
