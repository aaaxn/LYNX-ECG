"""Paper metrics and paired exam-bootstrap summaries across trained model seeds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from training.bootstrap import MetricCI, paired_bootstrap_model_differences
from training.metrics import PRIMARY_METRIC_KEYS as _PRIMARY_METRIC_KEYS

__all__ = [
    "paired_model_bootstrap",
    "paper_metric_values",
    "summarize_model_metrics",
]


def _validate_labels_and_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [exams, classes], got {labels.shape}")
    if labels.shape != probabilities.shape:
        raise ValueError(
            f"labels and probabilities must share shape, got "
            f"{labels.shape} vs {probabilities.shape}"
        )
    if labels.shape[0] == 0 or labels.shape[1] == 0:
        raise ValueError("labels and probabilities must be non-empty")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only binary 0/1 values")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return labels.astype(np.uint8, copy=False), probabilities


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def paper_metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute the three paper macro metrics directly from probabilities.

    This is the probability-input equivalent of :func:`training.metrics.compute_metrics`
    restricted to the reported F1@0.5, AUROC, and AUPRC keys. Classes without
    positive support are excluded from each macro, matching that module's
    contract while avoiding unrelated calibration work inside every bootstrap.
    """

    labels, probabilities = _validate_labels_and_probabilities(labels, probabilities)
    f1_values = np.full(labels.shape[1], np.nan, dtype=np.float64)
    auroc_values = np.full(labels.shape[1], np.nan, dtype=np.float64)
    auprc_values = np.full(labels.shape[1], np.nan, dtype=np.float64)
    predictions = probabilities >= 0.5
    for class_index in range(labels.shape[1]):
        target = labels[:, class_index]
        positive_count = int(target.sum())
        if positive_count == 0:
            continue
        f1_values[class_index] = f1_score(
            target,
            predictions[:, class_index],
            zero_division=0,
        )
        auprc_values[class_index] = average_precision_score(
            target,
            probabilities[:, class_index],
        )
        if positive_count < len(target):
            auroc_values[class_index] = roc_auc_score(
                target,
                probabilities[:, class_index],
            )

    return {
        "f1/macro": _finite_mean(f1_values),
        "auroc/macro": _finite_mean(auroc_values),
        "pr_auc/macro": _finite_mean(auprc_values),
    }


def _validate_model_scores(
    labels: np.ndarray,
    model_scores: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    if not model_scores:
        raise ValueError("model_scores must be non-empty")
    validated: dict[str, np.ndarray] = {}
    for model, scores in model_scores.items():
        if not isinstance(model, str) or not model:
            raise ValueError("model names must be non-empty strings")
        array = np.asarray(scores, dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != labels.shape:
            raise ValueError(
                f"{model}: scores must have shape [seeds, {labels.shape[0]}, "
                f"{labels.shape[1]}], got {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError(f"{model}: scores must contain at least one trained seed")
        _validate_labels_and_probabilities(labels, array[0])
        if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
            raise ValueError(f"{model}: scores must be finite probabilities in [0, 1]")
        validated[model] = array
    return validated


def summarize_model_metrics(
    labels: np.ndarray,
    model_scores: Mapping[str, np.ndarray],
    *,
    seeds: Sequence[int],
) -> dict[str, dict[str, object]]:
    """Return per-seed values plus across-seed mean and sample SD per model."""

    validated = _validate_model_scores(labels, model_scores)
    seed_tuple = tuple(int(seed) for seed in seeds)
    summary: dict[str, dict[str, object]] = {}
    for model, scores in validated.items():
        if scores.shape[0] != len(seed_tuple):
            raise ValueError(
                f"{model}: score seed axis {scores.shape[0]} != seed count {len(seed_tuple)}"
            )
        per_seed = {
            str(seed): paper_metric_values(labels, scores[index])
            for index, seed in enumerate(seed_tuple)
        }
        summary[model] = {
            "per_seed": per_seed,
            "mean": {
                key: float(np.mean([metrics[key] for metrics in per_seed.values()]))
                for key in _PRIMARY_METRIC_KEYS
            },
            "sample_sd": {
                key: float(
                    np.std(
                        [metrics[key] for metrics in per_seed.values()],
                        ddof=1 if len(per_seed) > 1 else 0,
                    )
                )
                for key in _PRIMARY_METRIC_KEYS
            },
        }
    return summary


def paired_model_bootstrap(
    labels: np.ndarray,
    model_scores: Mapping[str, np.ndarray],
    *,
    reference_model: str,
    n_boot: int,
    alpha: float,
    seed: int,
) -> dict[str, dict[str, MetricCI]]:
    """Bootstrap reference-minus-comparator paper metrics on shared exams."""

    validated = _validate_model_scores(labels, model_scores)
    return paired_bootstrap_model_differences(
        labels,
        validated,
        paper_metric_values,
        _PRIMARY_METRIC_KEYS,
        reference_model=reference_model,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
