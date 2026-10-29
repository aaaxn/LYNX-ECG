"""Nonparametric bootstrap confidence intervals.

The paired routine resamples exams with replacement, reuses every exam
resample across all model families, and reports the difference between their
mean metrics across trained seeds. The metric is supplied as a callable
``(labels, scores) -> dict[str, float]`` so the CI uses the *exact* metric
definitions reported in the paper tables (for example,
:func:`training.metrics.compute_metrics`), with no divergent metric
reimplementation.

:func:`median_iqr_ci` gives a percentile bootstrap CI on the median of a
per-item sample, such as a per-positive region-alignment lift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MetricCI",
    "median_iqr_ci",
    "paired_bootstrap_model_differences",
]


@dataclass
class MetricCI:
    """Point estimate and a percentile confidence interval for one metric.

    ``n_effective`` is the number of bootstrap resamples that yielded a finite
    value for this key. For a PER-CLASS key it falls below ``n_boot`` on
    resamples where that class lost all its positives (NaN, per the metric's
    positive-support rule). For a MACRO key it stays ~``n_boot`` because the
    macro is finite as long as *some* class has support, but then such a
    resample averages over only the surviving classes, so a macro CI implicitly
    assumes consistent per-class support. That assumption is exact for large
    test sets; for small sets with very rare classes a few resamples shift the
    macro denominator (a minor, disclosed approximation).
    """

    point: float
    lo: float
    hi: float
    n_effective: int


def paired_bootstrap_model_differences(
    labels: np.ndarray,
    model_scores: Mapping[str, np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    keys: Sequence[str],
    *,
    reference_model: str,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, MetricCI]]:
    """Paired percentile CIs for one reference model minus each comparator.

    ``model_scores`` maps model names to arrays shaped
    ``[trained_seeds, exams, classes]``. The number of trained seeds may differ
    between models, but every array must describe the same exams, in the same
    order, with the same class columns as ``labels``. For both the full-sample
    point and every bootstrap resample, ``metric_fn`` is evaluated separately
    on each trained seed and then averaged within model. The reported estimand
    is therefore

    ``mean_seed(metric(reference)) - mean_seed(metric(comparator))``.

    Every model and seed receives the identical resampled exam indices. This
    preserves within-exam pairing and isolates sampling uncertainty from the
    separately reported between-seed variation. Seeds themselves are not
    resampled, so the interval is conditional on the supplied trained runs.

    Returns:
        ``{comparator: {metric_key: MetricCI}}``. Positive values favor the
        reference model for metrics where larger is better.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [exams, classes], got {labels.shape}")
    if labels.shape[0] == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if reference_model not in model_scores:
        raise ValueError(f"reference_model {reference_model!r} is absent from model_scores")
    if len(model_scores) < 2:
        raise ValueError("model_scores must contain a reference and at least one comparator")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    metric_keys = tuple(keys)
    if not metric_keys:
        raise ValueError("keys must contain at least one metric key")
    if len(set(metric_keys)) != len(metric_keys):
        raise ValueError(f"keys contains duplicates: {metric_keys}")

    scores_by_model: dict[str, np.ndarray] = {}
    for model_name, scores in model_scores.items():
        array = np.asarray(scores)
        if array.ndim != 3:
            raise ValueError(
                f"model_scores[{model_name!r}] must have shape "
                f"[seeds, exams, classes], got {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError(f"model_scores[{model_name!r}] has no trained seeds")
        if array.shape[1:] != labels.shape:
            raise ValueError(
                f"model_scores[{model_name!r}] exam/class shape {array.shape[1:]} "
                f"does not match labels {labels.shape}"
            )
        scores_by_model[model_name] = array

    def mean_seed_metrics(
        sample_labels: np.ndarray,
        scores: np.ndarray,
        sample_indices: np.ndarray | None,
    ) -> dict[str, float]:
        values = {key: np.empty(scores.shape[0], dtype=np.float64) for key in metric_keys}
        for seed_idx, seed_scores in enumerate(scores):
            selected_scores = seed_scores if sample_indices is None else seed_scores[sample_indices]
            metrics = metric_fn(sample_labels, selected_scores)
            missing = [key for key in metric_keys if key not in metrics]
            if missing:
                raise ValueError(f"metric_fn did not return required keys: {missing}")
            for key in metric_keys:
                values[key][seed_idx] = metrics[key]
        return {key: float(np.mean(seed_values)) for key, seed_values in values.items()}

    point_by_model = {
        model_name: mean_seed_metrics(labels, scores, None)
        for model_name, scores in scores_by_model.items()
    }
    comparator_names = [name for name in scores_by_model if name != reference_model]
    sample_differences = {
        name: {key: np.full(n_boot, np.nan, dtype=np.float64) for key in metric_keys}
        for name in comparator_names
    }

    rng = np.random.default_rng(seed)
    n_exams = labels.shape[0]
    for boot_idx in range(n_boot):
        indices = rng.integers(0, n_exams, size=n_exams)
        sample_labels = labels[indices]
        sampled_metrics = {
            model_name: mean_seed_metrics(sample_labels, scores, indices)
            for model_name, scores in scores_by_model.items()
        }
        reference_metrics = sampled_metrics[reference_model]
        for comparator_name in comparator_names:
            comparator_metrics = sampled_metrics[comparator_name]
            for key in metric_keys:
                difference = reference_metrics[key] - comparator_metrics[key]
                if np.isfinite(difference):
                    sample_differences[comparator_name][key][boot_idx] = difference

    lo_pct, hi_pct = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    output: dict[str, dict[str, MetricCI]] = {}
    reference_point = point_by_model[reference_model]
    for comparator_name in comparator_names:
        comparator_point = point_by_model[comparator_name]
        output[comparator_name] = {}
        for key in metric_keys:
            finite = sample_differences[comparator_name][key]
            finite = finite[np.isfinite(finite)]
            point = reference_point[key] - comparator_point[key]
            if finite.size == 0:
                output[comparator_name][key] = MetricCI(
                    point=float(point),
                    lo=float("nan"),
                    hi=float("nan"),
                    n_effective=0,
                )
                continue
            lo, hi = np.percentile(finite, [lo_pct, hi_pct])
            output[comparator_name][key] = MetricCI(
                point=float(point),
                lo=float(lo),
                hi=float(hi),
                n_effective=int(finite.size),
            )
    return output


def median_iqr_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Median, IQR, and a percentile bootstrap CI on the median, as a flat dict.

    Returns ``{median, q25, q75, ci_lo, ci_hi, n}`` ready for JSON. The IQR
    (``q25``/``q75``) describes the spread of the per-item sample; ``ci_lo`` /
    ``ci_hi`` is the bootstrap uncertainty on the median *estimate* itself, which
    is what a small-``n`` claim rests on. Non-finite values are dropped; an empty
    or all-NaN sample yields all-NaN fields with ``n=0`` (so per-class metrics
    with no support report the gap instead of crashing).
    """
    a = np.asarray(values, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n": 0,
        }
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Draw all resamples at once and take the median of each row.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot_medians = np.median(a[idx], axis=1)
    lo, hi = np.percentile(boot_medians, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(a.size),
    }
