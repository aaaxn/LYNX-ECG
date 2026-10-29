"""Multi-label classification metrics: F1, AUROC, PR-AUC, Precision, Recall, Accuracy.

Both per-class and macro variants are returned. The macro is the simple mean
over classes that have at least one positive in the targets (AUROC requires
both classes); other classes contribute ``NaN`` per-class and are excluded
from the macro.

The metric dict keys follow ``<name>/<class>`` for per-class entries and
``<name>/macro`` for macro values.

Additional calibration and confusion metrics are also returned:

- ``confusion/<class>/tp|fp|fn|tn``: threshold-based confusion counts;
- ``brier/<class>`` / ``brier/macro``: mean squared error of predicted
  probabilities vs. binary targets (lower is better);
- ``ece/<class>`` / ``ece/macro``: expected calibration error (equal-width
  binning, 10 bins, one-vs-rest per class; lower is better).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dataset.labels import validate_label_matrix, validate_label_names

__all__ = ["PRIMARY_METRIC_KEYS", "compute_metrics"]

PRIMARY_METRIC_KEYS: tuple[str, ...] = ("f1/macro", "auroc/macro", "pr_auc/macro")

_ECE_BINS = 10


def _expected_calibration_error(
    y_true_col: np.ndarray,
    prob_col: np.ndarray,
    n_bins: int,
) -> float:
    """Compute the expected calibration error for a single binary class.

    Uses equal-width binning over ``[0, 1]`` (``n_bins`` uniform intervals).
    For each non-empty bin the signed gap between mean predicted confidence
    and mean empirical accuracy is weighted by the fraction of samples in
    that bin.

    ``ECE = sum_b (n_b / N) * |acc_b - conf_b|``

    Empty bins contribute nothing to the sum (they have zero weight).

    Args:
        y_true_col: ``[N]`` binary 0/1 targets for one class.
        prob_col: ``[N]`` predicted probabilities for that class (in [0, 1]).
        n_bins: number of equal-width bins; normally ``_ECE_BINS``.

    Returns:
        ECE as a float in [0, 1], or ``float("nan")`` when ``prob_col`` is
        empty.
    """
    if prob_col.size == 0:
        return float("nan")

    n = prob_col.size
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(bin_edges):
        if hi == bin_edges[-1]:
            mask = (prob_col >= lo) & (prob_col <= hi)
        else:
            mask = (prob_col >= lo) & (prob_col < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        confidence = float(prob_col[mask].mean())
        accuracy = float(y_true_col[mask].mean())
        ece += (n_bin / n) * abs(accuracy - confidence)
    return ece


def _per_class(y_true: np.ndarray, y_pred: np.ndarray, score_fn, **kwargs) -> np.ndarray:
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for c in range(y_true.shape[1]):
        out[c] = score_fn(y_true[:, c], y_pred[:, c], **kwargs)
    return out


def _macro(per_class_values: np.ndarray) -> float:
    finite = per_class_values[np.isfinite(per_class_values)]
    return float(finite.mean()) if finite.size > 0 else float("nan")


def _nan_without_positive_support(values: np.ndarray, positive_support: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    out[~positive_support] = np.nan
    return out


def compute_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    *,
    class_names: Sequence[str],
    threshold: float | np.ndarray = 0.5,
) -> dict[str, float]:
    """Compute the full metric dict from binary targets and raw logits.

    Args:
        y_true: ``[N, C]`` 0/1 multi-hot targets.
        logits: ``[N, C]`` raw logits (no sigmoid applied yet).
        class_names: ``[C]`` human-readable labels for the per-class keys.
        threshold: scalar (default 0.5) or ``[C]`` per-class probability cutoff
            applied after sigmoid for F1 / Precision / Recall / Accuracy.

    Returns:
        Flat ``dict[str, float]`` with the following key families:

        * ``f1/<class>`` / ``f1/macro``: binary F1 at the applied threshold.
        * ``auroc/<class>`` / ``auroc/macro``: area under the ROC curve.
        * ``pr_auc/<class>`` / ``pr_auc/macro``: average precision (area
          under the precision-recall curve). Threshold-free like AUROC, but
          conditioned on the positive class, so it keeps discriminating
          between models when AUROC saturates under heavy class imbalance.
        * ``precision/<class>`` / ``precision/macro``: binary precision.
        * ``recall/<class>`` / ``recall/macro``: binary recall.
        * ``accuracy/<class>`` / ``accuracy/macro``: binary accuracy.
        * ``confusion/<class>/tp|fp|fn|tn``: confusion counts (as floats) at
          the applied threshold.  No macro key (counts are not averageable).
        * ``brier/<class>`` / ``brier/macro``: Brier score per class.
        * ``ece/<class>`` / ``ece/macro``: expected calibration error per
          class (equal-width binning, ``_ECE_BINS`` bins).

        For ``f1``, ``precision``, ``recall``, and ``accuracy``: classes with
        no positives in ``y_true`` yield ``NaN`` and are excluded from the
        macro.  For ``auroc`` and ``pr_auc``: same rule (both are undefined
        without positive samples).

        ``brier``, ``ece``, and ``confusion`` are **not** positive-support
        masked: all three are mathematically defined for a constant-zero class
        and provide meaningful signal (e.g. ECE reflects how well the model
        has learned to output low probabilities for always-negative classes).
    """
    try:
        y_true = np.asarray(y_true)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must be a rank-2 array") from exc
    try:
        logits = np.asarray(logits)
    except (TypeError, ValueError) as exc:
        raise ValueError("logits must be a rank-2 array") from exc
    if y_true.ndim != 2:
        raise ValueError(f"y_true must be rank 2, got shape {y_true.shape}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank 2, got shape {logits.shape}")
    if y_true.shape != logits.shape:
        raise ValueError(
            f"y_true and logits must share shape, got {y_true.shape} vs {logits.shape}"
        )
    class_names = validate_label_names(class_names, context="compute_metrics class names")
    if len(class_names) != y_true.shape[1]:
        raise ValueError(
            f"class_names length ({len(class_names)}) != num_classes ({y_true.shape[1]})"
        )
    y_true = validate_label_matrix(
        y_true,
        num_exams=y_true.shape[0],
        num_classes=y_true.shape[1],
        context="compute_metrics y_true",
    )

    positive_support = y_true.sum(axis=0) > 0

    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    try:
        if np.isscalar(threshold):
            thresholds = np.full(y_true.shape[1], float(threshold), dtype=np.float64)
        else:
            thresholds = np.asarray(threshold, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be numeric") from exc
    if thresholds.shape != (y_true.shape[1],):
        raise ValueError(
            f"per-class threshold must have shape ({y_true.shape[1]},), got {thresholds.shape}"
        )
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("threshold values must be finite")
    if not np.all((thresholds >= 0.0) & (thresholds <= 1.0)):
        raise ValueError("threshold values must be in [0, 1]")
    y_pred = (probs >= thresholds[None, :]).astype(np.int64)

    auroc_per_class = _per_class(y_true, probs, roc_auc_score)
    pr_auc_per_class = _per_class(y_true, probs, average_precision_score)
    f1_per_class = _per_class(y_true, y_pred, f1_score, zero_division=0)
    precision_per_class = _per_class(y_true, y_pred, precision_score, zero_division=0)
    recall_per_class = _per_class(y_true, y_pred, recall_score, zero_division=0)
    accuracy_per_class = _per_class(y_true, y_pred, accuracy_score)
    f1_per_class = _nan_without_positive_support(f1_per_class, positive_support)
    pr_auc_per_class = _nan_without_positive_support(pr_auc_per_class, positive_support)
    precision_per_class = _nan_without_positive_support(precision_per_class, positive_support)
    recall_per_class = _nan_without_positive_support(recall_per_class, positive_support)
    accuracy_per_class = _nan_without_positive_support(accuracy_per_class, positive_support)

    metrics: dict[str, float] = {}
    for name, vals in (
        ("f1", f1_per_class),
        ("auroc", auroc_per_class),
        ("pr_auc", pr_auc_per_class),
        ("precision", precision_per_class),
        ("recall", recall_per_class),
        ("accuracy", accuracy_per_class),
    ):
        for c, cls_name in enumerate(class_names):
            metrics[f"{name}/{cls_name}"] = float(vals[c])
        metrics[f"{name}/macro"] = _macro(vals)

    for c, cls_name in enumerate(class_names):
        tp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 1)).sum())
        fp = int(((y_pred[:, c] == 1) & (y_true[:, c] == 0)).sum())
        fn = int(((y_pred[:, c] == 0) & (y_true[:, c] == 1)).sum())
        tn = int(((y_pred[:, c] == 0) & (y_true[:, c] == 0)).sum())
        metrics[f"confusion/{cls_name}/tp"] = float(tp)
        metrics[f"confusion/{cls_name}/fp"] = float(fp)
        metrics[f"confusion/{cls_name}/fn"] = float(fn)
        metrics[f"confusion/{cls_name}/tn"] = float(tn)

    brier_per_class = np.array(
        [
            float(np.mean((probs[:, c] - y_true[:, c].astype(np.float64)) ** 2))
            for c in range(len(class_names))
        ],
        dtype=np.float64,
    )
    for c, cls_name in enumerate(class_names):
        metrics[f"brier/{cls_name}"] = float(brier_per_class[c])
    metrics["brier/macro"] = float(brier_per_class.mean())

    ece_per_class = np.array(
        [
            _expected_calibration_error(y_true[:, c].astype(np.float64), probs[:, c], _ECE_BINS)
            for c in range(len(class_names))
        ],
        dtype=np.float64,
    )
    for c, cls_name in enumerate(class_names):
        metrics[f"ece/{cls_name}"] = float(ece_per_class[c])
    metrics["ece/macro"] = _macro(ece_per_class)

    return metrics
