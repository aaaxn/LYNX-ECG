"""Integration tests for the multilabel metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from training.metrics import compute_metrics

pytestmark = pytest.mark.integration


def _logits(probs: np.ndarray) -> np.ndarray:
    return np.log(probs / (1.0 - probs))


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), -0.1, 1.1, np.array([0.2, np.nan]), np.array([0.2])],
)
def test_compute_metrics_rejects_invalid_thresholds(threshold):
    y_true = np.array([[0, 1], [1, 0]])
    logits = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="threshold"):
        compute_metrics(y_true, logits, class_names=["A", "B"], threshold=threshold)


def test_compute_metrics_matches_sklearn():
    rng = np.random.default_rng(seed=42)
    n, c = 200, 3
    y_true = rng.integers(0, 2, size=(n, c))
    probs = rng.uniform(0.05, 0.95, size=(n, c))
    logits = _logits(probs)
    class_names = ["A", "B", "C"]

    metrics = compute_metrics(y_true, logits, class_names=class_names, threshold=0.5)
    y_pred = (probs >= 0.5).astype(int)

    for i, cls in enumerate(class_names):
        assert np.isclose(
            metrics[f"f1/{cls}"], f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        )
        assert np.isclose(metrics[f"auroc/{cls}"], roc_auc_score(y_true[:, i], probs[:, i]))
        assert np.isclose(
            metrics[f"pr_auc/{cls}"], average_precision_score(y_true[:, i], probs[:, i])
        )


def test_compute_metrics_per_class_threshold():
    rng = np.random.default_rng(seed=1)
    n, c = 50, 2
    y_true = rng.integers(0, 2, size=(n, c))
    probs = rng.uniform(0.05, 0.95, size=(n, c))
    logits = _logits(probs)

    threshold = np.array([0.3, 0.7])
    metrics = compute_metrics(y_true, logits, class_names=["X", "Y"], threshold=threshold)
    y_pred = (probs >= threshold[None, :]).astype(int)
    assert np.isclose(metrics["f1/X"], f1_score(y_true[:, 0], y_pred[:, 0], zero_division=0))
    assert np.isclose(metrics["f1/Y"], f1_score(y_true[:, 1], y_pred[:, 1], zero_division=0))


def test_compute_metrics_handles_constant_class():
    """A class with only one label value: AUROC is NaN, macro skips it."""
    y_true = np.zeros((20, 2), dtype=int)
    y_true[:, 0] = np.arange(20) % 2
    logits = np.random.RandomState(0).normal(size=(20, 2))
    metrics = compute_metrics(y_true, logits, class_names=["good", "bad"])
    assert np.isnan(metrics["auroc/bad"])
    assert np.isfinite(metrics["auroc/macro"])


def test_compute_metrics_excludes_absent_positive_classes_from_macro_metrics():
    """Classes with no positives are unmeasurable and excluded from macro metrics."""
    y_true = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=int,
    )
    probs = np.array(
        [
            [0.90, 0.20, 0.80],
            [0.40, 0.70, 0.60],
            [0.80, 0.60, 0.70],
            [0.20, 0.30, 0.40],
            [0.55, 0.60, 0.90],
            [0.45, 0.80, 0.20],
        ]
    )
    logits = _logits(probs)
    metrics = compute_metrics(y_true, logits, class_names=["alpha", "beta", "absent"])
    y_pred = (probs >= 0.5).astype(int)
    supported = [0, 1]

    assert np.isnan(metrics["f1/absent"])
    assert np.isnan(metrics["precision/absent"])
    assert np.isnan(metrics["recall/absent"])
    assert np.isnan(metrics["accuracy/absent"])

    f1_expected = [f1_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in supported]
    precision_expected = [
        precision_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in supported
    ]
    recall_expected = [recall_score(y_true[:, i], y_pred[:, i], zero_division=0) for i in supported]
    accuracy_expected = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in supported]

    assert np.isclose(metrics["f1/macro"], np.mean(f1_expected))
    assert np.isclose(metrics["precision/macro"], np.mean(precision_expected))
    assert np.isclose(metrics["recall/macro"], np.mean(recall_expected))
    assert np.isclose(metrics["accuracy/macro"], np.mean(accuracy_expected))
    assert np.isnan(metrics["auroc/absent"])
    assert np.isfinite(metrics["auroc/macro"])
    assert np.isnan(metrics["pr_auc/absent"])
    assert np.isfinite(metrics["pr_auc/macro"])


def test_pr_auc_zero_positive_class_does_not_drag_macro():
    """sklearn returns AP=0.0 (not NaN) for a zero-positive column; the
    positive-support mask must exclude it so pr_auc/macro equals the mean over
    measurable classes only."""
    rng = np.random.default_rng(7)
    n = 100
    y_true = np.zeros((n, 2), dtype=np.int64)
    y_true[:, 0] = rng.integers(0, 2, size=n)
    probs = rng.uniform(0.05, 0.95, size=(n, 2))
    logits = _logits(probs)

    metrics = compute_metrics(y_true, logits, class_names=["seen", "empty"])
    expected = average_precision_score(y_true[:, 0], probs[:, 0])
    assert np.isnan(metrics["pr_auc/empty"])
    assert np.isclose(metrics["pr_auc/macro"], expected)


@pytest.mark.parametrize(
    ("y_true", "logits", "message"),
    [
        (np.array([0, 1]), np.zeros((2, 1)), "y_true.*rank 2"),
        (np.zeros((2, 1)), np.zeros(2), "logits.*rank 2"),
    ],
)
def test_compute_metrics_requires_rank_two_inputs(y_true, logits, message):
    with pytest.raises(ValueError, match=message):
        compute_metrics(y_true, logits, class_names=["A"])


def test_compute_metrics_rejects_empty_or_duplicate_class_names():
    y_true = np.array([[0], [1]])
    logits = np.zeros((2, 1))

    with pytest.raises(ValueError, match="at least one"):
        compute_metrics(y_true, logits, class_names=[])
    with pytest.raises(ValueError, match="unique"):
        compute_metrics(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            class_names=["A", "A"],
        )


def test_compute_metrics_rejects_non_binary_targets():
    """Regression: soft / mis-scaled y_true must fail loudly rather than
    silently produce NaN or wrong scores."""
    logits = np.zeros((4, 2))
    with pytest.raises(ValueError, match="binary"):
        compute_metrics(
            np.array([[0.1, 0.9], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
            logits,
            class_names=["a", "b"],
        )
    with pytest.raises(ValueError, match="binary"):
        compute_metrics(
            np.array([[0, 2], [1, 0], [0, 1], [1, 1]]),
            logits,
            class_names=["a", "b"],
        )


def test_metrics_macro_invariant():
    """All-zero predictions: per-class F1 == 0, macro == 0."""
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    logits = np.full((3, 2), -10.0)
    metrics = compute_metrics(y_true, logits, class_names=["a", "b"])
    assert metrics["f1/a"] == 0.0
    assert metrics["f1/b"] == 0.0
    assert metrics["f1/macro"] == 0.0


def test_confusion_counts_match_manual():
    """Confusion counts derived from compute_metrics must equal hand-computed values."""
    y_true = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=np.int64)
    logits = np.array(
        [
            [5.0, -5.0],
            [5.0, -5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
        ],
        dtype=np.float64,
    )
    metrics = compute_metrics(y_true, logits, class_names=["A", "B"], threshold=0.5)

    assert metrics["confusion/A/tp"] == 1.0
    assert metrics["confusion/A/fp"] == 1.0
    assert metrics["confusion/A/fn"] == 1.0
    assert metrics["confusion/A/tn"] == 1.0
    n = y_true.shape[0]
    assert metrics["confusion/A/tp"] + metrics["confusion/A/fp"] + metrics[
        "confusion/A/fn"
    ] + metrics["confusion/A/tn"] == float(n)

    assert metrics["confusion/B/tp"] == 2.0
    assert metrics["confusion/B/fp"] == 0.0
    assert metrics["confusion/B/fn"] == 0.0
    assert metrics["confusion/B/tn"] == 2.0
    assert metrics["confusion/B/tp"] + metrics["confusion/B/fp"] + metrics[
        "confusion/B/fn"
    ] + metrics["confusion/B/tn"] == float(n)


def test_brier_score_known_value():
    """Brier score must equal mean((prob - y)^2) for known prob/label pairs."""
    probs = np.array([[0.9, 0.7], [0.2, 0.3]])
    logits = np.log(probs / (1.0 - probs))
    y_true = np.array([[1, 0], [0, 1]], dtype=np.int64)

    metrics = compute_metrics(y_true, logits, class_names=["X", "Y"], threshold=0.5)

    expected_brier_x = float(np.mean((probs[:, 0] - y_true[:, 0]) ** 2))
    expected_brier_y = float(np.mean((probs[:, 1] - y_true[:, 1]) ** 2))
    assert metrics["brier/X"] == pytest.approx(expected_brier_x, rel=1e-6)
    assert metrics["brier/Y"] == pytest.approx(expected_brier_y, rel=1e-6)
    assert metrics["brier/macro"] == pytest.approx(
        (expected_brier_x + expected_brier_y) / 2.0, rel=1e-6
    )


def test_ece_perfectly_calibrated_is_low():
    """ECE must be near zero for perfectly calibrated predictions.

    A Bernoulli model is perfectly calibrated by construction: labels are drawn
    from Bernoulli(p) for each predicted probability p, so the empirical
    positive rate within any bin converges to the mean predicted probability in
    that bin.  With 5000 samples the ECE must stay well below 0.05.
    """
    rng = np.random.default_rng(42)
    n = 5000
    probs_col = rng.uniform(0.01, 0.99, size=n)
    y_col = rng.binomial(1, probs_col).astype(np.int64)

    logits = np.log(probs_col / (1.0 - probs_col)).reshape(-1, 1)
    y_true = y_col.reshape(-1, 1)

    metrics = compute_metrics(y_true, logits, class_names=["C"], threshold=0.5)

    assert metrics["ece/C"] < 0.05, f"ECE={metrics['ece/C']:.4f} unexpectedly high"


def test_ece_maximally_miscalibrated():
    """ECE must be near 1 for fully anti-calibrated predictions.

    When the model always outputs probability ~1 for the negative class,
    every sample lands in the highest bin with confidence≈0.99 but accuracy=0.
    """
    n = 200
    probs_col = np.full(n, 0.99)
    y_col = np.zeros(n, dtype=np.int64)

    logits = np.log(probs_col / (1.0 - probs_col)).reshape(-1, 1)
    y_true = y_col.reshape(-1, 1)

    metrics = compute_metrics(y_true, logits, class_names=["C"], threshold=0.5)

    assert metrics["ece/C"] > 0.90, f"ECE={metrics['ece/C']:.4f} unexpectedly low"
