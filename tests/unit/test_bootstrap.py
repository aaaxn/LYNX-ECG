"""Unit tests for bootstrap confidence intervals (training.bootstrap)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from training.bootstrap import median_iqr_ci, paired_bootstrap_model_differences

pytestmark = pytest.mark.unit


def _score_mean_metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    del labels
    return {"score_mean": float(scores.mean())}


def test_paired_bootstrap_uses_identical_exam_resamples():
    rng = np.random.default_rng(20)
    labels = np.zeros((80, 1), dtype=np.int64)
    baseline = rng.normal(size=(2, 80, 1))
    ours = baseline + 0.25

    differences = paired_bootstrap_model_differences(
        labels,
        {"Ours": ours, "Baseline": baseline},
        _score_mean_metric,
        ["score_mean"],
        reference_model="Ours",
        n_boot=300,
        seed=4,
    )

    ci = differences["Baseline"]["score_mean"]
    assert ci.point == pytest.approx(0.25)
    assert ci.lo == pytest.approx(0.25)
    assert ci.hi == pytest.approx(0.25)
    assert ci.n_effective == 300


def test_paired_bootstrap_point_is_difference_between_seed_means():
    labels = np.zeros((4, 1), dtype=np.int64)
    scores = {
        "Ours": np.array([[[2.0]] * 4, [[4.0]] * 4]),
        "A": np.array([[[1.0]] * 4, [[2.0]] * 4]),
        "B": np.array([[[3.0]] * 4]),
    }

    differences = paired_bootstrap_model_differences(
        labels,
        scores,
        _score_mean_metric,
        ["score_mean"],
        reference_model="Ours",
        n_boot=20,
    )

    assert differences["A"]["score_mean"].point == pytest.approx(1.5)
    assert differences["B"]["score_mean"].point == pytest.approx(0.0)
    assert set(differences) == {"A", "B"}


def test_paired_bootstrap_validates_model_score_shapes_and_reference():
    labels = np.zeros((5, 2), dtype=np.int64)
    valid = np.zeros((1, 5, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="reference_model"):
        paired_bootstrap_model_differences(
            labels,
            {"Baseline": valid},
            _score_mean_metric,
            ["score_mean"],
            reference_model="Ours",
        )
    with pytest.raises(ValueError, match=r"\[seeds, exams, classes\]"):
        paired_bootstrap_model_differences(
            labels,
            {"Ours": valid, "Baseline": valid[0]},
            _score_mean_metric,
            ["score_mean"],
            reference_model="Ours",
        )
    with pytest.raises(ValueError, match="exam/class shape"):
        paired_bootstrap_model_differences(
            labels,
            {"Ours": valid, "Baseline": np.zeros((1, 4, 2))},
            _score_mean_metric,
            ["score_mean"],
            reference_model="Ours",
        )
    with pytest.raises(ValueError, match="did not return required keys"):
        paired_bootstrap_model_differences(
            labels,
            {"Ours": valid, "Baseline": valid},
            _score_mean_metric,
            ["missing"],
            reference_model="Ours",
            n_boot=2,
        )


def test_median_iqr_ci_fields_and_median():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = median_iqr_ci(values, n_boot=200, seed=0)
    assert set(r) == {"median", "q25", "q75", "ci_lo", "ci_hi", "n"}
    assert r["median"] == pytest.approx(3.0)
    assert r["q25"] == pytest.approx(2.0)
    assert r["q75"] == pytest.approx(4.0)
    assert r["ci_lo"] <= r["median"] <= r["ci_hi"]
    assert r["n"] == 5


def test_median_iqr_ci_drops_nonfinite():
    r = median_iqr_ci([1.0, np.nan, 2.0, np.inf, 3.0], n_boot=100, seed=0)
    assert r["n"] == 3
    assert r["median"] == pytest.approx(2.0)


def test_median_iqr_ci_empty_is_nan_with_zero_n():
    r = median_iqr_ci([np.nan, np.inf], n_boot=100, seed=0)
    assert r["n"] == 0
    assert all(math.isnan(r[k]) for k in ("median", "q25", "q75", "ci_lo", "ci_hi"))


def test_median_iqr_ci_reproducible_with_seed():
    values = np.random.default_rng(4).standard_normal(120)
    assert median_iqr_ci(values, n_boot=300, seed=9) == median_iqr_ci(values, n_boot=300, seed=9)


def test_median_iqr_ci_brackets_known_median_and_narrows_with_more_data():
    rng = np.random.default_rng(2)
    small = median_iqr_ci(rng.random(40), n_boot=1000, seed=0)
    big = median_iqr_ci(rng.random(4000), n_boot=1000, seed=0)
    assert big["ci_lo"] <= 0.5 <= big["ci_hi"]
    assert (big["ci_hi"] - big["ci_lo"]) < (small["ci_hi"] - small["ci_lo"])


def test_median_iqr_ci_validation_errors():
    with pytest.raises(ValueError, match="n_boot"):
        median_iqr_ci(np.arange(5.0), n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        median_iqr_ci(np.arange(5.0), alpha=0.0)
