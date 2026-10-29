"""Contracts for strict paired-prediction cohorts and paper metric bootstraps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_paired_predictions import (
    combine_paired_prediction_cohorts,
    load_paired_prediction_cohort,
    write_paired_prediction_cohort,
)
from training.metrics import PRIMARY_METRIC_KEYS, compute_metrics
from training.paired_model_bootstrap import (
    paired_model_bootstrap,
    paper_metric_values,
    summarize_model_metrics,
)

pytestmark = pytest.mark.unit


def _sources(models: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "seed": seed,
            "kind": "test_prediction",
            "provenance": {"token": f"{model}-{seed}"},
        }
        for model in models
        for seed in seeds
    ]


def _write_cohort(
    path: Path,
    *,
    models: tuple[str, ...] = ("Baseline",),
    seeds: tuple[int, ...] = (7, 11),
    ids: tuple[str, ...] = ("e1", "e2", "e3", "e4"),
    probability_offset: float = 0.0,
):
    labels = ("A", "B")
    y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.uint8)
    base = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8], [0.8, 0.7], [0.1, 0.2]],
            [[0.8, 0.2], [0.1, 0.9], [0.7, 0.8], [0.2, 0.1]],
        ],
        dtype=np.float64,
    )
    y_prob = np.stack([base + probability_offset for _model in models], axis=0)
    return write_paired_prediction_cohort(
        path,
        dataset="Tiny",
        ids=ids,
        labels=labels,
        y_true=y_true,
        models=models,
        seeds=seeds,
        y_prob=y_prob,
        sources=_sources(models, seeds),
    )


def test_paired_prediction_cohort_round_trip_preserves_identity(tmp_path: Path) -> None:
    path = tmp_path / "paired.npz"
    written = _write_cohort(path)
    loaded = load_paired_prediction_cohort(path)

    assert loaded.dataset == written.dataset == "Tiny"
    assert loaded.models == ("Baseline",)
    assert loaded.seeds == (7, 11)
    np.testing.assert_array_equal(loaded.ids, ["e1", "e2", "e3", "e4"])
    np.testing.assert_array_equal(loaded.y_true, written.y_true)
    np.testing.assert_allclose(loaded.y_prob, written.y_prob)
    assert [(source["model"], source["seed"]) for source in loaded.sources] == [
        ("Baseline", 7),
        ("Baseline", 11),
    ]


def test_paired_prediction_cohort_rejects_incomplete_source_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cartesian product"):
        write_paired_prediction_cohort(
            tmp_path / "bad.npz",
            dataset="Tiny",
            ids=("e1",),
            labels=("A",),
            y_true=np.array([[1]], dtype=np.uint8),
            models=("M",),
            seeds=(1, 2),
            y_prob=np.array([[[[0.8]], [[0.7]]]]),
            sources=_sources(("M",), (1,)),
        )


def test_paired_prediction_loader_rejects_metadata_array_drift(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.npz"
    _write_cohort(valid_path)
    with np.load(valid_path, allow_pickle=False) as archive:
        members = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = json.loads(str(members["metadata_json"].item()))
    metadata["seeds"] = [11, 7]
    members["metadata_json"] = np.asarray(json.dumps(metadata), dtype=np.str_)
    corrupt_path = tmp_path / "corrupt.npz"
    with corrupt_path.open("wb") as stream:
        np.savez(stream, **members)

    with pytest.raises(ValueError, match=r"metadata\.seeds does not match"):
        load_paired_prediction_cohort(corrupt_path)


def test_combining_cohorts_requires_exact_exam_order_and_disjoint_models(
    tmp_path: Path,
) -> None:
    baseline = _write_cohort(tmp_path / "baseline.npz")
    ours = _write_cohort(
        tmp_path / "ours.npz",
        models=("Ours",),
        probability_offset=-0.02,
    )
    combined = combine_paired_prediction_cohorts((baseline, ours))

    assert combined.models == ("Baseline", "Ours")
    assert combined.y_prob.shape == (2, 2, 4, 2)

    reordered = _write_cohort(
        tmp_path / "reordered.npz",
        models=("Other",),
        ids=("e2", "e1", "e3", "e4"),
    )
    with pytest.raises(ValueError, match="exam ID order differs"):
        combine_paired_prediction_cohorts((baseline, reordered))
    with pytest.raises(ValueError, match="duplicate models"):
        combine_paired_prediction_cohorts((baseline, baseline))


def test_probability_metrics_match_full_metric_contract() -> None:
    rng = np.random.default_rng(19)
    labels = (rng.random((200, 3)) < np.array([0.2, 0.4, 0.6])).astype(np.uint8)
    probabilities = rng.uniform(0.01, 0.99, size=labels.shape)
    logits = np.log(probabilities / (1.0 - probabilities))

    expected = compute_metrics(labels, logits, class_names=("A", "B", "C"), threshold=0.5)
    observed = paper_metric_values(labels, probabilities)

    for key in PRIMARY_METRIC_KEYS:
        assert observed[key] == pytest.approx(expected[key], abs=1e-12)


def test_paired_bootstrap_reports_seed_mean_difference() -> None:
    labels = np.array([[0], [0], [1], [1]], dtype=np.uint8)
    ours = np.array(
        [
            [[0.1], [0.2], [0.8], [0.9]],
            [[0.2], [0.3], [0.7], [0.8]],
        ]
    )
    baseline = np.full((2, 4, 1), 0.5)
    scores = {"Ours": ours, "Baseline": baseline}

    summary = summarize_model_metrics(labels, scores, seeds=(7, 11))
    intervals = paired_model_bootstrap(
        labels,
        scores,
        reference_model="Ours",
        n_boot=50,
        alpha=0.05,
        seed=3,
    )

    expected_delta = summary["Ours"]["mean"]["f1/macro"] - summary["Baseline"]["mean"]["f1/macro"]
    assert intervals["Baseline"]["f1/macro"].point == pytest.approx(expected_delta)
    assert 0 < intervals["Baseline"]["f1/macro"].n_effective <= 50
