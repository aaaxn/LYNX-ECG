"""Strict, provenance-carrying prediction cohorts for paired model comparisons."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ecg_artifact_io import staged_artifact

__all__ = [
    "PairedPredictionCohort",
    "combine_paired_prediction_cohorts",
    "load_paired_prediction_cohort",
    "write_paired_prediction_cohort",
]

_MEMBERS = frozenset(("ids", "labels", "y_true", "models", "seeds", "y_prob", "metadata_json"))
_METADATA_KEYS = frozenset(("dataset", "models", "seeds", "sources"))
_SOURCE_KEYS = frozenset(("model", "seed", "kind", "provenance"))


@dataclass(frozen=True)
class PairedPredictionCohort:
    """Predictions from multiple models and seeds on one exact exam cohort.

    ``y_prob`` is shaped ``[models, seeds, exams, classes]``. Every model must
    provide every listed seed; this rectangular contract keeps the estimand
    explicit and prevents silent seed or exam dropping during comparison.
    """

    dataset: str
    ids: np.ndarray
    labels: tuple[str, ...]
    y_true: np.ndarray
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    y_prob: np.ndarray
    sources: tuple[dict[str, Any], ...]

    def scores_by_model(self) -> dict[str, np.ndarray]:
        """Return ``{model: [seeds, exams, classes]}`` without copying arrays."""

        return {model: self.y_prob[index] for index, model in enumerate(self.models)}


def _text_vector(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    text = array.astype(str)
    if any(not item for item in text.tolist()):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(text.tolist())) != len(text):
        raise ValueError(f"{name} must contain unique values")
    return text


def _seed_vector(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"seeds must be one-dimensional, got {array.shape}")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"seeds must use an integer dtype, got {array.dtype}")
    seeds = array.astype(np.int64, copy=False)
    if len(set(seeds.tolist())) != len(seeds):
        raise ValueError("seeds must contain unique values")
    return seeds


def _require_string_keys(value: object, *, name: str) -> None:
    """Reject non-string mapping keys anywhere beneath ``value``.

    ``json.dumps`` silently coerces non-string keys (``{1: x}`` becomes
    ``{"1": x}``), so a payload carrying one would not survive a write/reload
    round-trip unchanged. The persisted cohort would quietly disagree with the
    object handed in. Nested containers are walked because provenance blobs are
    embedded whole into the metadata JSON, so a bad key two levels down coerces
    exactly like a top-level one.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} must be a JSON object with string keys")
            _require_string_keys(item, name=f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_string_keys(item, name=f"{name}[{index}]")


def _json_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object with string keys")
    _require_string_keys(value, name=name)
    return dict(cast("dict[str, Any]", value))


def _validate_sources(
    sources: object,
    *,
    models: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(sources, list):
        raise ValueError("metadata.sources must be a JSON array")

    validated: list[dict[str, Any]] = []
    observed: list[tuple[str, int]] = []
    for index, raw in enumerate(sources):
        source = _json_object(raw, name=f"metadata.sources[{index}]")
        if set(source) != _SOURCE_KEYS:
            raise ValueError(
                f"metadata.sources[{index}] must have exactly keys {sorted(_SOURCE_KEYS)}"
            )
        model = source["model"]
        seed = source["seed"]
        kind = source["kind"]
        provenance = source["provenance"]
        if not isinstance(model, str) or model not in models:
            raise ValueError(f"metadata.sources[{index}].model is not a cohort model")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in seeds:
            raise ValueError(f"metadata.sources[{index}].seed is not a cohort seed")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"metadata.sources[{index}].kind must be a non-empty string")
        _json_object(provenance, name=f"metadata.sources[{index}].provenance")
        observed.append((model, seed))
        validated.append(source)

    expected = [(model, seed) for model in models for seed in seeds]
    if observed != expected:
        raise ValueError(
            "metadata.sources must contain the model-major Cartesian product "
            f"{expected}, got {observed}"
        )
    return tuple(validated)


def _validate_cohort(
    *,
    dataset: object,
    ids: object,
    labels: object,
    y_true: object,
    models: object,
    seeds: object,
    y_prob: object,
    sources: object,
) -> PairedPredictionCohort:
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset must be a non-empty string")

    ids_array = _text_vector(ids, name="ids")
    labels_array = _text_vector(labels, name="labels")
    models_array = _text_vector(models, name="models")
    seeds_array = _seed_vector(seeds)
    if not len(ids_array) or not len(labels_array) or not len(models_array) or not len(seeds_array):
        raise ValueError("ids, labels, models, and seeds must be non-empty")

    truth = np.asarray(y_true)
    expected_truth_shape = (len(ids_array), len(labels_array))
    if truth.shape != expected_truth_shape:
        raise ValueError(f"y_true must have shape {expected_truth_shape}, got {truth.shape}")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("y_true must contain only binary 0/1 values")
    truth = truth.astype(np.uint8, copy=False)

    probabilities = np.asarray(y_prob, dtype=np.float64)
    expected_probability_shape = (
        len(models_array),
        len(seeds_array),
        len(ids_array),
        len(labels_array),
    )
    if probabilities.shape != expected_probability_shape:
        raise ValueError(
            f"y_prob must have shape {expected_probability_shape}, got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("y_prob contains non-finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("y_prob must contain probabilities in [0, 1]")

    model_tuple = tuple(models_array.tolist())
    seed_tuple = tuple(int(seed) for seed in seeds_array.tolist())
    source_tuple = _validate_sources(sources, models=model_tuple, seeds=seed_tuple)
    return PairedPredictionCohort(
        dataset=dataset,
        ids=ids_array,
        labels=tuple(labels_array.tolist()),
        y_true=truth,
        models=model_tuple,
        seeds=seed_tuple,
        y_prob=probabilities,
        sources=source_tuple,
    )


def load_paired_prediction_cohort(path: Path) -> PairedPredictionCohort:
    """Load and fully validate a versioned paired-prediction artifact."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            members = frozenset(archive.files)
            if members != _MEMBERS:
                raise ValueError(
                    f"{path}: members must be exactly {sorted(_MEMBERS)}, got {sorted(members)}"
                )
            ids = np.asarray(archive["ids"])
            labels = np.asarray(archive["labels"])
            y_true = np.asarray(archive["y_true"])
            models = np.asarray(archive["models"])
            seeds = np.asarray(archive["seeds"])
            y_prob = np.asarray(archive["y_prob"])
            metadata_value = np.asarray(archive["metadata_json"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: invalid paired-prediction archive: {exc}") from exc

    if metadata_value.shape != () or metadata_value.dtype.kind not in "US":
        raise ValueError(f"{path}: metadata_json must be one scalar Unicode string")
    try:
        metadata = json.loads(str(metadata_value.item()))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: metadata_json is not valid JSON: {exc}") from exc
    metadata = _json_object(metadata, name=f"{path}: metadata_json")
    if set(metadata) != _METADATA_KEYS:
        raise ValueError(
            f"{path}: metadata keys must be exactly {sorted(_METADATA_KEYS)}, "
            f"got {sorted(metadata)}"
        )

    cohort = _validate_cohort(
        dataset=metadata["dataset"],
        ids=ids,
        labels=labels,
        y_true=y_true,
        models=models,
        seeds=seeds,
        y_prob=y_prob,
        sources=metadata["sources"],
    )
    if metadata["models"] != list(cohort.models):
        raise ValueError(f"{path}: metadata.models does not match models array")
    if metadata["seeds"] != list(cohort.seeds):
        raise ValueError(f"{path}: metadata.seeds does not match seeds array")
    return cohort


def write_paired_prediction_cohort(
    path: Path,
    *,
    dataset: str,
    ids: Sequence[object],
    labels: Sequence[str],
    y_true: np.ndarray,
    models: Sequence[str],
    seeds: Sequence[int],
    y_prob: np.ndarray,
    sources: Sequence[Mapping[str, Any]],
    overwrite: bool = False,
) -> PairedPredictionCohort:
    """Validate, atomically publish, and strict-reload a cohort artifact."""

    source_objects = [dict(source) for source in sources]
    cohort = _validate_cohort(
        dataset=dataset,
        ids=ids,
        labels=labels,
        y_true=y_true,
        models=models,
        seeds=seeds,
        y_prob=y_prob,
        sources=source_objects,
    )
    metadata = {
        "dataset": cohort.dataset,
        "models": list(cohort.models),
        "seeds": list(cohort.seeds),
        "sources": list(cohort.sources),
    }
    metadata_json = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

    path = Path(path)
    with staged_artifact(path, overwrite=overwrite) as staged_path:
        with staged_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                ids=cohort.ids,
                labels=np.asarray(cohort.labels, dtype=np.str_),
                y_true=cohort.y_true,
                models=np.asarray(cohort.models, dtype=np.str_),
                seeds=np.asarray(cohort.seeds, dtype=np.int64),
                y_prob=cohort.y_prob,
                metadata_json=np.asarray(metadata_json, dtype=np.str_),
            )
        load_paired_prediction_cohort(staged_path)
    return load_paired_prediction_cohort(path)


def combine_paired_prediction_cohorts(
    cohorts: Sequence[PairedPredictionCohort],
) -> PairedPredictionCohort:
    """Combine disjoint model families after exact cohort-identity checks."""

    if not cohorts:
        raise ValueError("at least one paired prediction cohort is required")
    reference = cohorts[0]
    models: list[str] = []
    sources: list[dict[str, Any]] = []
    probability_blocks: list[np.ndarray] = []
    for index, cohort in enumerate(cohorts):
        context = f"cohort {index}"
        if cohort.dataset != reference.dataset:
            raise ValueError(f"{context}: dataset differs from cohort 0")
        if cohort.labels != reference.labels:
            raise ValueError(f"{context}: label order differs from cohort 0")
        if cohort.seeds != reference.seeds:
            raise ValueError(f"{context}: seed order differs from cohort 0")
        if not np.array_equal(cohort.ids, reference.ids):
            raise ValueError(f"{context}: exam ID order differs from cohort 0")
        if not np.array_equal(cohort.y_true, reference.y_true):
            raise ValueError(f"{context}: target matrix differs from cohort 0")
        duplicate_models = sorted(set(models) & set(cohort.models))
        if duplicate_models:
            raise ValueError(f"{context}: duplicate models {duplicate_models}")
        models.extend(cohort.models)
        sources.extend(cohort.sources)
        probability_blocks.append(cohort.y_prob)

    return _validate_cohort(
        dataset=reference.dataset,
        ids=reference.ids,
        labels=reference.labels,
        y_true=reference.y_true,
        models=models,
        seeds=reference.seeds,
        y_prob=np.concatenate(probability_blocks, axis=0),
        sources=sources,
    )
