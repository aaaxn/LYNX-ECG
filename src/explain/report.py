"""Shared JSON serialization and provenance for interpretability reports.

The analysis scripts write their reports through these helpers, which bind each
report to the checkpoint, dataset, and graph representation it used.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from config.train_models import SubgraphRepresentationConfig
from ecg_artifact_io import staged_artifact
from training.checkpoint import (
    load_checkpoint_with_sha256,
    resolve_subgraph_representation,
    resolve_symmetrize_edges,
)
from training.hdf5_meta import HDF5Meta, read_hdf5_meta


def json_num(x: float | None) -> float | None:
    """Map non-finite floats (and None) to None so the output is valid JSON (RFC 8259)."""
    return x if x is not None and math.isfinite(x) else None


def sanitize_stats(d: dict) -> dict:
    """Apply :func:`json_num` to every float value in ``d``; other values pass through."""
    return {k: (json_num(v) if isinstance(v, float) else v) for k, v in d.items()}


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _normalize_report_path(path: Path | str) -> str:
    """Return the non-empty, normalized, absolute string form of ``path``."""
    raw = str(path)
    if not raw:
        raise ValueError("report path must be a non-empty string")
    return str(Path(raw).resolve())


def _ensure_json_safe(value: Any, *, where: str) -> Any:
    """Recursively assert ``value`` is a JSON object graph of finite primitives.

    Accepts ``None``, ``bool``, ``int``, finite ``float``, ``str``, and lists,
    tuples, or string-keyed mappings thereof (tuples normalize to lists, exactly
    as :func:`json.dump` would serialise them). Rejects NaN/Infinity and any
    other Python object so ``parameters``/``payload`` cannot smuggle a non-JSON
    value.
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: non-finite float {value!r} is not valid JSON")
        return value
    if isinstance(value, (list, tuple)):
        return [_ensure_json_safe(v, where=f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, val in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: mapping key {key!r} is not a string")
            out[key] = _ensure_json_safe(val, where=f"{where}.{key}")
        return out
    raise ValueError(f"{where}: value of type {type(value).__name__} is not JSON-serialisable")


def _validate_bound_path(value: str, *, label: str) -> str:
    if not value:
        raise ValueError(f"{label} path must be non-empty")
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise ValueError(
            f"{label} path must be absolute and lexically normalized/canonical: {value!r}"
        )
    return value


class ArtifactPathBinding(BaseModel):
    """Canonical path to a separately validated analysis artifact."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str

    @field_validator("path")
    @classmethod
    def _canonical_path(cls, value: str) -> str:
        if not value or not Path(value).is_absolute() or str(Path(value).resolve()) != value:
            raise ValueError("artifact path must be absolute and canonical")
        return value


class CheckpointBinding(BaseModel):
    """The checkpoint identity: normalized path plus a sha256 byte digest."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def _canonical_path(cls, v: str) -> str:
        return _validate_bound_path(v, label="checkpoint")

    @field_validator("sha256")
    @classmethod
    def _hex_digest(cls, v: str) -> str:
        if not _SHA256_RE.fullmatch(v):
            raise ValueError(f"checkpoint sha256 must be 64 lowercase hex chars: {v!r}")
        return v


class DatasetFingerprintModel(BaseModel):
    """Envelope mirror of :class:`training.hdf5_meta.DatasetFingerprint`.

    Field set and order match ``dataclasses.asdict(meta.fingerprint)`` exactly;
    this plan consumes that semantic identity unchanged rather than inventing a
    second dataset fingerprint.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    num_classes: int
    num_leads: int
    class_names: list[str]
    sampling_rate: int
    node_feature_names: list[str]
    edge_attr_names: list[str]
    amplitude_normalization: str | None
    node_reduction: dict[str, Any]


class DatasetBinding(BaseModel):
    """The dataset identity: normalized path plus semantic fingerprint."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str
    fingerprint: DatasetFingerprintModel

    @field_validator("path")
    @classmethod
    def _canonical_path(cls, v: str) -> str:
        return _validate_bound_path(v, label="dataset")


class RepresentationBinding(BaseModel):
    """The checkpoint's analysis-time graph semantics.

    The ``subgraph_representation`` vocabulary is reused from
    :class:`config.train_models.SubgraphRepresentationConfig`; no second
    representation vocabulary is defined here.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    subgraph_representation: SubgraphRepresentationConfig
    symmetrize_edges: bool


class ReportProvenance(BaseModel):
    """Captured checkpoint/dataset/representation bindings for one report."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    checkpoint: CheckpointBinding
    dataset: DatasetBinding
    representation: RepresentationBinding


class AnalysisReport(BaseModel):
    """The strict analysis-report envelope.

    Exactly the documented top-level keys, no others. Producer-specific
    scientific values live opaquely under ``payload``.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    script: str
    checkpoint: CheckpointBinding
    dataset: DatasetBinding
    representation: RepresentationBinding
    parameters: dict[str, Any]
    payload: dict[str, Any]

    @field_validator("script")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("script must be a non-empty string")
        return v

    @field_validator("parameters")
    @classmethod
    def _json_safe_parameters(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _ensure_json_safe(v, where="parameters")

    @field_validator("payload")
    @classmethod
    def _json_safe_payload(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _ensure_json_safe(v, where="payload")


def capture_report_provenance(
    checkpoint_path: Path | str,
    dataset_path: Path | str,
    *,
    checkpoint_state: dict[str, Any],
    checkpoint_sha256: str,
    dataset_meta: HDF5Meta,
) -> ReportProvenance:
    """Bind provenance from the exact checkpoint/dataset snapshot already analyzed.

    Callers must pass state, digest, and HDF5 metadata returned by the shared
    checkpoint/inference loading boundary. This function performs no artifact
    I/O, so a path replacement cannot make provenance describe bytes other than
    those that produced the payload.
    """
    if not isinstance(checkpoint_state, dict):
        raise ValueError("checkpoint_state must be the validated loaded checkpoint mapping")
    if not isinstance(dataset_meta, HDF5Meta):
        raise ValueError("dataset_meta must be an HDF5Meta from the analyzed dataset")

    representation = resolve_subgraph_representation(checkpoint_state)
    fingerprint = asdict(dataset_meta.fingerprint)
    return ReportProvenance(
        checkpoint=CheckpointBinding(
            path=_normalize_report_path(checkpoint_path),
            sha256=checkpoint_sha256,
        ),
        dataset=DatasetBinding(
            path=_normalize_report_path(dataset_path),
            fingerprint=DatasetFingerprintModel(**fingerprint),
        ),
        representation=RepresentationBinding(
            subgraph_representation=representation,
            symmetrize_edges=resolve_symmetrize_edges(checkpoint_state),
        ),
    )


def build_analysis_report(
    *,
    script: str,
    provenance: ReportProvenance,
    parameters: dict[str, Any],
    payload: dict[str, Any],
) -> AnalysisReport:
    """Assemble the strict :class:`AnalysisReport` from captured provenance."""
    return AnalysisReport(
        script=script,
        checkpoint=provenance.checkpoint,
        dataset=provenance.dataset,
        representation=provenance.representation,
        parameters=parameters,
        payload=payload,
    )


def write_analysis_report(path: Path | str, report: AnalysisReport) -> None:
    """Atomically publish only while the analyzed artifacts remain current."""
    path = Path(path)
    text = report.model_dump_json(indent=2) + "\n"
    with staged_artifact(path, overwrite=True) as staged:
        staged.write_text(text)
        AnalysisReport.model_validate_json(staged.read_text())
        loaded = load_checkpoint_with_sha256(Path(report.checkpoint.path))
        dataset_meta = read_hdf5_meta(Path(report.dataset.path))
        live = capture_report_provenance(
            report.checkpoint.path,
            report.dataset.path,
            checkpoint_state=loaded.state,
            checkpoint_sha256=loaded.sha256,
            dataset_meta=dataset_meta,
        )
        if live.checkpoint != report.checkpoint:
            raise ValueError(
                f"{path}: checkpoint changed during analysis; refusing to publish stale report"
            )
        if live.dataset != report.dataset:
            raise ValueError(
                f"{path}: dataset identity/provenance changed during analysis; "
                "refusing to publish stale report"
            )
        if live.representation != report.representation:
            raise ValueError(
                f"{path}: checkpoint representation changed during analysis; "
                "refusing to publish stale report"
            )
