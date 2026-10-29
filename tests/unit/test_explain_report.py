"""Unit tests for analysis-report serialization and provenance (explain.report).

The tests cover RFC-8259-safe float serialization, the JSON-safety guard on
report contents, and the publication guard that refuses a report whose
checkpoint or dataset changed during the analysis.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from explain.report import (
    AnalysisReport,
    build_analysis_report,
    capture_report_provenance,
    json_num,
    sanitize_stats,
    write_analysis_report,
)
from tests.fixtures.synthetic_graph_hdf5 import write_current_hdf5
from training.checkpoint import load_checkpoint_with_sha256
from training.hdf5_meta import read_hdf5_meta

pytestmark = pytest.mark.unit


def _write_tiny_hdf5(
    path: Path,
    *,
    sampling_rate: int = 500,
    class_names: list[str] | None = None,
) -> None:
    """Write a small HDF5 that satisfies the current dataset contract."""
    names = tuple(class_names if class_names is not None else ["ClassA", "ClassB"])
    write_current_hdf5(
        path,
        n_exams=2,
        label_names=names,
        sampling_rate=sampling_rate,
        target_nodes=30,
        overwrite=path.exists(),
    )


def _write_tiny_checkpoint(
    path: Path,
    *,
    symmetrize_edges: bool = False,
    adjacency: str = "stored_visibility",
    edge_attributes: str = "derived",
    nonce: int = 0,
) -> None:
    """Torch-saved checkpoint dict that ``load_checkpoint`` accepts."""
    torch.save(
        {
            "symmetrize_edges": symmetrize_edges,
            "subgraph_representation": {
                "adjacency": adjacency,
                "edge_attributes": edge_attributes,
            },
            "_nonce": nonce,
        },
        path,
    )


def _capture(ckpt: Path, ds: Path):
    loaded = load_checkpoint_with_sha256(ckpt)
    meta = read_hdf5_meta(ds)
    return capture_report_provenance(
        ckpt,
        ds,
        checkpoint_state=loaded.state,
        checkpoint_sha256=loaded.sha256,
        dataset_meta=meta,
    )


def _make_report(
    tmp_path: Path,
    *,
    script: str = "clinical_alignment_gate",
    parameters: dict | None = None,
    payload: dict | None = None,
    symmetrize_edges: bool = False,
    adjacency: str = "stored_visibility",
) -> tuple[AnalysisReport, Path, Path]:
    ckpt = tmp_path / "ckpt.pt"
    ds = tmp_path / "dataset.h5"
    _write_tiny_checkpoint(ckpt, symmetrize_edges=symmetrize_edges, adjacency=adjacency)
    _write_tiny_hdf5(ds)
    prov = _capture(ckpt, ds)
    report = build_analysis_report(
        script=script,
        provenance=prov,
        parameters={} if parameters is None else parameters,
        payload={} if payload is None else payload,
    )
    return report, ckpt, ds


def test_json_num_maps_only_non_finite_to_none():
    assert json_num(-3.25) == -3.25
    assert json_num(float("nan")) is None


def test_sanitize_stats_sanitizes_floats_only():
    d = {
        "a": float("nan"),
        "b": float("inf"),
        "c": 2.5,
        "n": 3,
        "s": "label",
        "none": None,
        "lst": [1.0, 2.0],
    }
    out = sanitize_stats(d)
    assert out["a"] is None
    assert out["b"] is None
    assert out["c"] == 2.5
    assert out["n"] == 3
    assert out["s"] == "label"
    assert out["none"] is None
    assert out["lst"] == [1.0, 2.0]


def test_written_report_roundtrips_nested_payload(tmp_path):
    payload = {
        "RBBB": {
            "n": 12,
            "ig_in_region": {"median": 0.25, "ci_lo": 0.2, "ci_hi": 0.3},
            "flags": [True, False, None],
            "empty": {},
        }
    }
    report, _, _ = _make_report(tmp_path, parameters={"ig_steps": 64}, payload=payload)
    out = tmp_path / "report.json"
    write_analysis_report(out, report)

    assert AnalysisReport.model_validate_json(out.read_text()) == report


def test_non_json_payload_value_rejected(tmp_path):
    from pydantic import ValidationError

    _report, ckpt, ds = _make_report(tmp_path)
    prov = _capture(ckpt, ds)
    with pytest.raises(ValidationError):
        build_analysis_report(
            script="clinical_alignment_gate",
            provenance=prov,
            parameters={},
            payload={"x": float("nan")},
        )
    with pytest.raises(ValidationError):
        build_analysis_report(
            script="clinical_alignment_gate",
            provenance=prov,
            parameters={"obj": object()},
            payload={},
        )


def test_checkpoint_replacement_during_analysis_preserves_prior_report(tmp_path):
    prior_report, ckpt, ds = _make_report(tmp_path, payload={"version": "prior"})
    out = tmp_path / "report.json"
    write_analysis_report(out, prior_report)
    prior_bytes = out.read_bytes()

    candidate = build_analysis_report(
        script="clinical_alignment_gate",
        provenance=_capture(ckpt, ds),
        parameters={},
        payload={"version": "candidate"},
    )
    _write_tiny_checkpoint(ckpt, adjacency="temporal_chain", nonce=99)

    with pytest.raises(ValueError, match="checkpoint changed during analysis"):
        write_analysis_report(out, candidate)

    assert out.read_bytes() == prior_bytes
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_dataset_replacement_during_analysis_preserves_prior_report(tmp_path):
    prior_report, ckpt, ds = _make_report(tmp_path, payload={"version": "prior"})
    out = tmp_path / "report.json"
    write_analysis_report(out, prior_report)
    prior_bytes = out.read_bytes()

    candidate = build_analysis_report(
        script="clinical_alignment_gate",
        provenance=_capture(ckpt, ds),
        parameters={},
        payload={"version": "candidate"},
    )
    _write_tiny_hdf5(ds, sampling_rate=100)

    with pytest.raises(ValueError, match="dataset identity/provenance changed during analysis"):
        write_analysis_report(out, candidate)

    assert out.read_bytes() == prior_bytes
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_capture_uses_preloaded_snapshot_without_artifact_io(tmp_path, monkeypatch):
    import explain.report as report_mod

    ckpt = tmp_path / "ckpt.pt"
    ds = tmp_path / "dataset.h5"
    _write_tiny_checkpoint(ckpt, symmetrize_edges=True)
    _write_tiny_hdf5(ds)
    loaded = load_checkpoint_with_sha256(ckpt)
    meta = read_hdf5_meta(ds)

    monkeypatch.setattr(
        report_mod,
        "load_checkpoint_with_sha256",
        lambda *_args, **_kwargs: pytest.fail("capture must not reload checkpoint"),
    )
    monkeypatch.setattr(
        report_mod,
        "read_hdf5_meta",
        lambda *_args, **_kwargs: pytest.fail("capture must not reread HDF5"),
    )
    prov = capture_report_provenance(
        ckpt,
        ds,
        checkpoint_state=loaded.state,
        checkpoint_sha256=loaded.sha256,
        dataset_meta=meta,
    )

    assert prov.representation.symmetrize_edges is True
    assert prov.dataset.fingerprint.num_classes == meta.num_classes
