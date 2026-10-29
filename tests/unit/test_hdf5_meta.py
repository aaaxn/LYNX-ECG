"""Tests for training.hdf5_meta, typed HDF5 metadata reader."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from config.models import DatasetConfig
from tests.fixtures.synthetic_graph_hdf5 import build_tiny_hdf5

pytestmark = [pytest.mark.unit]


def _write_minimal_h5(
    path: Path,
    *,
    include_strat_fold: bool = True,
    include_patient_ids: bool = True,
    node_feature_names: list[str] | None = None,
    node_x_width: int | None = None,
    amplitude_normalization: str | None = None,
    node_reduction: dict | None = None,
) -> None:
    """Write a minimal HDF5 satisfying the current dataset contract."""
    features = (
        node_feature_names if node_feature_names is not None else ["amplitude", "t_norm_beat"]
    )
    width = node_x_width if node_x_width is not None else len(features)
    reducer = node_reduction or {"method": "uniform", "uniform": {"target_nodes": 30}}
    config = {
        "dataset_source": "ptbxl",
        "source_variant": "500hz",
        "sampling_rate": 500,
        "graph_leads": [0, 1],
        "node_features": features,
        "node_reduction": reducer,
    }
    if amplitude_normalization is not None:
        config["amplitude_normalization"] = {"method": amplitude_normalization}
    config = json.loads(DatasetConfig.model_validate(config).model_dump_json())

    with h5py.File(path, "w") as f:
        f.attrs["created_by_version"] = "test"
        f.attrs["created_at"] = "2026-01-01T00:00:00+00:00"
        f.attrs["dataset_source"] = "ptbxl"
        f.attrs["source_variant"] = "500hz"
        f.attrs["sampling_rate"] = 500
        f.attrs["source_sampling_rate"] = 500
        f.attrs["segmentation_mode"] = "per_lead"
        f.attrs["node_reduction_method"] = reducer["method"]
        f.attrs["graph_leads"] = [0, 1]
        f.attrs["node_feature_names"] = features
        f.attrs["edge_attr_names"] = ["delta_t", "delta_y"]
        f.attrs["edge_ordering"] = "src_lt_dst"
        f.attrs["exam_label_taxonomy"] = "test"
        f.attrs["exam_label_names"] = ["ClassA", "ClassB"]
        f.attrs["config_json"] = json.dumps(config)
        f.attrs["rpeak_polarity_mode"] = "auto"
        if amplitude_normalization is not None:
            f.attrs["amplitude_normalization"] = amplitude_normalization

        f.create_dataset("exam_id", data=np.array([b"E001", b"E002", b"E003"]))
        f.create_dataset("exam_labels", data=np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8))
        f.create_dataset("graph_exam_index", data=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
        f.create_dataset("graph_lead", data=np.array([0, 1, 0, 1, 0, 1], dtype=np.int16))
        f.create_dataset("graph_beat_index", data=np.arange(6, dtype=np.int64))
        for name in ("graph_start_sample", "graph_end_sample", "graph_r_peak_sample"):
            f.create_dataset(name, data=np.zeros(6, dtype=np.int64))
        for name in ("graph_boundary_type", "graph_rpeak_polarity"):
            f.create_dataset(name, data=np.zeros(6, dtype=np.int8))
        for name in ("graph_original_num_nodes", "graph_reduced_num_nodes"):
            f.create_dataset(name, data=np.ones(6, dtype=np.int64))
        f.create_dataset("graph_node_ptr", data=np.arange(7, dtype=np.int64))
        f.create_dataset("graph_edge_ptr", data=np.zeros(7, dtype=np.int64))
        f.create_dataset("node_x", data=np.zeros((6, width), dtype=np.float32))
        f.create_dataset("node_t", data=np.zeros(6, dtype=np.float32))
        f.create_dataset("node_original_sample", data=np.zeros(6, dtype=np.int64))
        f.create_dataset("edge_index", data=np.empty((2, 0), dtype=np.int16))
        if include_strat_fold:
            f.create_dataset("exam_strat_fold", data=np.array([1, 2, 3], dtype=np.int8))
        if include_patient_ids:
            f.create_dataset("exam_patient_id", data=np.array([100, 200, 300], dtype=np.int64))


def test_read_hdf5_meta_returns_typed_fields(tmp_path: Path) -> None:
    from training.hdf5_meta import HDF5Meta, read_hdf5_meta

    h5_path = tmp_path / "test.h5"
    _write_minimal_h5(h5_path)
    meta = read_hdf5_meta(h5_path)

    assert isinstance(meta, HDF5Meta)
    assert meta.num_classes == 2
    assert meta.num_leads == 2
    assert meta.class_names == ["ClassA", "ClassB"]
    assert meta.sampling_rate == 500
    assert meta.total_exams == 3
    assert meta.total_graphs == 6
    assert meta.exam_labels.shape == (3, 2)
    assert meta.exam_labels.dtype == np.uint8
    assert meta.strat_fold is not None
    assert meta.strat_fold.shape == (3,)
    assert meta.patient_ids is not None
    assert meta.patient_ids.shape == (3,)
    assert meta.node_feature_names == ["amplitude", "t_norm_beat"]
    assert meta.edge_attr_names == ["delta_t", "delta_y"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonbinary", "binary"),
        ("duplicate_names", "unique"),
        ("empty_taxonomy", "taxonomy"),
        ("rank", "rank 2"),
        ("rows", r"expected \(3, 2\)"),
        ("width", r"expected \(3, 2\)"),
    ],
)
def test_read_hdf5_meta_rejects_malformed_label_contract(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = tmp_path / f"malformed-{mutation}.h5"
    _write_minimal_h5(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        if mutation == "nonbinary":
            h5["exam_labels"][0, 0] = 2
        elif mutation == "duplicate_names":
            h5.attrs["exam_label_names"] = ["ClassA", "ClassA"]
        elif mutation == "empty_taxonomy":
            h5.attrs["exam_label_taxonomy"] = ""
        else:
            data = np.asarray(h5["exam_labels"])
            del h5["exam_labels"]
            if mutation == "rank":
                malformed = data[:, 0]
            elif mutation == "rows":
                malformed = data[:-1]
            else:
                malformed = data[:, :1]
            h5.create_dataset("exam_labels", data=malformed, dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        read_hdf5_meta(h5_path)


def test_read_hdf5_meta_strat_fold_nullable(tmp_path: Path) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = tmp_path / "no_fold.h5"
    _write_minimal_h5(h5_path, include_strat_fold=False)
    meta = read_hdf5_meta(h5_path)

    assert meta.strat_fold is None
    assert meta.total_exams == 3


def test_read_hdf5_meta_patient_ids_nullable(tmp_path: Path) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = tmp_path / "no_patient.h5"
    _write_minimal_h5(h5_path, include_patient_ids=False)
    meta = read_hdf5_meta(h5_path)

    assert meta.patient_ids is None
    assert meta.total_exams == 3


def _meta(
    *,
    num_classes: int = 2,
    num_leads: int = 2,
    class_names: list[str] | None = None,
    sampling_rate: int = 500,
    node_feature_names: list[str] | None = None,
    edge_attr_names: list[str] | None = None,
    amplitude_normalization: str | None = None,
    node_reduction: dict | None = None,
):
    from training.hdf5_meta import HDF5Meta

    return HDF5Meta(
        num_classes=num_classes,
        num_leads=num_leads,
        class_names=class_names if class_names is not None else ["ClassA", "ClassB"],
        strat_fold=None,
        exam_labels=np.zeros((1, num_classes), dtype=np.uint8),
        sampling_rate=sampling_rate,
        total_exams=1,
        total_graphs=1,
        patient_ids=None,
        node_feature_names=(
            node_feature_names if node_feature_names is not None else ["amplitude", "t_norm_beat"]
        ),
        edge_attr_names=edge_attr_names if edge_attr_names is not None else ["delta_t", "delta_y"],
        amplitude_normalization=amplitude_normalization,
        node_reduction=node_reduction
        if node_reduction is not None
        else {"method": "uniform", "target_nodes": 30},
    )


def test_read_hdf5_meta_extracts_reducer_provenance(tmp_path: Path) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = tmp_path / "provenance.h5"
    _write_minimal_h5(
        h5_path,
        node_reduction={
            "method": "turning_points",
            "turning_points": {"cutoff_hz": 35.0, "max_gap_ms": 80.0},
        },
    )

    meta = read_hdf5_meta(h5_path)

    assert meta.node_reduction == {
        "method": "turning_points",
        "cutoff_hz": 35.0,
        "max_gap_ms": 80.0,
    }
    assert meta.fingerprint.node_reduction == meta.node_reduction


@pytest.mark.parametrize(
    ("field", "reference", "candidate", "message"),
    [
        (
            "node_reduction",
            {"method": "uniform", "target_nodes": 30},
            {"method": "uniform", "target_nodes": 20},
            "node_reduction",
        ),
    ],
)
def test_assert_test_meta_rejects_representation_provenance_mismatch(
    field: str,
    reference,
    candidate,
    message: str,
) -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    with pytest.raises(ValueError, match=message):
        assert_test_meta_compatible(
            _meta(**{field: reference}),
            _meta(**{field: candidate}),
        )


def test_assert_test_meta_compatible_accepts_matching() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    assert_test_meta_compatible(_meta(), _meta())


def test_assert_test_meta_compatible_rejects_class_count() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    with pytest.raises(ValueError, match="classes"):
        assert_test_meta_compatible(_meta(num_classes=2), _meta(num_classes=3))


def test_assert_test_meta_compatible_rejects_class_name_reorder() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    with pytest.raises(ValueError, match="label names"):
        assert_test_meta_compatible(
            _meta(class_names=["RBBB", "LBBB"]),
            _meta(class_names=["LBBB", "RBBB"]),
        )


def test_assert_test_meta_compatible_rejects_lead_count() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    with pytest.raises(ValueError, match="leads"):
        assert_test_meta_compatible(_meta(num_leads=12), _meta(num_leads=8))


def test_assert_test_meta_compatible_rejects_sampling_rate() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    with pytest.raises(ValueError, match="sampling_rate"):
        assert_test_meta_compatible(_meta(sampling_rate=500), _meta(sampling_rate=100))


def test_assert_test_meta_compatible_rejects_node_feature_mismatch() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    with pytest.raises(ValueError, match="node features"):
        assert_test_meta_compatible(
            _meta(node_feature_names=["amplitude", "t_norm_beat"]),
            _meta(node_feature_names=["amplitude"]),
        )


def test_assert_test_meta_compatible_rejects_edge_attr_mismatch() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    with pytest.raises(ValueError, match="edge attrs"):
        assert_test_meta_compatible(
            _meta(edge_attr_names=["delta_t", "delta_y"]),
            _meta(edge_attr_names=["delta_t"]),
        )


def test_read_hdf5_meta_rejects_feature_name_width_mismatch(tmp_path: Path) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = tmp_path / "corrupt.h5"
    _write_minimal_h5(h5_path, node_feature_names=["amplitude", "t_norm_beat"], node_x_width=3)
    with pytest.raises(ValueError, match="node_x has 3 columns"):
        read_hdf5_meta(h5_path)


def test_assert_model_meta_compatible_accepts_matching() -> None:
    from config.train_models import ModelConfig
    from training.hdf5_meta import assert_model_meta_compatible

    assert_model_meta_compatible(ModelConfig(), _meta())


def test_assert_model_meta_compatible_rejects_in_channels() -> None:
    from config.train_models import ModelConfig
    from training.hdf5_meta import assert_model_meta_compatible

    with pytest.raises(ValueError, match=r"in_channels=2.*1 node features"):
        assert_model_meta_compatible(ModelConfig(), _meta(node_feature_names=["amplitude"]))


def test_assert_model_meta_compatible_rejects_edge_dim() -> None:
    from config.train_models import ModelConfig
    from training.hdf5_meta import assert_model_meta_compatible

    with pytest.raises(ValueError, match=r"edge_dim=2.*1 edge attrs"):
        assert_model_meta_compatible(ModelConfig(), _meta(edge_attr_names=["delta_t"]))


def test_read_hdf5_meta_reads_amplitude_normalization(tmp_path: Path) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = tmp_path / "normalized.h5"
    _write_minimal_h5(h5_path, amplitude_normalization="p95_lead_median")
    meta = read_hdf5_meta(h5_path)

    assert meta.amplitude_normalization == "p95_lead_median"


def test_read_hdf5_meta_amplitude_normalization_absent_is_none(tmp_path: Path) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = tmp_path / "raw.h5"
    _write_minimal_h5(h5_path)
    meta = read_hdf5_meta(h5_path)

    assert meta.amplitude_normalization is None


def test_assert_test_meta_compatible_accepts_matching_normalization() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    assert_test_meta_compatible(
        _meta(amplitude_normalization="p95_lead_median"),
        _meta(amplitude_normalization="p95_lead_median"),
    )


def test_assert_test_meta_compatible_rejects_normalization_mismatch() -> None:
    from training.hdf5_meta import assert_test_meta_compatible

    with pytest.raises(ValueError, match="amplitude_normalization"):
        assert_test_meta_compatible(
            _meta(amplitude_normalization="p95_lead_median"),
            _meta(amplitude_normalization=None),
        )

    with pytest.raises(ValueError, match="amplitude_normalization"):
        assert_test_meta_compatible(
            _meta(amplitude_normalization=None),
            _meta(amplitude_normalization="p95_lead_median"),
        )


def test_hdf5_meta_fingerprint_extracts_semantic_fields() -> None:
    from training.hdf5_meta import DatasetFingerprint

    m = _meta(
        num_classes=2,
        num_leads=2,
        class_names=["ClassA", "ClassB"],
        sampling_rate=500,
        node_feature_names=["amplitude", "t_norm_beat"],
        edge_attr_names=["delta_t", "delta_y"],
        amplitude_normalization="p95_lead_median",
    )
    fp = m.fingerprint
    assert isinstance(fp, DatasetFingerprint)
    assert fp.num_classes == 2
    assert fp.num_leads == 2
    assert fp.class_names == ["ClassA", "ClassB"]
    assert fp.sampling_rate == 500
    assert fp.node_feature_names == ["amplitude", "t_norm_beat"]
    assert fp.edge_attr_names == ["delta_t", "delta_y"]
    assert fp.amplitude_normalization == "p95_lead_median"


def test_assert_fingerprints_compatible_accepts_matching() -> None:
    from training.hdf5_meta import assert_fingerprints_compatible

    fp = _meta().fingerprint
    assert_fingerprints_compatible(
        fp,
        fp,
        reference_label="train dataset",
        candidate_label="evaluation dataset",
    )


def test_assert_fingerprints_compatible_rejects_sampling_rate() -> None:
    from training.hdf5_meta import assert_fingerprints_compatible

    with pytest.raises(ValueError, match="sampling_rate"):
        assert_fingerprints_compatible(
            _meta(sampling_rate=500).fingerprint,
            _meta(sampling_rate=100).fingerprint,
            reference_label="train dataset",
            candidate_label="evaluation dataset",
        )


def test_assert_fingerprints_compatible_rejects_normalization() -> None:
    from training.hdf5_meta import assert_fingerprints_compatible

    with pytest.raises(ValueError, match="amplitude_normalization"):
        assert_fingerprints_compatible(
            _meta(amplitude_normalization="p95_lead_median").fingerprint,
            _meta(amplitude_normalization=None).fingerprint,
            reference_label="train dataset",
            candidate_label="evaluation dataset",
        )


def test_assert_fingerprints_compatible_uses_labels_in_message() -> None:
    from training.hdf5_meta import assert_fingerprints_compatible

    with pytest.raises(ValueError, match="evaluation dataset"):
        assert_fingerprints_compatible(
            _meta(sampling_rate=500).fingerprint,
            _meta(sampling_rate=100).fingerprint,
            reference_label="train dataset",
            candidate_label="evaluation dataset",
        )


def test_require_compatible_dataset_accepts_matching(tmp_path: Path) -> None:
    from dataclasses import asdict

    from training.hdf5_meta import HDF5Meta, read_hdf5_meta, require_compatible_dataset

    h5_path = tmp_path / "matching.h5"
    _write_minimal_h5(h5_path)
    fingerprint = read_hdf5_meta(h5_path).fingerprint
    ckpt_state = {
        "dataset_fingerprint": asdict(fingerprint),
        "class_names": ["ClassA", "ClassB"],
        "symmetrize_edges": False,
    }

    meta = require_compatible_dataset(ckpt_state, h5_path)

    assert isinstance(meta, HDF5Meta)
    assert meta.class_names == ["ClassA", "ClassB"]


def test_require_compatible_dataset_rejects_missing_fingerprint(tmp_path: Path) -> None:
    from training.hdf5_meta import require_compatible_dataset

    h5_path = tmp_path / "matching.h5"
    _write_minimal_h5(h5_path)
    ckpt_state = {"class_names": ["ClassA", "ClassB"], "symmetrize_edges": False}

    with pytest.raises(ValueError, match="dataset_fingerprint"):
        require_compatible_dataset(ckpt_state, h5_path)


def test_require_compatible_dataset_rejects_malformed_fingerprint(tmp_path: Path) -> None:
    from dataclasses import asdict

    from training.hdf5_meta import read_hdf5_meta, require_compatible_dataset

    h5_path = tmp_path / "matching.h5"
    _write_minimal_h5(h5_path)
    malformed = asdict(read_hdf5_meta(h5_path).fingerprint)
    malformed["extra_unexpected_key"] = "boom"
    ckpt_state = {
        "dataset_fingerprint": malformed,
        "class_names": ["ClassA", "ClassB"],
        "symmetrize_edges": False,
    }

    with pytest.raises(ValueError, match="malformed"):
        require_compatible_dataset(ckpt_state, h5_path)


def test_require_compatible_dataset_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    from dataclasses import asdict

    from training.hdf5_meta import read_hdf5_meta, require_compatible_dataset

    h5_path = tmp_path / "matching.h5"
    _write_minimal_h5(h5_path)
    fingerprint = read_hdf5_meta(h5_path).fingerprint
    wrong_fp = asdict(fingerprint)
    wrong_fp["sampling_rate"] = 100
    ckpt_state = {
        "dataset_fingerprint": wrong_fp,
        "class_names": ["ClassA", "ClassB"],
        "symmetrize_edges": False,
    }

    with pytest.raises(ValueError, match="sampling_rate"):
        require_compatible_dataset(ckpt_state, h5_path)


def test_require_compatible_dataset_rejects_stale_class_name_order(tmp_path: Path) -> None:
    from dataclasses import asdict

    from training.hdf5_meta import read_hdf5_meta, require_compatible_dataset

    h5_path = tmp_path / "matching.h5"
    _write_minimal_h5(h5_path)
    fingerprint = read_hdf5_meta(h5_path).fingerprint
    ckpt_state = {
        "dataset_fingerprint": asdict(fingerprint),
        "class_names": ["ClassB", "ClassA"],
        "symmetrize_edges": False,
    }

    with pytest.raises(ValueError, match="wrong labels"):
        require_compatible_dataset(ckpt_state, h5_path)


def test_require_compatible_dataset_rejects_symmetrized_checkpoint_by_default(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from training.hdf5_meta import read_hdf5_meta, require_compatible_dataset

    h5_path = tmp_path / "matching.h5"
    _write_minimal_h5(h5_path)
    fingerprint = read_hdf5_meta(h5_path).fingerprint
    ckpt_state = {
        "dataset_fingerprint": asdict(fingerprint),
        "class_names": ["ClassA", "ClassB"],
        "symmetrize_edges": True,
    }

    with pytest.raises(ValueError, match="symmetrize_edges"):
        require_compatible_dataset(ckpt_state, h5_path)


def test_require_compatible_dataset_accepts_symmetrized_checkpoint_when_allowed(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from training.hdf5_meta import HDF5Meta, read_hdf5_meta, require_compatible_dataset

    h5_path = tmp_path / "matching.h5"
    _write_minimal_h5(h5_path)
    fingerprint = read_hdf5_meta(h5_path).fingerprint
    ckpt_state = {
        "dataset_fingerprint": asdict(fingerprint),
        "class_names": ["ClassA", "ClassB"],
        "symmetrize_edges": True,
    }

    meta = require_compatible_dataset(ckpt_state, h5_path, allow_symmetrized=True)

    assert isinstance(meta, HDF5Meta)


def test_read_hdf5_meta_rejects_incomplete_construction_config(tmp_path: Path) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = Path(build_tiny_hdf5(tmp_path))
    with h5py.File(h5_path, "r+") as h5:
        config = json.loads(str(h5.attrs["config_json"]))
        del config["graph"]["edge_attr_names"]
        h5.attrs["config_json"] = json.dumps(config)

    with pytest.raises(ValueError, match="complete canonical DatasetConfig"):
        read_hdf5_meta(h5_path)


@pytest.mark.parametrize(
    ("attr_name", "attr_value"),
    [
        ("node_reduction_method", "turning_points"),
        ("target_nodes", 999),
    ],
)
def test_read_hdf5_meta_rejects_reducer_root_config_contradictions(
    tmp_path: Path,
    attr_name: str,
    attr_value,
) -> None:
    from training.hdf5_meta import read_hdf5_meta

    h5_path = Path(build_tiny_hdf5(tmp_path))
    with h5py.File(h5_path, "r+") as h5:
        h5.attrs[attr_name] = attr_value

    with pytest.raises(ValueError, match=attr_name):
        read_hdf5_meta(h5_path)
