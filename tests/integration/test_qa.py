"""Integration tests for the QA validator (quick + full profiles)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from config.models import DatasetConfig
from dataset.hdf5_writer import DatasetWriter
from dataset.qa import validate_dataset
from dataset.records import ExamMetadata, GraphRecord
from tests.fixtures.synthetic_graph_hdf5 import write_exam

pytestmark = pytest.mark.integration


def _cfg(method="uniform") -> DatasetConfig:
    payload = {
        "dataset_source": "ptbxl",
        "source_variant": "500hz",
        "sampling_rate": 500,
        "graph_leads": [0, 1],
        "node_reduction": {"method": method, method: {"target_nodes": 30}}
        if method != "none"
        else {"method": "none"},
    }
    return DatasetConfig.model_validate(payload)


def _make_graph(
    *,
    exam_id,
    lead,
    beat_index,
    start,
    end,
    rpeak,
    n_nodes,
    sampling_rate=500,
    boundary_type=0,
    rpeak_polarity=1,
    n_features=2,
):
    sample_indices = np.linspace(start, end, n_nodes).round().astype(np.int64)
    sample_indices[0] = start
    sample_indices[-1] = end
    closest = int(np.argmin(np.abs(sample_indices - rpeak)))
    sample_indices[closest] = rpeak
    sample_indices = np.sort(np.unique(sample_indices))
    n_nodes = sample_indices.size

    node_t = ((sample_indices - start) / float(sampling_rate)).astype(np.float32)
    rng = np.random.default_rng(seed=hash((exam_id, lead, beat_index)) & 0xFFFF)
    amplitude = rng.standard_normal(n_nodes).astype(np.float32)
    if n_features == 2:
        t_norm = node_t / max(float(node_t[-1]), 1e-12)
        node_x = np.column_stack([amplitude, t_norm]).astype(np.float32)
    else:
        node_x = amplitude.reshape(-1, 1)

    edges = [(i, i + 1) for i in range(n_nodes - 1)]
    edges.append((0, n_nodes - 1))
    edges_arr = np.asarray(edges, dtype=np.int64)
    src = np.minimum(edges_arr[:, 0], edges_arr[:, 1])
    dst = np.maximum(edges_arr[:, 0], edges_arr[:, 1])
    edge_index = np.column_stack([src, dst]).T

    return GraphRecord(
        exam_id=exam_id,
        lead=lead,
        beat_index=beat_index,
        boundary_type=boundary_type,
        start_sample=start,
        end_sample=end,
        r_peak_sample=rpeak,
        rpeak_polarity=rpeak_polarity,
        original_num_nodes=end - start + 1,
        node_x=node_x,
        node_t=node_t,
        node_original_sample=sample_indices.astype(np.int64),
        edge_index=edge_index,
    )


def _build_valid_dataset(
    path: Path,
    *,
    method="uniform",
    n_features=2,
    beats_per_lead: int = 3,
    exam_ids: tuple[str, ...] = ("1", "2"),
) -> Path:
    cfg = _cfg(method=method)
    if n_features == 1:
        cfg = DatasetConfig.model_validate({**cfg.model_dump(), "node_features": ["amplitude"]})
    w = DatasetWriter(
        path,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    for exam_id in exam_ids:
        graphs = [
            _make_graph(
                exam_id=exam_id,
                lead=ld,
                beat_index=b,
                start=200 + b * 700,
                end=200 + b * 700 + 600,
                rpeak=200 + b * 700 + 300,
                n_nodes=20,
                n_features=n_features,
                boundary_type=1 if b == 0 else (2 if b == 2 else 0),
            )
            for ld in (0, 1)
            for b in range(beats_per_lead)
        ]
        meta = ExamMetadata(
            exam_id=exam_id,
            patient_id=f"p{exam_id}",
            strat_fold=1,
            labels=np.array([1, 0, 0, 0, 0], dtype=np.uint8),
            label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
            label_taxonomy="synthetic_test_labels",
        )
        write_exam(w, meta, graphs)
    w.finalize()
    return path


def test_full_profile_passes_on_uniform(tmp_path):
    path = _build_valid_dataset(tmp_path / "uni.h5", method="uniform")
    res = validate_dataset(path, profile="full")
    assert res.success, [i.message for i in res.issues if i.severity == "error"]


def test_full_profile_passes_on_single_feature(tmp_path):
    path = _build_valid_dataset(tmp_path / "single.h5", n_features=1)
    res = validate_dataset(path, profile="full")
    assert res.success, [i.message for i in res.issues if i.severity == "error"]


@pytest.mark.parametrize("profile", ("quick", "full"))
def test_graph_exam_ownership_valid_with_single_element_chunks(tmp_path, profile):
    path = _build_valid_dataset(tmp_path / "ok.h5")

    result = validate_dataset(path, profile=profile, chunk_size=1)

    assert result.success, [issue.message for issue in result.errors()]


def test_missing_file(tmp_path):
    res = validate_dataset(tmp_path / "missing.h5", profile="quick")
    assert not res.success
    assert any(i.code == "missing_file" for i in res.issues)


def test_corrupt_node_t_detected(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        nt = np.asarray(f["node_t"])
        nt[0] = nt[0] + 0.1234
        f["node_t"][...] = nt
    res = validate_dataset(path, profile="full")
    assert not res.success
    assert any(i.code == "node_t_formula" for i in res.issues)


def test_corrupt_edge_index_src_ge_dst_detected(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        ei = np.asarray(f["edge_index"])
        ei[0, 0], ei[1, 0] = ei[1, 0], ei[0, 0]
        f["edge_index"][...] = ei
    res = validate_dataset(path, profile="full")
    assert not res.success
    assert any(i.code == "src_lt_dst_violation" for i in res.issues)


def test_corrupt_node_t_first_nonzero(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        ns = np.asarray(f["node_original_sample"])
        ns[0] += 5
        f["node_original_sample"][...] = ns
    res = validate_dataset(path, profile="full")
    assert not res.success
    error_codes = {i.code for i in res.issues if i.severity == "error"}
    assert "node_t_formula" in error_codes


def test_corrupt_node_t_breaks_derived_delta_t(tmp_path):
    """Making node_t non-increasing must surface as non-positive derived delta_t."""
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        nt = np.asarray(f["node_t"])
        nt[1] = nt[0] - 1.0
        f["node_t"][...] = nt
    res = validate_dataset(path, profile="full")
    assert not res.success
    assert any(i.code == "delta_t_non_positive" for i in res.issues)


def test_missing_attr_detected(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        del f.attrs["sampling_rate"]
    res = validate_dataset(path, profile="quick")
    assert not res.success
    assert any(i.code == "missing_attr" for i in res.issues)


@pytest.mark.parametrize(
    "mutation", ["nonbinary", "duplicate_names", "empty_taxonomy", "rank", "rows", "width"]
)
def test_label_contract_fails_closed_in_quick_and_full_profiles(tmp_path, mutation):
    path = _build_valid_dataset(tmp_path / f"labels-{mutation}.h5")
    with h5py.File(path, "r+") as f:
        if mutation == "nonbinary":
            f["exam_labels"][0, 0] = 2
        elif mutation == "duplicate_names":
            f.attrs["exam_label_names"] = ["1dAVb", "1dAVb", "LBBB", "SB", "AF"]
        elif mutation == "empty_taxonomy":
            f.attrs["exam_label_taxonomy"] = ""
        else:
            data = np.asarray(f["exam_labels"])
            del f["exam_labels"]
            if mutation == "rank":
                malformed = data[:, 0]
            elif mutation == "rows":
                malformed = data[:-1]
            else:
                malformed = data[:, :-1]
            f.create_dataset("exam_labels", data=malformed, dtype=np.uint8)

    for profile in ("quick", "full"):
        result = validate_dataset(path, profile=profile)
        assert not result.success
        assert any(issue.code == "contract_mismatch" for issue in result.issues)


def test_full_profile_summary_includes_counts(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    res = validate_dataset(path, profile="full")
    assert "num_exams" in res.summary
    assert "num_graphs" in res.summary
    assert "num_nodes" in res.summary
    assert "num_edges" in res.summary
    assert res.summary["num_exams"] == 2
    assert res.summary["num_graphs"] == 2 * 2 * 3
    assert res.summary["num_graphs_deep_checked"] == res.summary["num_graphs"]


def _ownership_issue(path: Path, profile: str, code: str):
    result = validate_dataset(path, profile=profile, chunk_size=1)
    matching = [issue for issue in result.issues if issue.code == code]
    assert not result.success
    assert len(matching) == 1
    return matching[0]


def test_graph_exam_ownership_rejects_negative_index(tmp_path):
    path = _build_valid_dataset(tmp_path / "negative.h5")
    with h5py.File(path, "r+") as f:
        f["graph_exam_index"][0] = -1

    for profile in ("quick", "full"):
        issue = _ownership_issue(path, profile, "graph_exam_index_out_of_range")

        assert "graph_exam_index[0]=-1" in issue.message
        assert "outside [0, 2)" in issue.message


def test_graph_exam_ownership_rejects_index_equal_to_exam_count(tmp_path):
    path = _build_valid_dataset(tmp_path / "high.h5")
    with h5py.File(path, "r+") as f:
        graph_pos = int(f["graph_exam_index"].shape[0]) - 1
        f["graph_exam_index"][graph_pos] = 2

    for profile in ("quick", "full"):
        issue = _ownership_issue(path, profile, "graph_exam_index_out_of_range")

        assert f"graph_exam_index[{graph_pos}]=2" in issue.message
        assert "outside [0, 2)" in issue.message


def test_graph_exam_ownership_rejects_interior_coverage_gap(tmp_path):
    path = _build_valid_dataset(tmp_path / "gap.h5", exam_ids=("1", "2", "3"))
    with h5py.File(path, "r+") as f:
        ownership = f["graph_exam_index"][:]
        ownership[ownership == 1] = 2
        f["graph_exam_index"][...] = ownership

    for profile in ("quick", "full"):
        issue = _ownership_issue(path, profile, "graph_exam_index_incomplete_coverage")

        assert "does not cover every exam in [0, 3)" in issue.message
        assert "first missing exam index 1" in issue.message


def test_graph_exam_ownership_rejects_decrease_across_chunk_boundary(tmp_path):
    path = _build_valid_dataset(tmp_path / "order.h5")
    with h5py.File(path, "r+") as f:
        ownership = f["graph_exam_index"][:]
        first_exam_one = int(np.flatnonzero(ownership == 1)[0])
        ownership[first_exam_one - 1] = 1
        ownership[first_exam_one] = 0
        f["graph_exam_index"][...] = ownership

    for profile in ("quick", "full"):
        issue = _ownership_issue(path, profile, "graph_exam_index_non_monotonic")

        assert f"decreases at graph {first_exam_one}" in issue.message
        assert "1 -> 0" in issue.message


@pytest.mark.parametrize(
    ("malformation", "issue_code"),
    (("fractional", "dtype_mismatch"), ("rank_two", "shape_mismatch")),
)
def test_graph_exam_ownership_schema_errors_return_qa_result(tmp_path, malformation, issue_code):
    path = _build_valid_dataset(tmp_path / f"{malformation}.h5")
    with h5py.File(path, "r+") as f:
        ownership = f["graph_exam_index"][:]
        del f["graph_exam_index"]
        if malformation == "fractional":
            malformed = ownership.astype(np.float64)
            malformed[0] += 0.5
        else:
            malformed = ownership[:, np.newaxis]
        f.create_dataset("graph_exam_index", data=malformed)

    for profile in ("quick", "full"):
        result = validate_dataset(path, profile=profile, chunk_size=1)

        assert not result.success
        assert any(issue.code == issue_code for issue in result.issues)


def test_full_profile_passes_with_tiny_chunks(tmp_path):
    """chunk_size much smaller than every array must not change the verdict."""
    path = _build_valid_dataset(tmp_path / "ok.h5")
    res = validate_dataset(path, profile="full", chunk_size=7)
    assert res.success, [i.message for i in res.issues if i.severity == "error"]
    assert res.summary == validate_dataset(path, profile="full").summary


def test_src_ge_dst_at_chunk_boundary_detected(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    chunk = 7
    with h5py.File(path, "r+") as f:
        ei = np.asarray(f["edge_index"])
        ei[0, chunk], ei[1, chunk] = ei[1, chunk], ei[0, chunk]
        f["edge_index"][...] = ei
    res = validate_dataset(path, profile="full", chunk_size=chunk)
    assert not res.success
    assert any(i.code == "src_lt_dst_violation" for i in res.issues)


def test_ptr_monotonicity_violation_across_chunk_boundary_detected(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    chunk = 7
    with h5py.File(path, "r+") as f:
        ptr = np.asarray(f["graph_node_ptr"])
        ptr[chunk] = ptr[chunk - 1] - 1
        f["graph_node_ptr"][...] = ptr
    res = validate_dataset(path, profile="quick", chunk_size=chunk)
    assert not res.success
    assert any(i.code == "ptr_non_monotonic" for i in res.issues)


def test_non_finite_node_t_in_late_chunk_detected(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        nt = np.asarray(f["node_t"])
        nt[-1] = np.nan
        f["node_t"][...] = nt
    res = validate_dataset(path, profile="full", chunk_size=7)
    assert not res.success
    assert any(i.code == "node_t_non_finite" for i in res.issues)


def test_reduced_num_mismatch_detected_chunked(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        rn = np.asarray(f["graph_reduced_num_nodes"])
        rn[-1] += 1
        f["graph_reduced_num_nodes"][...] = rn
    res = validate_dataset(path, profile="quick", chunk_size=5)
    assert not res.success
    assert any(i.code == "reduced_num_ptr_mismatch" for i in res.issues)


def test_invalid_chunk_size_rejected(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with pytest.raises(ValueError, match="chunk_size"):
        validate_dataset(path, profile="quick", chunk_size=0)


def test_full_profile_detects_corruption_beyond_256_graphs(tmp_path):
    """Corruption in graph index > 255 must be caught by full profile.

    With 2 exams x 2 leads x 65 beats = 260 graphs, a sample of at most 256
    graphs could leave indices 256-259 unchecked. The last graph (index 259)
    is corrupted here, so only an exhaustive check catches it.
    """
    path = _build_valid_dataset(tmp_path / "large.h5", beats_per_lead=65)
    with h5py.File(path, "r+") as f:
        node_ptr = np.asarray(f["graph_node_ptr"])
        n_graphs = node_ptr.shape[0] - 1
        assert n_graphs == 260, f"expected 260 graphs, got {n_graphs}"
        g = n_graphs - 1
        n_lo = int(node_ptr[g])
        nt = np.asarray(f["node_t"])
        nt[n_lo] += 0.1234
        f["node_t"][...] = nt
    res = validate_dataset(path, profile="full")
    assert not res.success
    assert any(i.code == "node_t_formula" for i in res.issues)


def test_exam_axis_shape_mismatch_detected(tmp_path):
    """Truncating an exam-axis dataset by one row must surface as shape_mismatch."""
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        data = np.asarray(f["exam_strat_fold"])
        del f["exam_strat_fold"]
        f.create_dataset("exam_strat_fold", data=data[:-1], dtype=data.dtype)
    res = validate_dataset(path, profile="quick")
    assert not res.success
    assert any(i.code == "shape_mismatch" for i in res.issues)


def test_graph_axis_shape_mismatch_detected(tmp_path):
    """Truncating a graph-axis dataset by one row must surface as shape_mismatch."""
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        data = np.asarray(f["graph_lead"])
        del f["graph_lead"]
        f.create_dataset("graph_lead", data=data[:-1], dtype=data.dtype)
    res = validate_dataset(path, profile="quick")
    assert not res.success
    assert any(i.code == "shape_mismatch" for i in res.issues)


def test_node_axis_shape_mismatch_detected(tmp_path):
    """Truncating a node-axis dataset by one row must surface as shape_mismatch."""
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        data = np.asarray(f["node_t"])
        del f["node_t"]
        f.create_dataset("node_t", data=data[:-1], dtype=data.dtype)
    res = validate_dataset(path, profile="quick")
    assert not res.success
    assert any(i.code == "shape_mismatch" for i in res.issues)


def test_edge_index_wrong_first_dim_detected(tmp_path):
    """edge_index with first dim != 2 must surface as shape_mismatch."""
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        ei = np.asarray(f["edge_index"])
        n_edges = ei.shape[1]
        del f["edge_index"]
        bad = np.zeros((3, n_edges), dtype=ei.dtype)
        bad[:2] = ei
        f.create_dataset("edge_index", data=bad)
    res = validate_dataset(path, profile="quick")
    assert not res.success
    assert any(i.code == "shape_mismatch" for i in res.issues)


def test_exam_labels_width_mismatch_detected(tmp_path):
    """exam_labels second dim != len(exam_label_names) must surface as shape_mismatch."""
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        data = np.asarray(f["exam_labels"])
        del f["exam_labels"]
        f.create_dataset("exam_labels", data=data[:, :-2], dtype=data.dtype)
    res = validate_dataset(path, profile="quick")
    assert not res.success
    assert any(i.code == "shape_mismatch" for i in res.issues)


def test_node_x_width_mismatch_detected(tmp_path):
    """node_x second dim != len(node_feature_names) must surface as shape_mismatch."""
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        data = np.asarray(f["node_x"])
        n_nodes = data.shape[0]
        del f["node_x"]
        extra = np.zeros((n_nodes, data.shape[1] + 1), dtype=data.dtype)
        extra[:, : data.shape[1]] = data
        f.create_dataset("node_x", data=extra)
    res = validate_dataset(path, profile="quick")
    assert not res.success
    assert any(i.code == "shape_mismatch" for i in res.issues)


def test_full_profile_malformed_pointer_shape_no_crash(tmp_path):
    """Full profile must not crash when a pointer array has the wrong length.

    The per-graph loop is guarded against malformed pointer shapes (indexing a
    wrong-shaped array would crash). Crucially, the guard must NOT skip the
    file-level config_json check that follows the loop, so the function must
    return normally and report a shape_mismatch error rather than raising.
    """
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        ptr = np.asarray(f["graph_node_ptr"])
        del f["graph_node_ptr"]
        f.create_dataset("graph_node_ptr", data=ptr[:-1].astype(np.int64))
    res = validate_dataset(path, profile="full")
    assert not res.success
    assert any(i.code == "shape_mismatch" for i in res.issues)
