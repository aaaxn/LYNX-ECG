"""Integration tests for the current HDF5 writer contract."""

from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest

from config.models import DatasetConfig
from dataset.hdf5_writer import DatasetWriter
from dataset.labels import LabelContractError
from dataset.records import ExamMetadata, GraphRecord, pack_graphs
from tests.fixtures.synthetic_graph_hdf5 import write_exam

pytestmark = pytest.mark.integration


def _config(
    graph_leads=(0, 1),
    method="uniform",
    target_nodes=30,
    edge_attr_names=("delta_t", "delta_y"),
    node_features=("amplitude", "t_norm_beat"),
) -> DatasetConfig:
    payload = {
        "dataset_source": "ptbxl",
        "source_variant": "500hz",
        "sampling_rate": 500,
        "graph_leads": list(graph_leads),
        "node_features": list(node_features),
        "node_reduction": {"method": method, method: {"target_nodes": target_nodes}}
        if method != "none"
        else {"method": "none"},
        "graph": {"edge_attr_names": list(edge_attr_names)},
    }
    return DatasetConfig.model_validate(payload)


def _make_graph(
    *,
    exam_id: str,
    lead: int,
    beat_index: int,
    start: int,
    end: int,
    rpeak: int,
    n_nodes: int,
    sampling_rate: int = 500,
    boundary_type: int = 0,
    rpeak_polarity: int = 1,
) -> GraphRecord:
    rng = np.random.default_rng(seed=hash(exam_id + str(lead) + str(beat_index)) & 0xFFFF)
    sample_indices = np.linspace(start, end, n_nodes).round().astype(np.int64)
    sample_indices[0] = start
    sample_indices[-1] = end
    closest = int(np.argmin(np.abs(sample_indices - rpeak)))
    sample_indices[closest] = rpeak
    sample_indices = np.sort(np.unique(sample_indices))
    if sample_indices.size != n_nodes:
        n_nodes = sample_indices.size

    node_t = ((sample_indices - start) / float(sampling_rate)).astype(np.float32)
    amplitude = rng.standard_normal(n_nodes).astype(np.float32)
    t_norm = node_t / max(float(node_t[-1]), 1e-12)
    node_x = np.column_stack([amplitude, t_norm]).astype(np.float32)

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


def _fake_metadata(exam_id: str, num_label_classes: int = 5) -> ExamMetadata:
    labels = np.zeros(num_label_classes, dtype=np.uint8)
    labels[0] = 1
    return ExamMetadata(
        exam_id=exam_id,
        patient_id=f"patient_{exam_id}",
        strat_fold=1,
        labels=labels,
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF")[:num_label_classes],
        label_taxonomy="synthetic_test_labels",
    )


def test_writer_rejects_invalid_global_label_definition_before_file_creation(tmp_path):
    path = tmp_path / "invalid-definition.h5"

    with pytest.raises(LabelContractError, match=r"writer.*label"):
        DatasetWriter(
            path,
            config=_config(),
            label_taxonomy="",
            label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
            created_by_version="test",
        )

    assert not path.exists()


def test_writer_multiple_exams_multiple_leads(tmp_path):
    cfg = _config(graph_leads=(0, 1))
    out = tmp_path / "multi.h5"
    writer = DatasetWriter(
        out,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    for exam_id in ("100", "200", "300"):
        graphs = []
        for lead in (0, 1):
            for beat in range(3):
                start = 200 + beat * 700
                end = start + 600
                rpeak = start + 300
                graphs.append(
                    _make_graph(
                        exam_id=exam_id,
                        lead=lead,
                        beat_index=beat,
                        start=start,
                        end=end,
                        rpeak=rpeak,
                        n_nodes=30,
                        boundary_type=1 if beat == 0 else (2 if beat == 2 else 0),
                    )
                )
        write_exam(writer, _fake_metadata(exam_id), graphs)
    writer.finalize()

    with h5py.File(out, "r") as f:
        assert int(f["exam_id"].shape[0]) == 3
        assert int(f["graph_exam_index"].shape[0]) == 3 * 2 * 3
        assert int(f["graph_node_ptr"][-1]) == int(f["node_x"].shape[0])
        assert np.all(np.diff(np.asarray(f["graph_node_ptr"])) >= 0)
        assert np.all(np.diff(np.asarray(f["graph_edge_ptr"])) >= 0)


def test_writer_one_batch_matches_one_batch_per_exam(tmp_path):
    """The build writes one batch per chunk; the chunk size must not change the file."""
    cfg = _config(graph_leads=(0, 1))
    separate = tmp_path / "separate.h5"
    batched = tmp_path / "batched.h5"
    exam_graphs = []
    for exam_id in ("100", "200", "300"):
        graphs = [
            _make_graph(
                exam_id=exam_id,
                lead=lead,
                beat_index=lead,
                start=100 * lead,
                end=100 * lead + 200,
                rpeak=100 * lead + 80,
                n_nodes=12,
            )
            for lead in (0, 1)
        ]
        exam_graphs.append((_fake_metadata(exam_id), graphs))

    w = DatasetWriter(
        separate,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    for metadata, graphs in exam_graphs:
        w.add_packed_exam_batch([(metadata, pack_graphs(graphs))])
    w.finalize()

    w = DatasetWriter(
        batched,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    w.add_packed_exam_batch([(metadata, pack_graphs(graphs)) for metadata, graphs in exam_graphs])
    w.finalize()

    with h5py.File(separate, "r") as f_sep, h5py.File(batched, "r") as f_batch:
        for key in f_sep:
            if f_sep[key].dtype.kind in {"O", "S", "U"}:
                assert f_sep[key].asstr()[:].tolist() == f_batch[key].asstr()[:].tolist()
            else:
                np.testing.assert_array_equal(np.asarray(f_sep[key]), np.asarray(f_batch[key]))


def test_writer_attrs_contain_required_fields(tmp_path):
    cfg = _config()
    out = tmp_path / "attrs.h5"
    writer = DatasetWriter(
        out,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    write_exam(
        writer,
        _fake_metadata("1"),
        [_make_graph(exam_id="1", lead=0, beat_index=0, start=0, end=499, rpeak=250, n_nodes=20)],
    )
    writer.finalize()
    with h5py.File(out, "r") as f:
        for required in (
            "created_by_version",
            "created_at",
            "dataset_source",
            "source_variant",
            "sampling_rate",
            "source_sampling_rate",
            "segmentation_mode",
            "node_reduction_method",
            "graph_leads",
            "node_feature_names",
            "edge_attr_names",
            "edge_ordering",
            "exam_label_taxonomy",
            "exam_label_names",
            "config_json",
            "rpeak_polarity_mode",
        ):
            assert required in f.attrs, f"missing attr: {required}"
        assert str(f.attrs["edge_ordering"]) == "src_lt_dst"
        assert str(f.attrs["segmentation_mode"]) == "per_lead"


def test_writer_overwrite_protection(tmp_path):
    cfg = _config()
    out = tmp_path / "exists.h5"
    out.touch()
    with pytest.raises(FileExistsError):
        DatasetWriter(
            out,
            config=cfg,
            label_taxonomy="synthetic_test_labels",
            label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
            created_by_version="0.0.1",
        )
    w = DatasetWriter(
        out,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
        overwrite=True,
    )
    write_exam(
        w,
        _fake_metadata("1"),
        [_make_graph(exam_id="1", lead=0, beat_index=0, start=0, end=499, rpeak=250, n_nodes=20)],
    )
    w.finalize()


def test_writer_zero_graphs_raises_atomically(tmp_path):
    cfg = _config()
    path = tmp_path / "zero.h5"
    w = DatasetWriter(
        path,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    g = _make_graph(exam_id="1", lead=0, beat_index=0, start=0, end=499, rpeak=250, n_nodes=20)
    empty = replace(pack_graphs([g]), n_graphs=0)
    with pytest.raises(ValueError, match="zero graphs"):
        w.add_packed_exam_batch([(_fake_metadata("1"), empty)])
    w.finalize()

    with h5py.File(path, "r") as f:
        assert f["exam_id"].shape == (0,)
        assert f["graph_exam_index"].shape == (0,)


def test_writer_duplicate_exam_id_raises(tmp_path):
    cfg = _config()
    w = DatasetWriter(
        tmp_path / "dup.h5",
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    g = _make_graph(exam_id="1", lead=0, beat_index=0, start=0, end=499, rpeak=250, n_nodes=20)
    write_exam(w, _fake_metadata("1"), [g])
    with pytest.raises(ValueError, match="Duplicate exam_id"):
        write_exam(w, _fake_metadata("1"), [g])
    w.finalize()


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            lambda: replace(
                _fake_metadata("bad-names"),
                label_names=("RBBB", "1dAVb", "LBBB", "SB", "AF"),
            ),
            "expected order",
        ),
        (
            lambda: replace(_fake_metadata("bad-taxonomy"), label_taxonomy="other"),
            "taxonomy",
        ),
        (
            lambda: replace(_fake_metadata("bad-shape"), labels=np.zeros((1, 5), dtype=np.uint8)),
            "shape",
        ),
        (
            lambda: replace(
                _fake_metadata("bad-values"),
                labels=np.array([0, 2, 0, 0, 0], dtype=np.int64),
            ),
            "binary",
        ),
    ],
)
def test_writer_rejects_invalid_exam_label_metadata(tmp_path, metadata, message):
    path = tmp_path / f"{message}.h5"
    w = DatasetWriter(
        path,
        config=_config(),
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="test",
    )
    graph = _make_graph(
        exam_id=metadata().exam_id,
        lead=0,
        beat_index=0,
        start=0,
        end=499,
        rpeak=250,
        n_nodes=20,
    )

    with pytest.raises(LabelContractError, match=message):
        write_exam(w, metadata(), [graph])
    w.finalize()


def test_writer_preflights_whole_batch_before_any_append(tmp_path):
    path = tmp_path / "atomic-label-batch.h5"
    label_names = ("1dAVb", "RBBB", "LBBB", "SB", "AF")
    w = DatasetWriter(
        path,
        config=_config(),
        label_taxonomy="synthetic_test_labels",
        label_names=label_names,
        created_by_version="test",
    )
    first = _fake_metadata("first")
    write_exam(
        w,
        first,
        [
            _make_graph(
                exam_id="first", lead=0, beat_index=0, start=0, end=499, rpeak=250, n_nodes=20
            )
        ],
    )
    before = {name: w._h5[name].shape for name in w._h5}
    before_counts = (w.n_exams, w.n_graphs, w.n_nodes, w.n_edges)
    invalid = replace(
        _fake_metadata("invalid"),
        labels=np.array([0, 2, 0, 0, 0], dtype=np.int64),
    )
    valid_second = _fake_metadata("second")
    second_graph = _make_graph(
        exam_id="second", lead=0, beat_index=0, start=0, end=499, rpeak=250, n_nodes=20
    )
    invalid_graph = _make_graph(
        exam_id="invalid", lead=0, beat_index=0, start=0, end=499, rpeak=250, n_nodes=20
    )

    with pytest.raises(LabelContractError, match=r"invalid.*binary"):
        w.add_packed_exam_batch(
            [
                (valid_second, pack_graphs([second_graph])),
                (invalid, pack_graphs([invalid_graph])),
            ]
        )

    assert (w.n_exams, w.n_graphs, w.n_nodes, w.n_edges) == before_counts
    assert {name: w._h5[name].shape for name in w._h5} == before
    w.finalize()


def test_writer_never_stores_edge_attr(tmp_path):
    """edge_attr is derived at load time, never stored, but the names attr
    still declares what loaders must derive."""
    cfg = _config(edge_attr_names=("delta_t", "delta_y"))
    out = tmp_path / "noattr.h5"
    w = DatasetWriter(
        out,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    g = _make_graph(exam_id="1", lead=0, beat_index=0, start=0, end=499, rpeak=250, n_nodes=20)
    write_exam(w, _fake_metadata("1"), [g])
    w.finalize()
    with h5py.File(out, "r") as f:
        assert "edge_attr" not in f
        assert [str(n) for n in f.attrs["edge_attr_names"]] == ["delta_t", "delta_y"]


def test_writer_rejects_graph_exceeding_int16_nodes(tmp_path):
    """int16 edge_index requires every graph to have at most 32768 nodes."""
    cfg = _config()
    w = DatasetWriter(
        tmp_path / "too_big.h5",
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    n_nodes = np.iinfo(np.int16).max + 2
    g = _make_graph(
        exam_id="1",
        lead=0,
        beat_index=0,
        start=0,
        end=n_nodes - 1,
        rpeak=n_nodes // 2,
        n_nodes=n_nodes,
    )
    assert g.node_x.shape[0] > np.iinfo(np.int16).max + 1
    with pytest.raises(ValueError, match="int16"):
        write_exam(w, _fake_metadata("1"), [g])
    w.finalize()


def test_writer_round_trip_data_integrity(tmp_path):
    """Round-trip preserves exact node and graph data."""
    cfg = _config()
    out = tmp_path / "integrity.h5"
    g = _make_graph(
        exam_id="42",
        lead=1,
        beat_index=2,
        start=500,
        end=1200,
        rpeak=850,
        n_nodes=20,
    )
    w = DatasetWriter(
        out,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    write_exam(w, _fake_metadata("42"), [g])
    w.finalize()

    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(np.asarray(f["node_original_sample"]), g.node_original_sample)
        np.testing.assert_allclose(np.asarray(f["node_x"]), g.node_x, rtol=0)
        np.testing.assert_allclose(np.asarray(f["node_t"]), g.node_t, rtol=0)
        assert np.dtype(f["edge_index"].dtype) == np.dtype(np.int16)
        np.testing.assert_array_equal(np.asarray(f["edge_index"]).astype(np.int64), g.edge_index)
        assert int(f["graph_start_sample"][0]) == g.start_sample
        assert int(f["graph_end_sample"][0]) == g.end_sample
        assert int(f["graph_r_peak_sample"][0]) == g.r_peak_sample
        assert int(f["graph_boundary_type"][0]) == g.boundary_type
        assert int(f["graph_rpeak_polarity"][0]) == g.rpeak_polarity


def test_exit_propagates_finalize_error_and_closes_handle(tmp_path):
    """Regression: a failing finalize() on a clean exit must propagate the
    error AND release the HDF5 handle, not swallow it and leak the file."""
    out = tmp_path / "boom.h5"
    writer = DatasetWriter(
        out,
        config=_config(),
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="test",
    )
    handle = writer._h5
    assert handle

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    writer.finalize = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="disk full"), writer:
        pass

    assert not handle
