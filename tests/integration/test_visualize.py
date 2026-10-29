"""Integration tests for the per-graph HDF5 read used by the visualization module."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from config.models import DatasetConfig
from dataset.hdf5_writer import DatasetWriter
from dataset.records import ExamMetadata, GraphRecord
from dataset.schema import GraphSlice
from tests.fixtures.synthetic_graph_hdf5 import write_exam

pytestmark = pytest.mark.integration


def _make_graph(*, exam_id, lead, beat_index, start, end, rpeak, n_nodes=20):
    sample_indices = np.linspace(start, end, n_nodes).round().astype(np.int64)
    sample_indices[0] = start
    sample_indices[-1] = end
    closest = int(np.argmin(np.abs(sample_indices - rpeak)))
    sample_indices[closest] = rpeak
    sample_indices = np.unique(sample_indices)
    n_nodes = sample_indices.size
    node_t = ((sample_indices - start) / 500.0).astype(np.float32)
    amplitude = np.sin(np.linspace(0, 2 * np.pi, n_nodes)).astype(np.float32)
    t_norm = node_t / max(float(node_t[-1]), 1e-12)
    node_x = np.column_stack([amplitude, t_norm]).astype(np.float32)
    edges = [(i, i + 1) for i in range(n_nodes - 1)] + [(0, n_nodes - 1)]
    ei_arr = np.asarray(edges, dtype=np.int64)
    edge_index = np.column_stack([ei_arr.min(1), ei_arr.max(1)]).T
    return GraphRecord(
        exam_id=exam_id,
        lead=lead,
        beat_index=beat_index,
        boundary_type=1 if beat_index == 0 else (2 if beat_index == 2 else 0),
        start_sample=start,
        end_sample=end,
        r_peak_sample=rpeak,
        rpeak_polarity=1,
        original_num_nodes=end - start + 1,
        node_x=node_x,
        node_t=node_t,
        node_original_sample=sample_indices.astype(np.int64),
        edge_index=edge_index,
    )


def _build(path: Path) -> Path:
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "ptbxl",
            "source_variant": "500hz",
            "sampling_rate": 500,
            "graph_leads": [0, 1],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": 20}},
        }
    )
    w = DatasetWriter(
        path,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    for exam_id in ("e1", "e2"):
        graphs = [
            _make_graph(
                exam_id=exam_id,
                lead=lead,
                beat_index=b,
                start=200 + b * 700,
                end=200 + b * 700 + 600,
                rpeak=200 + b * 700 + 300,
            )
            for lead in (0, 1)
            for b in range(3)
        ]
        write_exam(
            w,
            ExamMetadata(
                exam_id=exam_id,
                patient_id=f"p_{exam_id}",
                strat_fold=1,
                labels=np.array([1, 0, 0, 0, 0], dtype=np.uint8),
                label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
                label_taxonomy="synthetic_test_labels",
            ),
            graphs,
        )
    w.finalize()
    return path


def test_graph_slice_returns_arrays(tmp_path):
    path = _build(tmp_path / "ok.h5")
    with h5py.File(path, "r") as f:
        g = GraphSlice.from_hdf5(f, 0)
    assert g.node_x.shape[0] == g.reduced_num_nodes
    assert g.node_t.shape[0] == g.reduced_num_nodes
    assert g.edge_index.dtype == np.int64
    assert g.edge_index.shape[0] == 2
    assert g.edge_attr is not None
    assert g.edge_attr.shape == (g.edge_index.shape[1], 2)
