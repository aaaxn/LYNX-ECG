"""Integration tests for the HDF5 timing bridge (explain.ecg_timing).

The bridge must return node_t / node_beat aligned row-for-row with the model's
x for an exam. We prove this against the real ExamGraphDataset by reconstructing
t_norm_beat from the bridge's node_t and matching it to the dataset's x[:, 1].
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from explain.ecg_timing import exam_timing
from tests.fixtures.synthetic_graph_hdf5 import build_tiny_hdf5 as _build_tiny_h5
from training.dataset import ExamGraphDataset
from training.inference import single_exam_inputs

pytestmark = pytest.mark.integration


def test_bridge_node_order_matches_dataset(tmp_path):
    """node_t / node_beat from the bridge must align row-for-row with the model x."""
    h5_path = _build_tiny_h5(tmp_path)
    ds = ExamGraphDataset(h5_path, np.arange(3), num_leads=2)

    with h5py.File(h5_path, "r") as h5:
        graph_exam_index = h5["graph_exam_index"][:]
        for exam_idx in range(3):
            item = ds[exam_idx]
            timing = exam_timing(h5, exam_idx, graph_exam_index=graph_exam_index)

            assert timing.node_t.shape[0] == item.x.shape[0]
            assert torch.equal(timing.node_beat, item.subgraph_node_batch)

            recon = torch.empty_like(timing.node_t)
            for beat in timing.node_beat.unique():
                m = timing.node_beat == beat
                recon[m] = timing.node_t[m] / timing.node_t[m].max()
            assert torch.allclose(recon, item.x[:, 1], atol=1e-5)

    ds.close()


def test_bridge_beat_r_peak_times(tmp_path):
    """R-peak is at start+350 samples @ 500 Hz => 0.7 s beat-local for every beat."""
    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r") as h5:
        timing = exam_timing(h5, 0)
    assert timing.beat_r_peak_t.shape == (6,)
    assert torch.allclose(timing.beat_r_peak_t, torch.full((6,), 0.7), atol=1e-5)


def test_bridge_beat_rpeak_polarity(tmp_path):
    """The fixture writes +1 polarity for every beat (see synthetic_graph_hdf5's
    ``make_beat_graph``); the bridge must surface it unchanged, one value per
    subgraph, indexable by the same convention as ``beat_r_peak_t``."""
    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r") as h5:
        timing = exam_timing(h5, 0)
    assert timing.beat_rpeak_polarity.shape == (6,)
    assert timing.beat_rpeak_polarity.dtype == torch.int8
    assert torch.equal(timing.beat_rpeak_polarity, torch.ones(6, dtype=torch.int8))


def test_bridge_rejects_exam_without_subgraphs(tmp_path):
    h5_path = _build_tiny_h5(tmp_path)
    with h5py.File(h5_path, "r") as h5:
        n_exams = int(h5["exam_id"].shape[0])
        with pytest.raises(ValueError, match="no subgraphs"):
            exam_timing(h5, n_exams + 5)


def test_single_exam_inputs_preserves_dataset_node_order(tmp_path):
    """single_exam_inputs must expose exactly the model x rows (same order)."""
    h5_path = _build_tiny_h5(tmp_path)
    ds = ExamGraphDataset(h5_path, np.arange(3), num_leads=2)
    expected_keys = {
        "x",
        "edge_index",
        "edge_attr",
        "subgraph_node_batch",
        "super_edge_index",
        "super_edge_step",
        "subgraph_lead",
        "subgraph_exam",
        "batch_size",
        "subgraph_beat",
    }
    for i in range(3):
        item = ds[i]
        inputs = single_exam_inputs(ds, i, torch.device("cpu"))
        assert set(inputs) == expected_keys
        assert torch.equal(inputs["x"], item.x)
        assert torch.equal(inputs["subgraph_node_batch"], item.subgraph_node_batch)
        assert inputs["batch_size"] == 1
    ds.close()
