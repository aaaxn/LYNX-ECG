"""Unit tests for training.efficiency helpers.

Structure:
- Pure-helper tests (no I/O, no model, no wall-clock): count_parameters,
  _beats_to_exam_sums, compute_reduction_stats_from_arrays.
- Measurement tests: nonzero FLOPs and model-device restore, using a tiny
  synthetic model + dataset stub. Wall-clock values are NOT asserted.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from training.efficiency import (
    _beats_to_exam_sums,
    compute_reduction_stats_from_arrays,
    count_parameters,
    graph_reduction_stats,
    measure_flops,
)

pytestmark = pytest.mark.unit


class _TwoLayerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def test_count_parameters_known_values() -> None:
    model = _TwoLayerNet()
    result = count_parameters(model)
    assert result["submodules"]["fc1"] == 8 * 16 + 16  # type: ignore[index]
    assert result["submodules"]["fc2"] == 16 * 4 + 4  # type: ignore[index]
    assert result["total"] == (8 * 16 + 16) + (16 * 4 + 4)


def test_count_parameters_nested_children_rolled_up() -> None:
    """Grand-children parameters roll up into their direct parent."""
    parent = nn.Sequential(
        nn.Sequential(nn.Linear(4, 4)),
        nn.Linear(4, 2),
    )
    result = count_parameters(parent)
    child1_count = result["submodules"]["1"]  # type: ignore[index]
    assert child1_count == 4 * 2 + 2
    child0_count = result["submodules"]["0"]  # type: ignore[index]
    assert child0_count == 4 * 4 + 4


def test_beats_to_exam_sums_basic() -> None:
    per_beat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    beats_per_exam = np.array([2, 3])
    result = _beats_to_exam_sums(per_beat, beats_per_exam)
    assert result[0] == pytest.approx(3.0)
    assert result[1] == pytest.approx(12.0)


def test_beats_to_exam_sums_single_exam() -> None:
    per_beat = np.array([10.0, 20.0, 30.0])
    beats_per_exam = np.array([3])
    result = _beats_to_exam_sums(per_beat, beats_per_exam)
    assert result[0] == pytest.approx(60.0)


def test_beats_to_exam_sums_uneven_split() -> None:
    per_beat = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    beats_per_exam = np.array([1, 2, 3])
    result = _beats_to_exam_sums(per_beat, beats_per_exam)
    assert result.tolist() == pytest.approx([1.0, 2.0, 3.0])


def _make_simple_arrays(
    orig: list[int],
    reduced: list[int],
    edges: list[int],
    beats_per_exam: list[int],
    source_sr: int = 400,
    target_sr: int = 300,
) -> dict:
    return compute_reduction_stats_from_arrays(
        np.array(orig, dtype=np.int64),
        np.array(reduced, dtype=np.int64),
        np.array(edges, dtype=np.int64),
        np.array(beats_per_exam, dtype=np.int64),
        source_sr,
        target_sr,
    )


def test_retained_fraction_resampled_formula() -> None:
    """retained_fraction_resampled = reduced / orig (same sampling-rate space)."""
    stats = _make_simple_arrays([100, 200], [10, 20], [0, 0], [2], source_sr=400, target_sr=300)
    r = stats["retained_fraction_resampled"]
    assert r["mean"] == pytest.approx(0.10)
    assert r["median"] == pytest.approx(0.10)


def test_retained_fraction_native_formula() -> None:
    """retained_fraction_native = reduced / (orig * source_sr / target_sr).

    With orig=300, reduced=30, source_sr=400, target_sr=300:
        implied_native_count = 300 * (400/300) = 400
        retained = 30 / 400 = 0.075
    """
    stats = _make_simple_arrays([300], [30], [0], [1], source_sr=400, target_sr=300)
    r = stats["retained_fraction_native"]
    expected = 30.0 / (300.0 * 400 / 300)
    assert r["mean"] == pytest.approx(expected)


def test_edges_per_beat_stats() -> None:
    stats = _make_simple_arrays([100, 100], [10, 10], [4, 8], [2])
    epb = stats["edges_per_beat"]
    assert epb["mean"] == pytest.approx(6.0)
    assert epb["median"] == pytest.approx(6.0)


def test_nodes_per_exam_sum() -> None:
    """nodes_per_exam = sum of reduced nodes for each exam's beats."""
    stats = _make_simple_arrays(
        [100, 200, 150],
        [10, 20, 15],
        [0, 0, 0],
        [2, 1],
    )
    npe = stats["nodes_per_exam"]
    assert npe["mean"] == pytest.approx((30.0 + 15.0) / 2)


def test_edges_per_exam_from_synthetic_edge_ptr() -> None:
    """edges_per_exam sums edges across each exam's beats."""
    edges = [4, 8, 6]
    stats = _make_simple_arrays(
        [100, 200, 150],
        [10, 20, 15],
        edges,
        [2, 1],
    )
    epe = stats["edges_per_exam"]
    assert epe["mean"] == pytest.approx((12.0 + 6.0) / 2)


@pytest.mark.parametrize(
    ("attrs", "message"),
    [
        ({"sampling_rate": 300}, "source_sampling_rate"),
        ({"source_sampling_rate": 400}, "sampling_rate"),
        ({"source_sampling_rate": 0, "sampling_rate": 300}, "positive integers"),
        ({"source_sampling_rate": 400, "sampling_rate": np.inf}, "positive integers"),
    ],
)
def test_graph_reduction_stats_rejects_missing_or_invalid_sampling_rates(
    tmp_path, attrs, message
) -> None:
    path = tmp_path / "graphs.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("graph_original_num_nodes", data=np.array([10]))
        h5.create_dataset("graph_reduced_num_nodes", data=np.array([5]))
        h5.create_dataset("graph_edge_ptr", data=np.array([0, 2]))
        for name, value in attrs.items():
            h5.attrs[name] = value
    dataset = SimpleNamespace(
        h5_path=path,
        exam_indices=np.array([0]),
        exam_to_graph_start=np.array([0]),
        exam_to_graph_end=np.array([1]),
    )

    with pytest.raises(ValueError, match=message):
        graph_reduction_stats(dataset)


class _SyntheticDataset:
    """Minimal dataset stub that the measurement functions accept.

    Returns PyG Data objects with all fields required by batch_to_forward_kwargs.
    Node count varies by index to simulate graph-size variability.
    """

    def __init__(self, size: int = 3, n_nodes_base: int = 4) -> None:
        self._size = size
        self._n_nodes_base = n_nodes_base

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Data:
        n_nodes = self._n_nodes_base + idx
        n_edges = max(1, n_nodes - 1)
        return Data(
            x=torch.zeros(n_nodes, 2),
            edge_index=torch.zeros(2, n_edges, dtype=torch.long),
            edge_attr=torch.zeros(n_edges, 1),
            subgraph_node_batch=torch.zeros(n_nodes, dtype=torch.long),
            super_edge_index=torch.zeros(2, 1, dtype=torch.long),
            super_edge_step=torch.zeros(1),
            subgraph_lead=torch.zeros(1, dtype=torch.long),
            subgraph_exam=torch.zeros(1, dtype=torch.long),
            subgraph_beat=torch.zeros(1, dtype=torch.long),
        )


class _DummyForwardModel(nn.Module):
    """Minimal model whose forward signature accepts all batch_to_forward_kwargs keys."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        subgraph_node_batch: torch.Tensor,
        super_edge_index: torch.Tensor,
        super_edge_step: torch.Tensor,
        subgraph_lead: torch.Tensor,
        subgraph_exam: torch.Tensor,
        batch_size: int,
        subgraph_beat: torch.Tensor | None = None,
    ) -> Any:
        return self.linear(x).sum()


def test_measure_flops_smoke() -> None:
    """measure_flops returns a dict with the expected scalar keys and nonzero FLOPs."""
    model = _DummyForwardModel()
    ds = _SyntheticDataset(size=3)
    result = measure_flops(model, ds, n_exams=2)
    assert "mean_flops" in result
    assert "median_flops" in result
    assert "std_flops" in result
    assert result["n_exams_measured"] == 2
    assert result["mean_flops"] > 0, "FlopCounterMode returned 0; coverage may have broken"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_measure_flops_restores_model_device() -> None:
    """measure_flops counts on CPU and must move a CUDA model back afterwards.

    ``model.to(cpu_device)`` is in-place and must be undone in a finally block.
    """
    model = _DummyForwardModel().cuda()
    original_device = next(model.parameters()).device
    ds = _SyntheticDataset(size=3)
    measure_flops(model, ds, n_exams=2)
    current_device = next(model.parameters()).device
    assert current_device == original_device, (
        f"measure_flops left model on {current_device}, expected {original_device}"
    )
