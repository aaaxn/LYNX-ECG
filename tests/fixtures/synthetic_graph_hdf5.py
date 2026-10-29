"""Synthetic beat-graph + HDF5 builders shared by dataset/model integration tests.

``make_beat_graph`` and ``tiny_dataset_config`` are the low-level pieces: a
single deterministic (hash-seeded) beat ``GraphRecord`` and the 8-node
uniform-reduction ``DatasetConfig`` used by every synthetic-HDF5 integration
test. ``make_exam_metadata`` builds a single-positive-class ``ExamMetadata``
for an arbitrary label space, and ``write_exam`` appends one exam through the
writer's packed path, as the build does.

``build_tiny_hdf5`` composes all three into the minimal fixture shared by
tests/integration/test_dataset_supergraph.py and
tests/integration/test_explain_ecg_timing.py: 3 exams (train/val/test fold) x
2 leads x 3 beats, 5 synthetic labels. tests/integration/
test_train_synthetic.py needs a differently-shaped fixture (8 exams, 2-class,
alternating labels, for a real train/val split) and builds it directly from
the low-level pieces rather than from ``build_tiny_hdf5``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config.models import DatasetConfig
from dataset.hdf5_writer import DatasetWriter
from dataset.records import ExamMetadata, GraphRecord, pack_graphs

SYNTHETIC_LABEL_NAMES = ("1dAVb", "RBBB", "LBBB", "SB", "AF")
SYNTHETIC_LABEL_TAXONOMY = "synthetic_test_labels"


def write_exam(writer: DatasetWriter, metadata: ExamMetadata, graphs: list[GraphRecord]) -> None:
    """Append one exam through the packed writer path the build uses."""
    writer.add_packed_exam_batch([(metadata, pack_graphs(graphs))])


def tiny_dataset_config(*, graph_leads: tuple[int, ...] = (0, 1)) -> DatasetConfig:
    """8-node uniform-reduction config shared by the synthetic-HDF5 tests."""
    return DatasetConfig.model_validate(
        {
            "dataset_source": "ptbxl",
            "source_variant": "500hz",
            "sampling_rate": 500,
            "graph_leads": list(graph_leads),
            "node_features": ["amplitude", "t_norm_beat"],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": 8}},
            "graph": {"edge_attr_names": ["delta_t", "delta_y"]},
        }
    )


def make_beat_graph(
    *,
    exam_id: str,
    lead: int,
    beat_index: int,
    n_nodes: int = 8,
    sampling_rate: float = 500.0,
) -> GraphRecord:
    """Deterministic single-beat ``GraphRecord``, seeded from ``exam_id``/``lead``/``beat_index``."""
    rng = np.random.default_rng(seed=hash(exam_id + str(lead) + str(beat_index)) & 0xFFFF)
    start = beat_index * 1000
    end = start + 700
    rpeak = start + 350
    sample_indices = np.linspace(start, end, n_nodes).round().astype(np.int64)
    sample_indices[0] = start
    sample_indices[-1] = end
    sample_indices = np.sort(np.unique(sample_indices))
    n_nodes = sample_indices.size
    node_t = ((sample_indices - start) / sampling_rate).astype(np.float32)
    amplitude = rng.standard_normal(n_nodes).astype(np.float32)
    t_norm = node_t / max(float(node_t[-1]), 1e-12)
    node_x = np.column_stack([amplitude, t_norm]).astype(np.float32)
    edges = [(i, i + 1) for i in range(n_nodes - 1)] + [(0, n_nodes - 1)]
    edge_arr = np.asarray(edges, dtype=np.int64)
    src = np.minimum(edge_arr[:, 0], edge_arr[:, 1])
    dst = np.maximum(edge_arr[:, 0], edge_arr[:, 1])
    edge_index = np.column_stack([src, dst]).T
    return GraphRecord(
        exam_id=exam_id,
        lead=lead,
        beat_index=beat_index,
        boundary_type=0,
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


def make_exam_metadata(
    exam_id: str,
    label_index: int | None,
    fold: int,
    *,
    label_names: tuple[str, ...],
    label_taxonomy: str,
) -> ExamMetadata:
    """Single-positive-class ``ExamMetadata``: only ``label_names[label_index]`` is 1.

    ``label_index=None`` gives a negative exam with no positive label.
    """
    labels = np.zeros(len(label_names), dtype=np.uint8)
    if label_index is not None:
        labels[label_index] = 1
    return ExamMetadata(
        exam_id=exam_id,
        patient_id=f"p_{exam_id}",
        strat_fold=int(fold),
        labels=labels,
        label_names=label_names,
        label_taxonomy=label_taxonomy,
    )


def write_current_hdf5(
    path: Path,
    *,
    n_exams: int,
    label_names: tuple[str, ...],
    sampling_rate: int = 500,
    target_nodes: int = 30,
    overwrite: bool = False,
) -> Path:
    """Write a small, fully conforming HDF5 for metadata-oriented tests.

    ``sampling_rate`` is the built (effective) rate: any rate other than the
    500 Hz PTB-XL source rate is reached through ``resample_to_hz``.
    """
    config = DatasetConfig.model_validate(
        {
            "dataset_source": "ptbxl",
            "source_variant": "500hz",
            "sampling_rate": 500,
            "resample_to_hz": None if sampling_rate == 500 else sampling_rate,
            "graph_leads": [0, 1],
            "node_features": ["amplitude", "t_norm_beat"],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": target_nodes}},
        }
    )
    writer = DatasetWriter(
        path,
        config=config,
        label_taxonomy="test",
        label_names=label_names,
        created_by_version="test",
        overwrite=overwrite,
    )
    for exam_idx in range(n_exams):
        exam_id = f"T{exam_idx}"
        graphs = [
            make_beat_graph(
                exam_id=exam_id,
                lead=lead,
                beat_index=0,
                sampling_rate=float(sampling_rate),
            )
            for lead in (0, 1)
        ]
        write_exam(
            writer,
            make_exam_metadata(
                exam_id,
                exam_idx % len(label_names),
                exam_idx + 1,
                label_names=label_names,
                label_taxonomy="test",
            ),
            graphs,
        )
    writer.finalize()
    return path


def build_tiny_hdf5(tmp_path: Path) -> str:
    """Build a tiny HDF5: 3 exams across 3 folds, 2 leads, 3 beats each per exam.

    Shared by test_dataset_supergraph.py and test_explain_ecg_timing.py, which
    both only need a bare, schema-valid HDF5 (for a model forward pass /
    timing-bridge check) rather than a real train/val split.
    """
    out = tmp_path / "tiny.h5"
    cfg = tiny_dataset_config(graph_leads=(0, 1))
    writer = DatasetWriter(
        out,
        config=cfg,
        label_taxonomy=SYNTHETIC_LABEL_TAXONOMY,
        label_names=SYNTHETIC_LABEL_NAMES,
        created_by_version="test",
    )
    fold_seq = [1, 9, 10]
    label_seq = [0, 1, 2]
    exam_ids = ["E001", "E002", "E003"]
    for exam_id, fold, lab in zip(exam_ids, fold_seq, label_seq, strict=True):
        graphs = [
            make_beat_graph(exam_id=exam_id, lead=lead, beat_index=beat)
            for lead in (0, 1)
            for beat in (0, 1, 2)
        ]
        write_exam(
            writer,
            make_exam_metadata(
                exam_id,
                lab,
                fold,
                label_names=SYNTHETIC_LABEL_NAMES,
                label_taxonomy=SYNTHETIC_LABEL_TAXONOMY,
            ),
            graphs,
        )
    writer.finalize()
    return str(out)
