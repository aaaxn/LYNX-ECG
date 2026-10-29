"""Integration tests for the ``dataset-qa`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from cli.qa import main as qa_main
from config.models import DatasetConfig
from dataset.hdf5_writer import DatasetWriter
from dataset.records import ExamMetadata, GraphRecord
from tests.fixtures.synthetic_graph_hdf5 import write_exam

pytestmark = pytest.mark.integration


def _build_valid_dataset(path: Path) -> Path:
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "ptbxl",
            "source_variant": "500hz",
            "sampling_rate": 500,
            "graph_leads": [0, 1],
            "node_reduction": {"method": "uniform", "uniform": {"target_nodes": 20}},
        }
    )
    writer = DatasetWriter(
        path,
        config=cfg,
        label_taxonomy="synthetic_test_labels",
        label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
        created_by_version="0.0.1",
    )
    for exam_id in ("1", "2"):
        graphs = []
        for lead in (0, 1):
            for b in range(3):
                start = 200 + b * 700
                end = start + 600
                rpeak = start + 300
                idx = np.linspace(start, end, 20).round().astype(np.int64)
                idx[0] = start
                idx[-1] = end
                idx[int(np.argmin(np.abs(idx - rpeak)))] = rpeak
                idx = np.unique(idx)
                node_t = ((idx - start) / 500.0).astype(np.float32)
                amplitude = np.linspace(-1, 1, idx.size).astype(np.float32)
                t_norm = node_t / max(float(node_t[-1]), 1e-12)
                node_x = np.column_stack([amplitude, t_norm]).astype(np.float32)
                edges = [(i, i + 1) for i in range(idx.size - 1)] + [(0, idx.size - 1)]
                ei_arr = np.asarray(edges, dtype=np.int64)
                ei = np.column_stack([ei_arr.min(1), ei_arr.max(1)]).T
                graphs.append(
                    GraphRecord(
                        exam_id=exam_id,
                        lead=lead,
                        beat_index=b,
                        boundary_type=1 if b == 0 else (2 if b == 2 else 0),
                        start_sample=start,
                        end_sample=end,
                        r_peak_sample=rpeak,
                        rpeak_polarity=1,
                        original_num_nodes=end - start + 1,
                        node_x=node_x,
                        node_t=node_t,
                        node_original_sample=idx,
                        edge_index=ei,
                    )
                )
        write_exam(
            writer,
            ExamMetadata(
                exam_id=exam_id,
                patient_id=f"p{exam_id}",
                strat_fold=1,
                labels=np.array([1, 0, 0, 0, 0], dtype=np.uint8),
                label_names=("1dAVb", "RBBB", "LBBB", "SB", "AF"),
                label_taxonomy="synthetic_test_labels",
            ),
            graphs,
        )
    writer.finalize()
    return path


def test_qa_cli_passes_valid_dataset(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    code = qa_main(["--h5", str(path), "--profile", "full"])
    assert code == 0


def test_qa_cli_returns_1_on_corrupt(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    with h5py.File(path, "r+") as f:
        ei = np.asarray(f["edge_index"])
        ei[0, 0], ei[1, 0] = ei[1, 0], ei[0, 0]
        f["edge_index"][...] = ei
    code = qa_main(["--h5", str(path), "--profile", "full"])
    assert code == 1


def test_qa_cli_writes_json_output(tmp_path):
    path = _build_valid_dataset(tmp_path / "ok.h5")
    out = tmp_path / "qa.json"
    code = qa_main(["--h5", str(path), "--output", str(out)])
    assert code == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["success"] is True
    assert payload["profile"] == "full"
