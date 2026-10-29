"""Smoke build: 100 PTB-XL exams through the full pipeline.

Skips automatically when ``ECG_DATASETS_ROOT`` is not configured or the
PTB-XL release directory is absent.

Marker: ``@pytest.mark.smoke``, opt-in via ``uv run pytest -m smoke``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli.build import main as build_main
from cli.qa import main as qa_main

pytestmark = pytest.mark.smoke


def test_smoke_build_ptbxl_200(tmp_path, ecg_datasets_root):
    ptbxl_dir = ecg_datasets_root / "ptbxl/physionet.org/files/ptb-xl/1.0.3"
    if not (ptbxl_dir / "ptbxl_database.csv").exists():
        pytest.skip(f"PTB-XL metadata not found under {ptbxl_dir}")

    repo_root = Path(__file__).resolve().parents[2]
    config = repo_root / "configs" / "dataset" / "smoke" / "ptbxl_tp_smoke_200.json"
    assert config.exists(), f"missing preset: {config}"

    repo_under_test = tmp_path / "repo"
    repo_under_test.mkdir()

    build_code = build_main(
        [
            "--config",
            str(config),
            "--repo-root",
            str(repo_under_test),
            "--jobs",
            "1",
        ]
    )
    assert build_code == 0, "smoke build failed"

    h5_files = list((repo_under_test / "graph_datasets" / "ptbxl").glob("*.h5"))
    assert len(h5_files) == 1
    h5_path = h5_files[0]

    qa_code = qa_main(["--h5", str(h5_path), "--profile", "full"])
    assert qa_code == 0, "QA full failed on smoke HDF5"

    reports = list((repo_under_test / "reports").iterdir())
    assert len(reports) == 1
    payload = json.loads((reports[0] / "report.json").read_text())
    counts = payload["counts"]
    assert counts["n_exams_seen"] == 200
    assert counts["n_exams_kept"] >= 95
    assert counts["n_graphs_written"] >= 800
    assert payload["dataset_source"] == "ptbxl"

    figures = reports[0] / "figures"
    assert figures.exists()
    smoke = figures / "smoke.png"
    if smoke.exists():
        assert smoke.stat().st_size > 1000
