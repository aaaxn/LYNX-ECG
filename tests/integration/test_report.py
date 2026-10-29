"""Integration tests for the BuildReport (report.json + skipped.jsonl)."""

from __future__ import annotations

import json

import pytest

from config.models import DatasetConfig
from dataset.qc import SkipEvent
from dataset.report import BuildReport

pytestmark = pytest.mark.integration


def _cfg() -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "dataset_source": "ptbxl",
            "source_variant": "500hz",
            "sampling_rate": 500,
            "graph_leads": [1],
            "node_reduction": {"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}},
        }
    )


def test_close_releases_fd_when_flush_raises(tmp_path):
    """Regression: a failing flush() must still close skipped.jsonl and
    propagate, not leak the file descriptor."""
    rep = BuildReport(
        tmp_path / "reports",
        config=_cfg(),
        dataset_filename="x.h5",
        hdf5_path=tmp_path / "x.h5",
    )

    class _FailingFP:
        def __init__(self) -> None:
            self.closed = False

        def flush(self) -> None:
            raise OSError("disk full")

        def close(self) -> None:
            self.closed = True

    rep._skipped_fp.close()
    fp = _FailingFP()
    rep._skipped_fp = fp

    with pytest.raises(OSError, match="disk full"):
        rep.close()

    assert fp.closed
    assert rep._skipped_fp is None


def test_skipped_jsonl_one_event_per_line(tmp_path):
    rep_dir = tmp_path / "reports" / "test"
    rep = BuildReport(
        rep_dir,
        config=_cfg(),
        dataset_filename="test.h5",
        hdf5_path=tmp_path / "test.h5",
    )
    events = [
        SkipEvent(level="exam", reason="exam_no_valid_leads", exam_id="X1"),
        SkipEvent(level="lead", reason="rpeak_too_few_peaks", exam_id="X2", lead=0),
        SkipEvent(level="beat", reason="beat_flat_amplitude", exam_id="X2", lead=1, beat_index=3),
    ]
    rep.append_skips(events)
    rep.close()

    lines = rep.skipped_path.read_text().splitlines()
    assert len(lines) == 3
    decoded = [json.loads(line) for line in lines]
    assert decoded[0]["level"] == "exam"
    assert decoded[1]["lead"] == 0
    assert decoded[2]["beat_index"] == 3


def test_report_json_counts(tmp_path):
    rep_dir = tmp_path / "reports" / "test"
    rep = BuildReport(
        rep_dir,
        config=_cfg(),
        dataset_filename="test.h5",
        hdf5_path=tmp_path / "test.h5",
    )
    rep.increment_exam(kept=True, leads_seen=1)
    rep.record_lead(valid=True, candidate_beats=10, kept_beats=10)
    rep.record_writer_totals(n_graphs=10, n_nodes=300, n_edges=180)
    rep.close()

    payload = json.loads(rep.report_path.read_text())
    assert payload["counts"]["n_exams_kept"] == 1
    assert payload["counts"]["n_beats_kept"] == 10
    assert payload["counts"]["n_graphs_written"] == 10


def test_skip_breakdown_aggregates(tmp_path):
    rep_dir = tmp_path / "reports" / "test"
    rep = BuildReport(
        rep_dir,
        config=_cfg(),
        dataset_filename="test.h5",
        hdf5_path=tmp_path / "test.h5",
    )
    rep.append_skip(SkipEvent(level="beat", reason="beat_flat_amplitude"))
    rep.append_skip(SkipEvent(level="beat", reason="beat_flat_amplitude"))
    rep.append_skip(SkipEvent(level="beat", reason="beat_duration_out_of_range"))
    rep.close()
    payload = json.loads(rep.report_path.read_text())
    assert payload["skip_breakdown"]["beat_flat_amplitude"] == 2
    assert payload["skip_breakdown"]["beat_duration_out_of_range"] == 1


def test_no_summary_json_when_disabled(tmp_path):
    cfg = DatasetConfig.model_validate(
        {
            "dataset_source": "ptbxl",
            "source_variant": "500hz",
            "sampling_rate": 500,
            "graph_leads": [1],
            "node_reduction": {"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}},
            "reporting": {"write_summary_json": False},
        }
    )
    rep_dir = tmp_path / "reports" / "nosum"
    rep = BuildReport(rep_dir, config=cfg, dataset_filename="x.h5", hdf5_path=tmp_path / "x.h5")
    rep.append_skip(SkipEvent(level="exam", reason="exam_no_valid_leads"))
    rep.close()
    assert not rep.report_path.exists()
    assert rep.skipped_path.exists()
