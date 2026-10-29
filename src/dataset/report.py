"""Build report aggregator and writers for ``report.json`` + ``skipped.jsonl``.

Two outputs:

- ``reports/<base>/report.json``: final aggregated summary.
- ``reports/<base>/skipped.jsonl``: append-only stream of SkipEvent dicts,
  one JSON object per line. Workers stream events via ``BuildReport.append_skip``;
  the file is closed by ``BuildReport.close()``.
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from config.models import DatasetConfig
from dataset.qc import SkipEvent


class BuildReport:
    """Streaming report writer + in-memory counters."""

    def __init__(
        self,
        reports_dir: Path,
        *,
        config: DatasetConfig,
        dataset_filename: str,
        hdf5_path: Path,
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_filename = dataset_filename
        self.hdf5_path = Path(hdf5_path)
        self.config = config

        self.report_path = self.reports_dir / "report.json"
        self.skipped_path = self.reports_dir / "skipped.jsonl"

        self.n_exams_seen = 0
        self.n_exams_kept = 0
        self.n_exams_skipped = 0
        self.n_leads_seen = 0
        self.n_leads_kept = 0
        self.n_leads_skipped = 0
        self.n_candidate_beats = 0
        self.n_beats_kept = 0
        self.n_beats_skipped = 0

        self.n_graphs_written = 0
        self.n_nodes_written = 0
        self.n_edges_written = 0

        self._skip_breakdown: Counter[str] = Counter()
        self._skipped_fp: TextIO | None = None
        self._open_skipped()

    def _open_skipped(self) -> None:
        if self.config.reporting.write_skipped_jsonl:
            self._skipped_fp = open(self.skipped_path, "w", encoding="utf-8")  # noqa: SIM115

    def append_skip(self, event: SkipEvent) -> None:
        """Append one event to ``skipped.jsonl`` and update counters."""
        self._skip_breakdown[event.reason] += 1
        if self._skipped_fp is not None:
            self._skipped_fp.write(json.dumps(event.to_dict()) + "\n")

    def append_skips(self, events: list[SkipEvent]) -> None:
        for e in events:
            self.append_skip(e)

    def increment_exam(self, *, kept: bool, leads_seen: int = 0) -> None:
        self.n_exams_seen += 1
        if kept:
            self.n_exams_kept += 1
        else:
            self.n_exams_skipped += 1
        self.n_leads_seen += int(leads_seen)

    def record_lead(self, *, valid: bool, candidate_beats: int = 0, kept_beats: int = 0) -> None:
        if valid:
            self.n_leads_kept += 1
        else:
            self.n_leads_skipped += 1
        self.n_candidate_beats += int(candidate_beats)
        self.n_beats_kept += int(kept_beats)
        self.n_beats_skipped += int(candidate_beats) - int(kept_beats)

    def record_writer_totals(
        self,
        *,
        n_graphs: int,
        n_nodes: int,
        n_edges: int,
    ) -> None:
        self.n_graphs_written = int(n_graphs)
        self.n_nodes_written = int(n_nodes)
        self.n_edges_written = int(n_edges)

    def summary_dict(self) -> dict[str, Any]:
        """Aggregated summary as a plain dict (used both in JSON and HDF5 attrs)."""
        return {
            "dataset_filename": self.dataset_filename,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "dataset_source": self.config.dataset_source,
            "source_variant": self.config.source_variant,
            "config_json": json.loads(self.config.model_dump_json()),
            "counts": {
                "n_exams_seen": int(self.n_exams_seen),
                "n_exams_kept": int(self.n_exams_kept),
                "n_exams_skipped": int(self.n_exams_skipped),
                "n_leads_seen": int(self.n_leads_seen),
                "n_leads_kept": int(self.n_leads_kept),
                "n_leads_skipped": int(self.n_leads_skipped),
                "n_candidate_beats": int(self.n_candidate_beats),
                "n_beats_kept": int(self.n_beats_kept),
                "n_beats_skipped": int(self.n_beats_skipped),
                "n_graphs_written": int(self.n_graphs_written),
                "n_nodes_written": int(self.n_nodes_written),
                "n_edges_written": int(self.n_edges_written),
            },
            "skip_breakdown": dict(self._skip_breakdown),
            "qc_config": json.loads(self.config.qc.model_dump_json()),
            "paths": {
                "hdf5": str(self.hdf5_path),
                "skipped_jsonl": str(self.skipped_path),
            },
        }

    def hdf5_attrs(self) -> dict[str, Any]:
        """Subset of summary suitable for direct HDF5 attribute writing."""
        return {
            "n_exams_seen": int(self.n_exams_seen),
            "n_exams_kept": int(self.n_exams_kept),
            "n_exams_skipped": int(self.n_exams_skipped),
            "n_leads_seen": int(self.n_leads_seen),
            "n_leads_kept": int(self.n_leads_kept),
            "n_leads_skipped": int(self.n_leads_skipped),
            "n_candidate_beats": int(self.n_candidate_beats),
            "n_beats_kept": int(self.n_beats_kept),
            "n_beats_skipped": int(self.n_beats_skipped),
            "skipped_breakdown_json": json.dumps(dict(self._skip_breakdown)),
            "report_json_path": str(self.report_path),
            "skip_report_path": str(self.skipped_path),
        }

    def close(self) -> None:
        """Write ``report.json`` and close ``skipped.jsonl``."""
        fp = self._skipped_fp
        if fp is not None:
            self._skipped_fp = None
            try:
                fp.flush()
            finally:
                fp.close()
        if self.config.reporting.write_summary_json:
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump(self.summary_dict(), f, indent=2)

    def __enter__(self) -> BuildReport:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
