"""Unit tests for dataset.raw_signal (per-exam preprocessed waveform loading).

Focuses on ``_load_raw_twelve_lead``'s exam_id -> row lookup for codetest,
which must resolve by *identity* (searching the raw file's own ``exam_id``
array), never by *position* (``row_idx = int(exam_id)``). Treating exam_id as a
row index would silently read the wrong exam (or miss out of range).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dataset.raw_signal import (
    _check_native_rate,
    _load_raw_twelve_lead,
    _ptbxl_metadata,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_ptbxl_metadata_cache():
    """Prevent frames cached in one test from leaking into another."""
    _ptbxl_metadata.cache_clear()
    yield
    _ptbxl_metadata.cache_clear()


def _write_raw_h5(path, exam_ids: list[int], n_samples: int = 8, n_leads: int = 12) -> None:
    """Write a synthetic packed raw HDF5 with ``tracings[row] == row`` (constant
    fill per row), so a read can be checked against the row it came from."""
    with h5py.File(path, "w") as f:
        f.create_dataset("exam_id", data=np.asarray(exam_ids, dtype=np.int64))
        tracings = np.zeros((len(exam_ids), n_samples, n_leads), dtype=np.float32)
        for row in range(len(exam_ids)):
            tracings[row, :, :] = float(row)
        f.create_dataset("tracings", data=tracings)


def test_codetest_exam_id_lookup_non_sequential(tmp_path, monkeypatch):
    """Identity lookup for codetest, with ids reverse-ordered relative to their
    row position. A positional read would silently return
    a different exam's signal for every id except the (coincidental) fixed
    point."""
    raw_path = tmp_path / "codetest_raw.hdf5"
    exam_ids = [9, 4, 1, 0]
    _write_raw_h5(raw_path, exam_ids)
    monkeypatch.setattr("dataset.raw_signal.resolve_source_path", lambda *a, **k: raw_path)

    signal = _load_raw_twelve_lead("codetest", "raw_400hz", "1", 400)

    assert signal.shape == (12, 8)
    np.testing.assert_array_equal(signal, np.full((12, 8), 2.0, dtype=np.float32))


def test_missing_exam_id_raises(tmp_path, monkeypatch):
    raw_path = tmp_path / "codetest_raw.hdf5"
    _write_raw_h5(raw_path, [1, 2, 3])
    monkeypatch.setattr("dataset.raw_signal.resolve_source_path", lambda *a, **k: raw_path)

    with pytest.raises(ValueError, match="not found in raw source"):
        _load_raw_twelve_lead("codetest", "raw_400hz", "999", 400)


def test_unsupported_source_raises():
    with pytest.raises(ValueError, match="unsupported dataset_source"):
        _load_raw_twelve_lead("mimic", "500hz", "1", 400)


def test_check_native_rate_mismatch_raises():
    with pytest.raises(ValueError, match="wrong record for this exam_id"):
        _check_native_rate(500, 400, "42")


def test_check_native_rate_match_does_not_raise():
    _check_native_rate(400, 400, "42")


def _write_ptbxl_csv(path, rows: list[tuple[int, str, str]]) -> None:
    """Write a minimal ``ptbxl_database.csv`` with ``ecg_id, filename_lr,
    filename_hr`` columns."""
    with open(path / "ptbxl_database.csv", "w") as f:
        f.write("ecg_id,filename_lr,filename_hr\n")
        for ecg_id, filename_lr, filename_hr in rows:
            f.write(f"{ecg_id},{filename_lr},{filename_hr}\n")


def test_load_raw_twelve_lead_ptbxl_uses_cached_metadata(tmp_path, monkeypatch):
    """End-to-end through ``_load_raw_twelve_lead``'s ptbxl branch: the metadata
    lookup resolves the filename for the requested exam_id, and a missing
    ecg_id still raises KeyError from the underlying ``.loc`` lookup."""
    release_dir = tmp_path / "ptbxl_release"
    release_dir.mkdir()
    _write_ptbxl_csv(
        release_dir,
        [(10, "records100/00000/00010_lr", "records500/00000/00010_hr")],
    )
    monkeypatch.setattr("dataset.raw_signal.resolve_metadata_dir", lambda *a, **k: release_dir)

    captured_paths: list[str] = []

    def fake_rdrecord(path):
        captured_paths.append(path)

        class _Record:
            fs = 500.0
            p_signal = np.zeros((8, 12), dtype=np.float32)

        return _Record()

    monkeypatch.setattr("wfdb.rdrecord", fake_rdrecord)

    signal = _load_raw_twelve_lead("ptbxl", "500hz", "10", 500)

    assert signal.shape == (12, 8)
    assert captured_paths == [str(release_dir / "records500/00000/00010_hr")]

    with pytest.raises(KeyError):
        _load_raw_twelve_lead("ptbxl", "500hz", "999", 500)
