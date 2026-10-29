"""Unit tests for the source-variant registry."""

from __future__ import annotations

import pytest

from dataset.sources import (
    get_datasets_root,
    get_source_variant,
    resolve_metadata_dir,
    resolve_source_path,
    validate_source_variant,
)

pytestmark = pytest.mark.unit


def test_unset_root_raises(monkeypatch):
    monkeypatch.delenv("ECG_DATASETS_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="ECG_DATASETS_ROOT"):
        get_datasets_root()


def test_env_overrides_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(tmp_path))
    assert get_datasets_root() == tmp_path


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown source_variant"):
        get_source_variant("ptbxl", "999hz")


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown dataset_source"):
        get_source_variant("mimic", "100hz")


def test_validate_matches():
    validate_source_variant("ptbxl", "500hz", 500)


def test_validate_mismatch_raises():
    with pytest.raises(ValueError, match="implies sampling_rate=500"):
        validate_source_variant("ptbxl", "500hz", 100)


def test_resolve_source_path_under_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(tmp_path))
    p = resolve_source_path("codetest", "raw_400hz")
    assert p == tmp_path / "code15" / "codetest_raw.hdf5"


def test_resolve_metadata_dir_ptbxl(monkeypatch, tmp_path):
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(tmp_path))
    p = resolve_metadata_dir("ptbxl", "500hz")
    assert p == tmp_path / "ptbxl/physionet.org/files/ptb-xl/1.0.3"


def test_resolve_metadata_dir_code100_is_dataset_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ECG_DATASETS_ROOT", str(tmp_path))
    p = resolve_metadata_dir("code100", "wfdb_300hz")
    assert p == tmp_path / "code100"
