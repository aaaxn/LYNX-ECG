"""Tests for the CODE-100 WFDB loader.

Pure-function checks for the shard addressing, plus synthetic-WFDB checks
that exercise the loader end-to-end (record selection, off-rate rejection,
missing-record skipping) without needing the real ~180 GB dataset on disk.
"""

from __future__ import annotations

import numpy as np
import pytest
import wfdb

from dataset.labels import LabelContractError
from dataset.loaders import CODE_6CLASS_LABEL_NAMES
from dataset.loaders.code100 import (
    CODE100_LABEL_TAXONOMY,
    TWELVE_LEAD_ORDER,
    Code100Loader,
    derive_twelve_lead,
    shard_dir_name,
)

_LEADS = ["DI", "DII", "V1", "V2", "V3", "V4", "V5", "V6"]
_N_SAMPLES = 100


def _stored8(di, dii):
    """An 8-lead [8, T] block with the given DI/DII rows and distinct V1..V6 rows."""
    di = np.asarray(di, dtype=np.float32)
    dii = np.asarray(dii, dtype=np.float32)
    v = np.stack([np.full_like(di, 10.0 + k) for k in range(6)])
    return np.vstack([di[None], dii[None], v]), di, dii


@pytest.mark.unit
def test_derive_twelve_lead_shape_and_diii():
    sig8, di, dii = _stored8([1, 2, 3], [4, 5, 6])
    out = derive_twelve_lead(sig8, _LEADS)
    assert out.shape == (12, 3)
    np.testing.assert_allclose(out[0], di)
    np.testing.assert_allclose(out[1], dii)
    np.testing.assert_allclose(out[2], dii - di)


@pytest.mark.unit
def test_derive_augmented_limb_leads():
    sig8, di, dii = _stored8([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    out = derive_twelve_lead(sig8, _LEADS)
    np.testing.assert_allclose(out[3], -(di + dii) / 2.0)
    np.testing.assert_allclose(out[4], di - dii / 2.0)
    np.testing.assert_allclose(out[5], dii - di / 2.0)


@pytest.mark.unit
def test_derive_passes_precordial_leads_through():
    sig8, _, _ = _stored8([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    out = derive_twelve_lead(sig8, _LEADS)
    np.testing.assert_allclose(out[6:12], sig8[2:8])


@pytest.mark.unit
def test_derive_matches_leads_by_name_not_position():
    di = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    dii = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)
    rows = {
        "V1": np.full(3, 10.0, np.float32),
        "V2": np.full(3, 11.0, np.float32),
        "DI": di,
        "V3": np.full(3, 12.0, np.float32),
        "V4": np.full(3, 13.0, np.float32),
        "DII": dii,
        "V5": np.full(3, 14.0, np.float32),
        "V6": np.full(3, 15.0, np.float32),
    }
    scrambled_names = list(rows.keys())
    scrambled_signal = np.vstack([rows[n] for n in scrambled_names])

    out = derive_twelve_lead(scrambled_signal, scrambled_names)
    assert TWELVE_LEAD_ORDER[2] == "DIII"
    np.testing.assert_allclose(out[0], di)
    np.testing.assert_allclose(out[1], dii)
    np.testing.assert_allclose(out[2], dii - di)
    np.testing.assert_allclose(out[6], rows["V1"])


@pytest.mark.unit
def test_derive_raises_when_stored_lead_missing():
    names = ["DI", "V1", "V2", "V3", "V4", "V5", "V6"]
    signal = np.ones((len(names), 3), dtype=np.float32)
    with pytest.raises(ValueError, match="DII"):
        derive_twelve_lead(signal, names)


@pytest.mark.unit
@pytest.mark.parametrize(
    "exam_id,expected",
    [
        (0, "S0000000"),
        (1, "S0000000"),
        (9_999, "S0000000"),
        (10_000, "S0010000"),
        (120_000, "S0120000"),
        (129_999, "S0120000"),
        (1_325_424, "S1320000"),
    ],
)
def test_shard_dir_name(exam_id: int, expected: str) -> None:
    assert shard_dir_name(exam_id) == expected


@pytest.mark.unit
def test_record_path_composition(tmp_path) -> None:
    loader = Code100Loader(root_dir=tmp_path, sampling_rate=300)
    assert loader._record_path(120_000) == tmp_path / "S0120000" / "TNMG120000_N1"


def _write_record(root, exam_id: int, fs: int, lead_names=_LEADS) -> None:
    """Write a synthetic 8-lead WFDB record into the exam's shard directory."""
    shard = root / shard_dir_name(exam_id)
    shard.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 1.0, _N_SAMPLES, dtype=np.float64)
    signal = np.stack([0.5 * np.sin(2.0 * np.pi * (k + 1) * t) for k in range(8)], axis=1)
    wfdb.wrsamp(
        f"TNMG{exam_id}_N1",
        fs=fs,
        units=["mV"] * 8,
        sig_name=list(lead_names),
        p_signal=signal,
        write_dir=str(shard),
    )


@pytest.fixture
def code100_root(tmp_path):
    """A tiny CODE-100 tree: one 300 Hz record, one 600 Hz record, one missing.

    annotations.csv lists three exams; only exam 1 (300 Hz) and exam 2 (600 Hz)
    have records on disk, exam 3 is annotated but absent.
    """
    _write_record(tmp_path, exam_id=1, fs=300)
    _write_record(tmp_path, exam_id=2, fs=600)
    header = "id_exam,id_patient," + ",".join(CODE_6CLASS_LABEL_NAMES)
    rows = [
        "1,1001,1,0,0,0,0,0",
        "2,1002,0,1,0,0,0,0",
        "3,1003,0,0,1,0,0,0",
    ]
    (tmp_path / "annotations.csv").write_text("\n".join([header, *rows]) + "\n")
    return tmp_path


@pytest.mark.integration
def test_keeps_only_configured_rate_300(code100_root) -> None:
    loader = Code100Loader(root_dir=code100_root, sampling_rate=300)
    samples = list(loader)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.exam_id == "1"
    assert sample.sampling_rate == 300
    assert sample.signal.shape == (12, _N_SAMPLES)
    assert sample.label_names == CODE_6CLASS_LABEL_NAMES
    assert sample.label_taxonomy == CODE100_LABEL_TAXONOMY
    assert sample.labels[0] == 1
    assert sample.labels.sum() == 1


@pytest.mark.integration
def test_loader_yields_twelve_lead_in_standard_order(code100_root) -> None:
    loader = Code100Loader(root_dir=code100_root, sampling_rate=300)
    sig = next(iter(loader)).signal
    assert sig.shape == (12, _N_SAMPLES)
    di, dii = sig[0], sig[1]
    np.testing.assert_allclose(sig[2], dii - di, atol=1e-5)
    np.testing.assert_allclose(sig[3], -(di + dii) / 2.0, atol=1e-5)
    np.testing.assert_allclose(sig[4], di - dii / 2.0, atol=1e-5)
    np.testing.assert_allclose(sig[5], dii - di / 2.0, atol=1e-5)


@pytest.mark.integration
def test_rate_threshold_is_config_driven(code100_root) -> None:
    loader = Code100Loader(root_dir=code100_root, sampling_rate=600)
    samples = list(loader)
    assert len(samples) == 1
    assert samples[0].exam_id == "2"
    assert samples[0].sampling_rate == 600
    assert samples[0].labels[1] == 1


@pytest.mark.integration
def test_len_is_annotated_row_count(code100_root) -> None:
    loader = Code100Loader(root_dir=code100_root, sampling_rate=300)
    assert len(loader) == 3


@pytest.mark.integration
def test_skips_record_with_unexpected_lead_set(tmp_path, caplog) -> None:
    _write_record(tmp_path, exam_id=1, fs=300)
    _write_record(
        tmp_path, exam_id=4, fs=300, lead_names=["DI", "XX", "V1", "V2", "V3", "V4", "V5", "V6"]
    )
    header = "id_exam,id_patient," + ",".join(CODE_6CLASS_LABEL_NAMES)
    rows = ["1,1001,1,0,0,0,0,0", "4,1004,0,0,0,0,0,1"]
    (tmp_path / "annotations.csv").write_text("\n".join([header, *rows]) + "\n")

    loader = Code100Loader(root_dir=tmp_path, sampling_rate=300)
    with caplog.at_level("WARNING"):
        samples = list(loader)

    assert [s.exam_id for s in samples] == ["1"]
    assert samples[0].signal.shape == (12, _N_SAMPLES)
    assert any("exam 4" in r.message for r in caplog.records)


@pytest.mark.integration
def test_limit_exams_truncates_annotations(code100_root) -> None:
    loader = Code100Loader(root_dir=code100_root, sampling_rate=300, limit_exams=1)
    assert len(loader) == 1
    assert [s.exam_id for s in loader] == ["1"]


@pytest.mark.integration
def test_accepts_heterogeneous_boolean_integer_annotation_columns(code100_root) -> None:
    import pandas as pd

    annotations = pd.DataFrame(
        {
            "id_exam": [1],
            "id_patient": [1001],
            CODE_6CLASS_LABEL_NAMES[0]: [True],
            CODE_6CLASS_LABEL_NAMES[1]: [0],
            CODE_6CLASS_LABEL_NAMES[2]: [1],
            CODE_6CLASS_LABEL_NAMES[3]: [False],
            CODE_6CLASS_LABEL_NAMES[4]: [0],
            CODE_6CLASS_LABEL_NAMES[5]: [1],
        }
    )
    assert annotations[list(CODE_6CLASS_LABEL_NAMES)].to_numpy().dtype == object

    loader = Code100Loader(root_dir=code100_root, sampling_rate=300)
    loader._df = annotations

    sample = next(iter(loader))

    np.testing.assert_array_equal(sample.labels, np.array([1, 0, 1, 0, 0, 1], dtype=np.uint8))


@pytest.mark.integration
@pytest.mark.parametrize("invalid_value", [2.0, 0.5])
def test_rejects_non_binary_annotation_before_reading_records(
    code100_root,
    monkeypatch,
    invalid_value: float,
) -> None:
    import pandas as pd

    annotations_path = code100_root / "annotations.csv"
    annotations = pd.read_csv(annotations_path)
    label = CODE_6CLASS_LABEL_NAMES[0]
    annotations[label] = annotations[label].astype(float)
    annotations.loc[0, label] = invalid_value
    annotations.to_csv(annotations_path, index=False)
    monkeypatch.setattr(wfdb, "rdrecord", lambda _path: pytest.fail("record was read"))

    loader = Code100Loader(root_dir=code100_root, sampling_rate=300)
    with pytest.raises(LabelContractError, match=r"CODE-100 annotations.*binary"):
        next(iter(loader))
