"""Hermetic tests for the source loaders and the per-sample source transforms."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from dataset.labels import LabelContractError
from dataset.loaders import ExamSample, prepare_exam_sample
from dataset.loaders.code15 import Code15Loader

pytestmark = pytest.mark.integration

_CANONICAL_LABELS = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]


def _exam_sample(signal: np.ndarray, sampling_rate: int) -> ExamSample:
    return ExamSample(
        exam_id="1",
        patient_id="p1",
        strat_fold=-1,
        labels=np.zeros(6, dtype=np.uint8),
        label_names=tuple(_CANONICAL_LABELS),
        label_taxonomy="codetest_6class",
        signal=signal.astype(np.float32),
        sampling_rate=sampling_rate,
    )


def test_prepare_exam_sample_resamples_and_restamps_rate():
    """``prepare_exam_sample`` resamples the signal to the target rate and
    re-stamps ``sampling_rate``, leaving labels/ids untouched."""
    t = np.arange(800) / 400.0
    sig = np.stack([np.sin(2 * np.pi * 5.0 * t) for _ in range(12)])

    s = prepare_exam_sample(_exam_sample(sig, 400), resample_to_hz=300)

    assert s.sampling_rate == 300
    assert s.signal.shape[0] == 12
    assert abs(s.signal.shape[1] - 600) <= 1
    assert s.exam_id == "1" and s.label_taxonomy == "codetest_6class"


def test_prepare_exam_sample_noop_without_transforms():
    """With no filter and no (or same-rate) resample target, the sample passes
    through unchanged."""
    sig = np.random.default_rng(0).normal(size=(12, 800)).astype(np.float32)
    sample = _exam_sample(sig, 400)

    assert prepare_exam_sample(sample) is sample
    assert prepare_exam_sample(sample, resample_to_hz=400) is sample


def test_prepare_exam_sample_filters_at_native_rate_before_resample():
    """The band-pass runs at the native rate, then the resample: the result
    equals filtering at 400 Hz followed by 400→300 resampling, not the other
    way around."""
    from config.models import SignalFilterConfig
    from dataset.filtering import apply_bandpass, build_bandpass_sos
    from dataset.resampling import resample_signal

    rng = np.random.default_rng(1)
    sig = rng.normal(size=(2, 1600)).astype(np.float32)
    sf = SignalFilterConfig(highpass_hz=0.5, lowpass_hz=40.0, order=4)

    out = prepare_exam_sample(_exam_sample(sig, 400), signal_filter=sf, resample_to_hz=300)

    sos = build_bandpass_sos(sampling_rate=400, highpass_hz=0.5, lowpass_hz=40.0, order=4)
    expected = resample_signal(apply_bandpass(sig, sos=sos), 400, 300)
    np.testing.assert_array_equal(out.signal, expected)
    assert out.sampling_rate == 300


def test_code15_remaps_label_columns_to_canonical_order_by_name(tmp_path):
    """Regression: the raw CODE-test HDF5 stores its label columns in the
    physical order ``[1dAVb, RBBB, LBBB, SB, ST, AF]`` (ST *before* AF) and records
    that true order in its ``label_names`` attr. The loader must remap columns to
    the canonical ``(..., AF, ST)`` order *by name*, not pass them through
    positionally, otherwise AF and ST are silently swapped in every build.
    """
    path = tmp_path / "codetest_like.h5"
    n, t, leads = 3, 5, 12
    physical_names = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    labels = np.zeros((n, 6), dtype=np.uint8)
    labels[0, physical_names.index("AF")] = 1
    labels[1, physical_names.index("ST")] = 1
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((n, t, leads), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(n, dtype=np.int64))
        f.create_dataset("labels", data=labels)
        f.attrs["label_names"] = np.array(physical_names, dtype=object)

    samples = list(Code15Loader(path, sampling_rate=400))

    canonical = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
    af, st = canonical.index("AF"), canonical.index("ST")
    assert list(samples[0].label_names) == canonical
    assert samples[0].labels[af] == 1 and samples[0].labels[st] == 0
    assert samples[1].labels[st] == 1 and samples[1].labels[af] == 0


def test_code15_raises_when_labels_dataset_missing(tmp_path):
    """A file that omits the ``labels`` dataset must be rejected; the loader must
    not silently emit all-zero labels for a missing supervised signal."""
    path = tmp_path / "no_labels.h5"
    physical_names = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((1, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(1, dtype=np.int64))
        f.attrs["label_names"] = np.array(physical_names, dtype=object)

    with pytest.raises(ValueError, match="labels"):
        list(Code15Loader(path, sampling_rate=400))


def test_code15_raises_when_labels_wrong_column_count(tmp_path):
    """A file whose ``labels`` dataset has the wrong number of columns must be
    rejected with a clear message that includes the observed shape."""
    path = tmp_path / "wrong_cols.h5"
    physical_names = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((2, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(2, dtype=np.int64))
        f.create_dataset("labels", data=np.zeros((2, 4), dtype=np.uint8))
        f.attrs["label_names"] = np.array(physical_names, dtype=object)

    with pytest.raises(ValueError, match="labels"):
        list(Code15Loader(path, sampling_rate=400))


def test_code15_raises_when_labels_too_few_rows(tmp_path):
    """A file whose ``labels`` dataset has a different row count is rejected."""
    path = tmp_path / "too_few_label_rows.h5"
    physical_names = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    n_exams = 3
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((n_exams, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(n_exams, dtype=np.int64))
        f.create_dataset("labels", data=np.zeros((2, 6), dtype=np.uint8))
        f.attrs["label_names"] = np.array(physical_names, dtype=object)

    with pytest.raises(ValueError, match="labels"):
        list(Code15Loader(path, sampling_rate=400))


def test_code15_raises_when_labels_have_surplus_rows(tmp_path):
    path = tmp_path / "too_many_label_rows.h5"
    physical_names = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((2, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(2, dtype=np.int64))
        f.create_dataset("labels", data=np.zeros((3, 6), dtype=np.uint8))
        f.attrs["label_names"] = np.array(physical_names, dtype=object)

    with pytest.raises(ValueError, match=r"too_many_label_rows.h5.*shape"):
        list(Code15Loader(path, sampling_rate=400))


def test_code15_raises_when_label_names_attr_missing(tmp_path):
    """A file without the ``label_names`` attr has unknown column semantics.
    Assuming a positional order is exactly the bug being fixed, so the loader
    must refuse loudly rather than guess."""
    path = tmp_path / "no_attr.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((1, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(1, dtype=np.int64))
        f.create_dataset("labels", data=np.zeros((1, 6), dtype=np.uint8))

    with pytest.raises(ValueError, match="label_names"):
        list(Code15Loader(path, sampling_rate=400))


def test_code15_raises_when_label_names_missing_a_class(tmp_path):
    """If the file's ``label_names`` omits a canonical class, the remap cannot be
    built. Surface which class is missing instead of an opaque lookup error."""
    path = tmp_path / "incomplete_attr.h5"
    bad_names = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "XX"]
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((1, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(1, dtype=np.int64))
        f.create_dataset("labels", data=np.zeros((1, 6), dtype=np.uint8))
        f.attrs["label_names"] = np.array(bad_names, dtype=object)

    with pytest.raises(LabelContractError, match="AF"):
        list(Code15Loader(path, sampling_rate=400))


@pytest.mark.parametrize("invalid_value", [2.0, 0.5])
def test_code15_rejects_non_binary_raw_label_values(tmp_path, invalid_value):
    path = tmp_path / "invalid_labels.h5"
    physical_names = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    labels = np.zeros((1, 6), dtype=np.float64)
    labels[0, physical_names.index("AF")] = invalid_value
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((1, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(1, dtype=np.int64))
        f.create_dataset("labels", data=labels)
        f.attrs["label_names"] = np.array(physical_names, dtype=object)

    loader = Code15Loader(path, sampling_rate=400)
    with pytest.raises(LabelContractError, match=r"invalid_labels.h5.*binary"):
        next(iter(loader))


def test_code15_rejects_duplicate_physical_label_name(tmp_path):
    path = tmp_path / "duplicate_names.h5"
    duplicate_names = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "AF"]
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((1, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(1, dtype=np.int64))
        f.create_dataset("labels", data=np.zeros((1, 6), dtype=np.uint8))
        f.attrs["label_names"] = np.array(duplicate_names, dtype=object)

    loader = Code15Loader(path, sampling_rate=400)
    with pytest.raises(LabelContractError, match=r"duplicate_names.h5.*unique"):
        next(iter(loader))


def test_code15_rejects_extra_physical_label_name(tmp_path):
    path = tmp_path / "extra_name.h5"
    physical_names = [*_CANONICAL_LABELS, "EXTRA"]
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=np.zeros((1, 5, 12), dtype=np.float32))
        f.create_dataset("exam_id", data=np.arange(1, dtype=np.int64))
        f.create_dataset("labels", data=np.zeros((1, 7), dtype=np.uint8))
        f.attrs["label_names"] = np.array(physical_names, dtype=object)

    loader = Code15Loader(path, sampling_rate=400)
    with pytest.raises(LabelContractError, match=r"extra_name.h5.*extra=.*EXTRA"):
        next(iter(loader))


def test_code15_iter_batches_matches_iter_and_reads_signal_slices(tmp_path, monkeypatch):
    path = tmp_path / "code15.h5"
    signals = np.arange(5 * 16 * 12, dtype=np.float32).reshape(5, 16, 12)
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=signals, chunks=(2, 16, 12))
        f.create_dataset("exam_id", data=np.arange(5, dtype=np.int64))
        f.create_dataset("patient_id", data=np.arange(5, dtype=np.int64) + 100)
        f.create_dataset("labels", data=np.eye(6, dtype=np.uint8)[:5])
        f.attrs["label_names"] = np.array(_CANONICAL_LABELS, dtype=object)

    real_file = h5py.File
    signal_accesses: list[object] = []

    class TrackingDataset:
        def __init__(self, dataset):
            self._dataset = dataset

        @property
        def shape(self):
            return self._dataset.shape

        def __getitem__(self, key):
            signal_accesses.append(key)
            return self._dataset[key]

    class TrackingFile:
        def __init__(self, *args, **kwargs):
            self._file = real_file(*args, **kwargs)

        def __enter__(self):
            self._file.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._file.__exit__(exc_type, exc, tb)

        @property
        def attrs(self):
            return self._file.attrs

        def __contains__(self, key):
            return key in self._file

        def __getitem__(self, key):
            dataset = self._file[key]
            if key == "tracings":
                return TrackingDataset(dataset)
            return dataset

    monkeypatch.setattr(h5py, "File", TrackingFile)

    loader = Code15Loader(path, sampling_rate=400)
    batches = list(loader.iter_batches(batch_size=2))
    batched = [sample for batch in batches for sample in batch]

    monkeypatch.setattr(h5py, "File", real_file)
    iter_samples = list(Code15Loader(path, sampling_rate=400))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert signal_accesses == [slice(0, 2), slice(2, 4), slice(4, 5)]
    assert [sample.exam_id for sample in batched] == [sample.exam_id for sample in iter_samples]
    for batch_sample, iter_sample in zip(batched, iter_samples, strict=True):
        np.testing.assert_array_equal(batch_sample.signal, iter_sample.signal)
        np.testing.assert_array_equal(batch_sample.labels, iter_sample.labels)


def test_code15_transposes_regardless_of_lead_count(tmp_path):
    """Regression: tracings are [N, T, L]; the loader must transpose to
    [L, T] unconditionally, not only for an 8/12-lead allow-list."""
    path = tmp_path / "code15_oddleads.h5"
    n, t, leads = 2, 5, 3
    signals = np.arange(n * t * leads, dtype=np.float32).reshape(n, t, leads)
    with h5py.File(path, "w") as f:
        f.create_dataset("tracings", data=signals)
        f.create_dataset("exam_id", data=np.arange(n, dtype=np.int64))
        f.create_dataset("labels", data=np.zeros((n, 6), dtype=np.uint8))
        f.attrs["label_names"] = np.array(_CANONICAL_LABELS, dtype=object)

    samples = list(Code15Loader(path, sampling_rate=400))

    assert len(samples) == n
    assert samples[0].signal.shape == (leads, t)
    np.testing.assert_array_equal(samples[0].signal, signals[0].T)


def _write_ptbxl_release(release: Path, *, rows: int) -> None:
    release.mkdir(parents=True, exist_ok=True)
    header = "ecg_id,patient_id,strat_fold,scp_codes,filename_lr,filename_hr\n"
    body = "".join(
        f"{i},{100 + i},1,\"{{'NORM': 100.0}}\",records100/{i:05d}_lr,records500/{i:05d}_hr\n"
        for i in range(1, rows + 1)
    )
    (release / "ptbxl_database.csv").write_text(header + body)


class _StubRecord:
    p_signal = np.zeros((10, 12), dtype=np.float32)


def test_ptbxl_iter_reads_waveforms_from_injected_release_dir(tmp_path, monkeypatch):
    """Regression: __iter__ must read waveforms from the injected
    release_dir, not re-derive the path from ECG_DATASETS_ROOT.

    Hermetic: tiny metadata CSVs + a stubbed wfdb.rdrecord, no real dataset.
    """
    import wfdb

    from dataset.loaders.ptbxl import PTBXLLoader

    release = tmp_path / "release"
    (release / "records500" / "00000").mkdir(parents=True)
    (release / "ptbxl_database.csv").write_text(
        "ecg_id,patient_id,strat_fold,scp_codes,filename_lr,filename_hr\n"
        "1,100,1,\"{'NORM': 100.0}\",records100/00000/00001_lr,records500/00000/00001_hr\n"
    )

    monkeypatch.setenv("ECG_DATASETS_ROOT", str(tmp_path / "BOGUS_ENV_ROOT"))

    seen_paths: list[str] = []

    class _StubRecord:
        p_signal = np.zeros((10, 12), dtype=np.float32)

    def _fake_rdrecord(path):
        seen_paths.append(str(path))
        return _StubRecord()

    monkeypatch.setattr(wfdb, "rdrecord", _fake_rdrecord)

    loader = PTBXLLoader(release_dir=release, sampling_rate=500, limit_exams=1)
    samples = list(loader)

    assert len(samples) == 1
    assert len(seen_paths) == 1
    assert seen_paths[0].startswith(str(release))
    assert "BOGUS_ENV_ROOT" not in seen_paths[0]
    assert seen_paths[0].endswith("records500/00000/00001_hr")


def test_ptbxl_limit_zero_has_zero_length_and_yields_no_samples(tmp_path, monkeypatch):
    import wfdb

    from dataset.loaders.ptbxl import PTBXLLoader

    release = tmp_path / "release"
    _write_ptbxl_release(release, rows=2)
    monkeypatch.setattr(wfdb, "rdrecord", lambda path: _StubRecord())
    loader = PTBXLLoader(release_dir=release, sampling_rate=500, limit_exams=0)

    assert len(loader) == 0
    assert list(loader) == []


@pytest.mark.parametrize("missing_column", ["patient_id", "strat_fold"])
def test_ptbxl_rejects_missing_identity_or_split_metadata(tmp_path, missing_column):
    import pandas as pd

    from dataset.loaders.ptbxl import PTBXLLoader

    release = tmp_path / "release"
    _write_ptbxl_release(release, rows=1)
    metadata_path = release / "ptbxl_database.csv"
    metadata = pd.read_csv(metadata_path)
    metadata.drop(columns=[missing_column]).to_csv(metadata_path, index=False)

    loader = PTBXLLoader(release_dir=release, sampling_rate=500)
    with pytest.raises(ValueError, match=missing_column):
        len(loader)


@pytest.mark.parametrize(
    ("column", "invalid"),
    [
        ("patient_id", np.nan),
        ("patient_id", 0),
        ("patient_id", True),
        ("strat_fold", np.nan),
        ("strat_fold", 11),
        ("strat_fold", True),
    ],
)
def test_ptbxl_rejects_invalid_identity_or_split_metadata(tmp_path, column, invalid):
    import pandas as pd

    from dataset.loaders.ptbxl import PTBXLLoader

    release = tmp_path / "release"
    _write_ptbxl_release(release, rows=1)
    metadata_path = release / "ptbxl_database.csv"
    metadata = pd.read_csv(metadata_path)
    if isinstance(invalid, (bool, np.bool_)):
        metadata[column] = invalid
    else:
        metadata.loc[0, column] = invalid
    metadata.to_csv(metadata_path, index=False)

    loader = PTBXLLoader(release_dir=release, sampling_rate=500)
    with pytest.raises(ValueError, match=column):
        len(loader)


@pytest.mark.parametrize("column", ["patient_id", "strat_fold"])
def test_ptbxl_rejects_boolean_token_in_mixed_numeric_metadata(tmp_path, column):
    import pandas as pd

    from dataset.loaders.ptbxl import PTBXLLoader

    release = tmp_path / "release"
    _write_ptbxl_release(release, rows=2)
    metadata_path = release / "ptbxl_database.csv"
    metadata = pd.read_csv(metadata_path)
    metadata[column] = metadata[column].astype(object)
    metadata.loc[1, column] = True
    metadata.to_csv(metadata_path, index=False)

    loader = PTBXLLoader(release_dir=release, sampling_rate=500)
    with pytest.raises(ValueError, match=column):
        len(loader)


def test_ptbxl_metadata_isolated_across_limit(tmp_path):
    from dataset.loaders.ptbxl import PTBXLLoader

    release = tmp_path / "release"
    _write_ptbxl_release(release, rows=3)

    a = PTBXLLoader(release_dir=release, sampling_rate=500, limit_exams=1)
    b = PTBXLLoader(release_dir=release, sampling_rate=500, limit_exams=2)

    assert len(a) == 1
    assert len(b) == 2
    assert a._metadata_df is not b._metadata_df
    assert len(a._metadata_df) == 1
    assert len(b._metadata_df) == 2


def test_ptbxl_iter_labels_do_not_alias_cached_metadata(tmp_path, monkeypatch):
    import wfdb

    from dataset.loaders.ptbxl import PTBXLLoader

    release = tmp_path / "release"
    _write_ptbxl_release(release, rows=3)
    monkeypatch.setattr(wfdb, "rdrecord", lambda path: _StubRecord())

    loader = PTBXLLoader(release_dir=release, sampling_rate=500)
    first = list(loader)
    cached = list(loader._metadata_df["_labels"])
    cached_before = [labels.copy() for labels in cached]

    first[0].labels[:] = 1 - first[0].labels

    assert not np.array_equal(first[0].labels, cached_before[0])
    assert not np.shares_memory(first[0].labels, cached[0])
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(cached, cached_before, strict=True)
    )

    second = list(loader)
    assert (
        [sample.exam_id for sample in first]
        == [sample.exam_id for sample in second]
        == ["1", "2", "3"]
    )
    assert all(
        np.array_equal(sample.labels, expected)
        for sample, expected in zip(second, cached_before, strict=True)
    )
    assert all(
        not np.shares_memory(sample.labels, cached_labels)
        for sample, cached_labels in zip(second, cached, strict=True)
    )
