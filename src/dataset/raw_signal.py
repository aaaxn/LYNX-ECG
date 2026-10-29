"""Load the full preprocessed 12-lead waveform for one exam of a built dataset.

The built graph HDF5 stores only reduced turning-point node features, not the
underlying waveform (see :mod:`dataset.schema`).  For figures that overlay the
graph on the real ECG, or simply show the clinical 12-lead trace a cardiologist
recognises, we must re-read the raw source recording for a specific exam and
re-apply the *exact* preprocessing the graph was built with, so the recovered
signal is on the same time/amplitude scale as the stored nodes
(``node_t`` seconds, ``node_x[:, 0]`` amplitude).

This is the canonical loader for all three sources:

* **All sources.**  Every source maps its own ``exam_id`` to a record by
  identity, not position: codetest by looking the id up in the raw file's
  ``exam_id`` array (a CODE-15 packed file need not store ``exam_id[i] == i``),
  code100 by WFDB shard + 12-lead derivation, PTB-XL by ``ecg_id`` lookup in
  ``ptbxl_database.csv``.
* **Whole exam.**  We return the full multi-second, 12-lead signal (not a single
  beat window), which is what a clinical strip and a full-lead node overlay need.

The preprocessing parameters (band-pass, resample, amplitude normalisation) are
read from the graph HDF5's ``config_json`` root attribute and driven through the
same :func:`dataset.loaders.prepare_exam_sample` used at build time, so this is
the signal the model consumed, not a differently-processed reconstruction.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from config.models import DatasetConfig
from dataset.loaders import ExamSample, prepare_exam_sample
from dataset.sources import resolve_metadata_dir, resolve_source_path

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["ExamSignal", "load_preprocessed_exam_signal"]


@functools.lru_cache(maxsize=4)
def _ptbxl_metadata(release_dir: str) -> pd.DataFrame:
    """Parsed ``ptbxl_database.csv`` for ``release_dir``, cached per process.

    Keyed on the string path (hash-stable); the CSV is immutable reference
    data, so a process-lifetime cache is safe. ``maxsize=4`` covers realistic
    multi-release use without unbounded growth.
    """
    import pandas as pd

    return pd.read_csv(Path(release_dir) / "ptbxl_database.csv", index_col="ecg_id")


@dataclass(frozen=True)
class ExamSignal:
    """One exam's preprocessed 12-lead waveform, aligned to the graph's scale.

    - ``signal`` ``[n_leads, n_samples]``: preprocessed amplitude, leads-first,
      at ``sampling_rate`` Hz, the same pipeline (filter → resample → amplitude
      normalisation) the graph nodes were built from.
    - ``sampling_rate``: the preprocessed rate; equals the graph HDF5's
      ``sampling_rate`` root attribute, so ``graph_start_sample`` /
      ``graph_r_peak_sample`` index directly into ``signal``.
    - ``exam_id``: the source exam identifier (string) this signal belongs to.
    - ``dataset_source`` / ``source_variant``: provenance from the build config.
    """

    exam_id: str
    signal: np.ndarray
    sampling_rate: int
    dataset_source: str
    source_variant: str


def _decode_str(value: object) -> str:
    """Decode an h5py string value (bytes or str) to a Python str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_raw_twelve_lead(
    dataset_source: str,
    source_variant: str,
    exam_id: str,
    native_sampling_rate: int,
) -> np.ndarray:
    """Read the native-rate raw ``[n_leads, n_samples]`` signal for one exam.

    Dispatches on ``dataset_source``.  Returns leads-first float32 in the
    standard 12-lead order; code100 (which stores 8 physical leads) is expanded
    to 12 via :func:`dataset.loaders.code100.derive_twelve_lead`.

    Raises
    ------
    ValueError
        For an unrecognised ``dataset_source``, or if a WFDB record's sampling
        rate disagrees with the build config (which would signal a wrong
        exam_id / shard, not a benign difference).
    """
    if dataset_source == "codetest":
        raw_path = resolve_source_path(dataset_source, source_variant)
        with h5py.File(raw_path, "r") as raw_h5:
            raw_exam_ids = np.asarray(raw_h5["exam_id"][:])
            matches = np.flatnonzero(raw_exam_ids == int(exam_id))
            if matches.size == 0:
                raise ValueError(f"exam_id {exam_id!r} not found in raw source {raw_path.name}")
            row_idx = int(matches[0])
            signal_key = "tracings" if "tracings" in raw_h5 else "signal"
            raw_chunk = np.asarray(raw_h5[signal_key][row_idx : row_idx + 1], dtype=np.float32)
        return np.transpose(raw_chunk, (0, 2, 1))[0]

    if dataset_source == "code100":
        import wfdb

        from dataset.loaders.code100 import derive_twelve_lead, shard_dir_name

        root = resolve_source_path(dataset_source, source_variant)
        record_path = root / shard_dir_name(int(exam_id)) / f"TNMG{int(exam_id)}_N1"
        record = wfdb.rdrecord(str(record_path))
        _check_native_rate(round(float(record.fs)), native_sampling_rate, exam_id)
        stored = np.asarray(record.p_signal, dtype=np.float32).T
        return derive_twelve_lead(stored, record.sig_name)

    if dataset_source == "ptbxl":
        import wfdb

        release_dir = resolve_metadata_dir(dataset_source, source_variant)
        df = _ptbxl_metadata(str(release_dir))
        rel = str(df.loc[int(exam_id), "filename_hr"])
        record = wfdb.rdrecord(str(release_dir / rel))
        _check_native_rate(round(float(record.fs)), native_sampling_rate, exam_id)
        return np.asarray(record.p_signal, dtype=np.float32).T

    raise ValueError(f"unsupported dataset_source={dataset_source!r} for raw-signal loading")


def _check_native_rate(record_fs: int, config_fs: int, exam_id: str) -> None:
    if record_fs != config_fs:
        raise ValueError(
            f"exam {exam_id!r}: WFDB record sampling rate {record_fs} Hz != build "
            f"config sampling rate {config_fs} Hz. This is probably the "
            f"wrong record for this exam_id."
        )


def load_preprocessed_exam_signal(graph_h5: h5py.File, exam_idx: int) -> ExamSignal:
    """Return the preprocessed 12-lead waveform for ``exam_idx`` of ``graph_h5``.

    Parameters
    ----------
    graph_h5:
        Open graph HDF5 satisfying the current dataset contract.
    exam_idx:
        Zero-based exam index (row into ``graph_h5['exam_id']`` / ``exam_labels``).

    Returns
    -------
    ExamSignal
        The full-exam preprocessed signal, on the same sampling-rate scale as the
        file's ``graph_start_sample`` / ``node_t`` fields.

    Notes
    -----
    The returned ``sampling_rate`` is asserted to equal the graph HDF5's
    ``sampling_rate`` root attribute; a mismatch means the config's
    ``resample_to_hz`` and the built graphs disagree, which must never happen for
    a correctly built file.
    """
    config = DatasetConfig.model_validate_json(_decode_str(graph_h5.attrs["config_json"]))
    native_sampling_rate = config.sampling_rate

    exam_id = _decode_str(graph_h5["exam_id"][exam_idx])
    raw_signal = _load_raw_twelve_lead(
        config.dataset_source, config.source_variant, exam_id, native_sampling_rate
    )

    sample = ExamSample(
        exam_id=exam_id,
        patient_id="unknown",
        strat_fold=-1,
        labels=np.zeros(0, dtype=np.uint8),
        label_names=(),
        label_taxonomy="unused",
        signal=raw_signal,
        sampling_rate=native_sampling_rate,
    )
    sample = prepare_exam_sample(
        sample,
        signal_filter=config.signal_filter,
        resample_to_hz=config.resample_to_hz,
        amplitude_normalization=config.amplitude_normalization,
        graph_leads=config.graph_leads if config.amplitude_normalization is not None else None,
    )

    graph_fs = int(graph_h5.attrs["sampling_rate"])
    if sample.sampling_rate != graph_fs:
        raise ValueError(
            f"preprocessed sampling rate {sample.sampling_rate} Hz != graph HDF5 "
            f"sampling_rate attr {graph_fs} Hz, build config / graph mismatch"
        )

    return ExamSignal(
        exam_id=exam_id,
        signal=np.asarray(sample.signal, dtype=np.float32),
        sampling_rate=sample.sampling_rate,
        dataset_source=config.dataset_source,
        source_variant=config.source_variant,
    )
