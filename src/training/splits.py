"""Split policies and per-run split resolution.

Training uses a deterministic patient-hash split into train and validation.
The test set is always a separate HDF5 file.

:func:`resolve_splits` is the single owner of "preset + dataset metadata →
train/val/test indices": strategy dispatch, the HDF5 sentinel checks, and the
optional separate test dataset (read + compatibility-checked here). Both CLI
entry points call it instead of keeping private copies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from training.hdf5_meta import HDF5Meta, assert_test_meta_compatible, read_hdf5_meta

if TYPE_CHECKING:
    from config.train_models import TrainPreset


def _normalize_patient_id(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def patient_hash_split(
    exam_patient_ids: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, val_idx)`` by hashing patient IDs deterministically.

    Each unique patient is assigned to val if
    ``sha256(f"{patient_id}:{seed}") % 10000 < val_fraction * 10000``,
    otherwise to train.  The split is fully deterministic, with no RNG state.
    """
    threshold = int(val_fraction * 10000)
    patient_ids = np.asarray([_normalize_patient_id(pid) for pid in exam_patient_ids])
    unique_patients = np.unique(patient_ids)

    val_patients: list[str] = []
    for pid in unique_patients:
        digest = hashlib.sha256(f"{pid}:{seed}".encode()).hexdigest()
        if int(digest, 16) % 10000 < threshold:
            val_patients.append(pid)

    val_mask = np.isin(patient_ids, val_patients)
    train_mask = ~val_mask

    train_idx = np.where(train_mask)[0].astype(np.int64)
    val_idx = np.where(val_mask)[0].astype(np.int64)

    return np.sort(train_idx), np.sort(val_idx)


@dataclass(frozen=True)
class ResolvedSplits:
    """Per-run split resolution result.

    ``test_meta`` is the metadata of the separate test HDF5 when the preset
    names one (``test_idx`` then spans that file), else ``None`` (``test_idx``
    indexes the training HDF5, and is empty for strategies without a test
    fold).
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    test_meta: HDF5Meta | None


def resolve_splits(preset: TrainPreset, meta: HDF5Meta) -> ResolvedSplits:
    """Resolve the preset's split strategy against the dataset metadata.

    Raises ``RuntimeError`` when the HDF5 lacks the field the strategy needs,
    and ``ValueError`` (via :func:`assert_test_meta_compatible`) when a
    separate test dataset is structurally incompatible with training.
    """
    if meta.patient_ids is None:
        raise RuntimeError(
            "split.strategy='patient_hash' requires exam_patient_id in HDF5, "
            "but the dataset does not contain it."
        )
    train_idx, val_idx = patient_hash_split(
        meta.patient_ids, preset.split.val_fraction, preset.split.seed
    )
    test_idx = np.array([], dtype=np.int64)

    test_meta: HDF5Meta | None = None
    if preset.test_dataset_path is not None:
        test_meta = read_hdf5_meta(Path(preset.test_dataset_path))
        assert_test_meta_compatible(meta, test_meta)
        test_idx = np.arange(test_meta.total_exams, dtype=np.int64)

    return ResolvedSplits(
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, test_meta=test_meta
    )
