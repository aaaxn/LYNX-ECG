"""Unit tests for patient_hash_split and split-related config models.

Covers:
- patient_hash_split: basic contract, determinism, patient isolation, approximate
  fraction, edge cases.
- TrainPreset: required test_dataset_path.
- resolve_splits: patient-hash train/val and the separate test HDF5.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from config.train_models import TrainPreset
from tests.fixtures.synthetic_graph_hdf5 import write_current_hdf5
from training.splits import patient_hash_split

pytestmark = pytest.mark.unit


def _minimal_train_dict(**overrides) -> dict:
    """Minimal TrainPreset dict without test_dataset_path (no split field uses the default)."""
    base: dict = {
        "run_name": "test_run",
        "dataset_path": "graph_datasets/ptbxl/some.h5",
        "sampling": {"n_neg": 1},
    }
    base.update(overrides)
    return base


def _make_patient_ids(n_patients: int, exams_per_patient: int = 1) -> np.ndarray:
    """Return a flat array of patient-id strings with ``exams_per_patient`` rows each."""
    ids = []
    for pid in range(n_patients):
        ids.extend([str(pid)] * exams_per_patient)
    return np.array(ids)


class TestPatientHashSplitBasic:
    def test_union_covers_all_exams(self):
        n = 50
        pids = _make_patient_ids(n)
        train_idx, val_idx = patient_hash_split(pids, val_fraction=0.2, seed=0)
        all_idx = sorted(np.concatenate([train_idx, val_idx]).tolist())
        assert all_idx == list(range(n))

    def test_no_overlap(self):
        pids = _make_patient_ids(50)
        train_idx, val_idx = patient_hash_split(pids, val_fraction=0.2, seed=0)
        assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_returns_int64_dtype(self):
        pids = _make_patient_ids(20)
        train_idx, val_idx = patient_hash_split(pids, val_fraction=0.2, seed=0)
        assert train_idx.dtype == np.int64
        assert val_idx.dtype == np.int64

    def test_indices_are_sorted(self):
        pids = _make_patient_ids(50)
        train_idx, val_idx = patient_hash_split(pids, val_fraction=0.2, seed=0)
        assert list(train_idx) == sorted(train_idx.tolist())
        assert list(val_idx) == sorted(val_idx.tolist())


class TestPatientHashSplitDeterminism:
    def test_same_inputs_same_output(self):
        pids = _make_patient_ids(100)
        train1, val1 = patient_hash_split(pids, val_fraction=0.2, seed=7)
        train2, val2 = patient_hash_split(pids, val_fraction=0.2, seed=7)
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)

    def test_bytes_ids_match_string_ids(self):
        pids_str = _make_patient_ids(100)
        pids_bytes = pids_str.astype("S")
        train_str, val_str = patient_hash_split(pids_str, val_fraction=0.2, seed=7)
        train_bytes, val_bytes = patient_hash_split(pids_bytes, val_fraction=0.2, seed=7)
        np.testing.assert_array_equal(train_bytes, train_str)
        np.testing.assert_array_equal(val_bytes, val_str)

    def test_different_seeds_give_different_splits(self):
        pids = _make_patient_ids(200)
        _, val_a = patient_hash_split(pids, val_fraction=0.2, seed=1)
        _, val_b = patient_hash_split(pids, val_fraction=0.2, seed=2)
        assert not np.array_equal(val_a, val_b)


class TestPatientHashSplitPatientAware:
    def test_all_exams_of_same_patient_in_same_split(self):
        """Each patient has 3 exams; no patient should have exams in both splits."""
        n_patients = 60
        exams_per_patient = 3
        pids = _make_patient_ids(n_patients, exams_per_patient=exams_per_patient)
        train_idx, val_idx = patient_hash_split(pids, val_fraction=0.2, seed=42)

        train_patients = set(pids[i] for i in train_idx)
        val_patients = set(pids[i] for i in val_idx)
        assert train_patients.isdisjoint(val_patients)

    def test_multi_exam_patient_all_exams_land_together(self):
        """Spot-check a specific patient whose hash lands in val."""
        pids = _make_patient_ids(10, exams_per_patient=5)
        _train_idx, val_idx = patient_hash_split(pids, val_fraction=0.5, seed=99)

        in_val = set(val_idx.tolist())
        for pid in np.unique(pids):
            exam_indices = np.where(pids == pid)[0]
            in_val_flags = [i in in_val for i in exam_indices]
            assert all(in_val_flags) or not any(in_val_flags), (
                f"patient {pid} has exams in both splits"
            )


class TestPatientHashSplitApproximateFraction:
    def test_val_fraction_approximately_correct(self):
        """With ~1000 patients the realised fraction should be within ±5 pp."""
        n_patients = 1000
        val_fraction = 0.2
        pids = _make_patient_ids(n_patients)
        _train_idx, val_idx = patient_hash_split(pids, val_fraction=val_fraction, seed=0)
        actual_fraction = len(val_idx) / n_patients
        assert abs(actual_fraction - val_fraction) < 0.05

    def test_small_val_fraction_most_in_train(self):
        """val_fraction=0.01 should leave the vast majority in train."""
        pids = _make_patient_ids(500)
        train_idx, val_idx = patient_hash_split(pids, val_fraction=0.01, seed=0)
        assert len(train_idx) > len(val_idx)
        assert len(train_idx) / 500 > 0.90


class TestPatientHashSplitEdgeCases:
    def test_single_patient_all_in_one_split(self):
        """Single patient. All exams must end up in one split (train or val)."""
        pids = np.array(["pat_0"] * 5)
        train_idx, val_idx = patient_hash_split(pids, val_fraction=0.2, seed=0)
        assert (len(train_idx) == 5 and len(val_idx) == 0) or (
            len(train_idx) == 0 and len(val_idx) == 5
        )

    def test_single_exam_single_patient(self):
        pids = np.array(["only"])
        train_idx, val_idx = patient_hash_split(pids, val_fraction=0.2, seed=0)
        total = len(train_idx) + len(val_idx)
        assert total == 1


class TestTrainPresetSplitField:
    def test_no_split_field_without_test_path_rejected(self):
        with pytest.raises(ValidationError, match="test_dataset_path"):
            TrainPreset.model_validate(_minimal_train_dict())


def _hdf5_meta(
    *,
    patient_ids: np.ndarray | None = None,
    class_names: list[str] | None = None,
    total_exams: int = 10,
):
    from training.hdf5_meta import HDF5Meta

    names = class_names if class_names is not None else ["ClassA", "ClassB"]
    return HDF5Meta(
        num_classes=len(names),
        num_leads=2,
        class_names=names,
        strat_fold=None,
        exam_labels=np.zeros((total_exams, len(names)), dtype=np.uint8),
        sampling_rate=500,
        total_exams=total_exams,
        total_graphs=total_exams,
        patient_ids=patient_ids,
        node_feature_names=["amplitude", "t_norm_beat"],
        edge_attr_names=["delta_t", "delta_y"],
        node_reduction={
            "method": "uniform",
            "target_nodes": 30,
        },
    )


def _write_compatible_test_h5(path, *, n_exams: int = 4, class_names: list[str] | None = None):
    """Write a current-contract HDF5 compatible with ``_hdf5_meta``."""
    names = tuple(class_names if class_names is not None else ["ClassA", "ClassB"])
    write_current_hdf5(path, n_exams=n_exams, label_names=names, target_nodes=30)


class TestResolveSplits:
    def test_patient_hash_requires_patient_ids(self, tmp_path):
        from training.splits import resolve_splits

        test_h5 = tmp_path / "test.h5"
        _write_compatible_test_h5(test_h5)
        preset = TrainPreset.model_validate(
            _minimal_train_dict(
                split={"strategy": "patient_hash", "val_fraction": 0.2, "seed": 0},
                test_dataset_path=str(test_h5),
            )
        )
        with pytest.raises(RuntimeError, match="exam_patient_id"):
            resolve_splits(preset, _hdf5_meta(patient_ids=None))

    def test_patient_hash_with_test_dataset(self, tmp_path):
        from training.splits import resolve_splits

        test_h5 = tmp_path / "test.h5"
        _write_compatible_test_h5(test_h5, n_exams=4)
        preset = TrainPreset.model_validate(
            _minimal_train_dict(
                split={"strategy": "patient_hash", "val_fraction": 0.2, "seed": 0},
                test_dataset_path=str(test_h5),
            )
        )
        meta = _hdf5_meta(patient_ids=_make_patient_ids(10), total_exams=10)
        result = resolve_splits(preset, meta)

        combined = sorted(np.concatenate([result.train_idx, result.val_idx]).tolist())
        assert combined == list(range(10))
        assert result.test_idx.tolist() == [0, 1, 2, 3]
        assert result.test_meta is not None
        assert result.test_meta.total_exams == 4

    def test_incompatible_test_dataset_rejected(self, tmp_path):
        from training.splits import resolve_splits

        test_h5 = tmp_path / "test.h5"
        _write_compatible_test_h5(test_h5, class_names=["LBBB", "RBBB"])
        preset = TrainPreset.model_validate(_minimal_train_dict(test_dataset_path=str(test_h5)))
        meta = _hdf5_meta(patient_ids=_make_patient_ids(3), total_exams=3)
        with pytest.raises(ValueError, match="label names"):
            resolve_splits(preset, meta)
