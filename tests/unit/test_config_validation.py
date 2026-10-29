"""Unit tests for the pydantic dataset and training configuration contracts.

Covers the cross-field rules (source_variant <-> sampling_rate registry,
resampling, Nyquist limits, filter bands, reducer sub-configs, QC ranges,
graph_leads, node_features) and checks that every shipped preset validates.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.models import (
    BuildConfig,
    DatasetConfig,
    NodeReductionConfig,
    QCConfig,
    load_config,
)
from config.train_models import load_train_preset

pytestmark = pytest.mark.unit

_TP = {"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}}


def _minimal_dataset_dict(**overrides):
    """A minimal ptbxl/turning_points config that validates."""
    base = {
        "dataset_source": "ptbxl",
        "source_variant": "500hz",
        "sampling_rate": 500,
        "graph_leads": [1],
        "node_reduction": _TP,
    }
    base.update(overrides)
    return base


def _codetest_tp_dict(**overrides):
    """A codetest/turning_points config (the resampling use case)."""
    base = {
        "dataset_source": "codetest",
        "source_variant": "raw_400hz",
        "sampling_rate": 400,
        "graph_leads": [0, 1],
        "node_reduction": _TP,
    }
    base.update(overrides)
    return base


def test_resample_to_hz_sets_effective_rate():
    cfg = DatasetConfig.model_validate(_codetest_tp_dict(resample_to_hz=300))
    assert cfg.sampling_rate == 400
    assert cfg.resample_to_hz == 300
    assert cfg.effective_sampling_rate == 300


def test_effective_rate_defaults_to_sampling_rate_without_resample():
    cfg = DatasetConfig.model_validate(_codetest_tp_dict())
    assert cfg.effective_sampling_rate == 400


def test_resample_to_same_rate_is_rejected():
    with pytest.raises(ValidationError, match="resample_to_hz"):
        DatasetConfig.model_validate(_codetest_tp_dict(resample_to_hz=400))


def test_cutoff_validated_against_effective_nyquist():
    with pytest.raises(ValidationError, match="Nyquist"):
        DatasetConfig.model_validate(_codetest_tp_dict(resample_to_hz=60))


def test_turning_points_cutoff_at_source_nyquist_fails():
    with pytest.raises(ValidationError, match="Nyquist"):
        DatasetConfig.model_validate(
            _minimal_dataset_dict(
                node_reduction={"method": "turning_points", "turning_points": {"cutoff_hz": 250.0}}
            )
        )


def test_signal_filter_rejects_inverted_band():
    with pytest.raises(ValidationError, match="highpass_hz < lowpass_hz"):
        DatasetConfig.model_validate(
            _codetest_tp_dict(signal_filter={"highpass_hz": 40.0, "lowpass_hz": 0.5})
        )


def test_signal_filter_lowpass_validated_against_source_nyquist():
    with pytest.raises(ValidationError, match="Nyquist"):
        DatasetConfig.model_validate(
            _codetest_tp_dict(signal_filter={"highpass_hz": 0.5, "lowpass_hz": 250.0})
        )


def test_turning_points_missing_subconfig_fails():
    with pytest.raises(ValidationError, match="requires a 'turning_points' sub-config"):
        DatasetConfig.model_validate(
            _minimal_dataset_dict(node_reduction={"method": "turning_points"})
        )


def test_wrong_subconfig_for_method_fails():
    with pytest.raises(ValidationError, match="does not accept 'uniform'"):
        DatasetConfig.model_validate(
            _minimal_dataset_dict(
                node_reduction={
                    "method": "turning_points",
                    "turning_points": {},
                    "uniform": {"target_nodes": 30},
                }
            )
        )


def test_none_with_subconfig_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NodeReductionConfig.model_validate({"method": "none", "uniform": {"target_nodes": 30}})
    assert any("ignored" in str(w.message) for w in caught)


def test_qc_rpeak_hr_range_inverted_fails():
    with pytest.raises(ValidationError):
        QCConfig.model_validate({"rpeak_hr_range_bpm": [240, 30]})


def test_qc_beat_rr_range_inverted_fails():
    with pytest.raises(ValidationError):
        QCConfig.model_validate({"beat_rr_range_seconds": [2.0, 0.25]})


def test_build_jobs_zero_fails():
    with pytest.raises(ValidationError):
        BuildConfig.model_validate({"jobs": 0})


def test_build_jobs_minus_one_passes():
    cfg = BuildConfig.model_validate({"jobs": -1})
    assert cfg.jobs == -1


def test_build_jobs_minus_two_fails():
    with pytest.raises(ValidationError):
        BuildConfig.model_validate({"jobs": -2})


def test_unknown_source_variant_fails():
    bad = _minimal_dataset_dict(source_variant="999hz")
    with pytest.raises(ValidationError) as exc:
        DatasetConfig.model_validate(bad)
    assert "source_variant" in str(exc.value).lower()


def test_sampling_rate_mismatch_fails():
    bad = _minimal_dataset_dict(sampling_rate=400)
    with pytest.raises(ValidationError) as exc:
        DatasetConfig.model_validate(bad)
    assert "sampling_rate" in str(exc.value).lower()


def test_graph_leads_empty_fails():
    bad = _minimal_dataset_dict(graph_leads=[])
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(bad)


def test_graph_leads_negative_fails():
    bad = _minimal_dataset_dict(graph_leads=[-1, 0])
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(bad)


def test_graph_leads_duplicates_fails():
    bad = _minimal_dataset_dict(graph_leads=[1, 1, 2])
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(bad)


def test_node_features_without_amplitude_fails():
    bad = _minimal_dataset_dict(node_features=["t_norm_beat"])
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(bad)


def test_node_features_amplitude_not_first_fails():
    bad = _minimal_dataset_dict(node_features=["t_norm_beat", "amplitude"])
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(bad)


def test_limit_exams_zero_fails():
    bad = _minimal_dataset_dict(limit_exams=0)
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(bad)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


SHIPPED_PRESETS = [
    (path.parent.name, path.stem)
    for path in sorted(
        (Path(__file__).resolve().parents[2] / "configs" / "dataset").rglob("*.json")
    )
]


@pytest.mark.parametrize("subdir,preset", SHIPPED_PRESETS)
def test_shipped_preset_validates(subdir: str, preset: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "configs" / "dataset" / subdir / f"{preset}.json"
    assert path.exists(), f"missing preset: {path}"
    load_config(path)


TRAIN_PRESET_PATHS = sorted((_repo_root() / "configs" / "train").rglob("*.json"))


def test_train_presets_discovered() -> None:
    assert TRAIN_PRESET_PATHS, "no training presets found under configs/train"


@pytest.mark.parametrize("path", TRAIN_PRESET_PATHS, ids=lambda p: str(p.relative_to(_repo_root())))
def test_shipped_train_preset_validates(path: Path) -> None:
    cfg = load_train_preset(path)
    assert cfg.results_dir == "results"
    assert not Path(cfg.dataset_path).is_absolute()
    assert cfg.test_dataset_path is None or not Path(cfg.test_dataset_path).is_absolute()
