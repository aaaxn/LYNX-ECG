"""Determinism and content tests for the dataset filename derivation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from config.models import DatasetConfig, load_config
from dataset.filename import (
    derive_dataset_filename,
    derive_dataset_filename_base,
    derive_output_path,
    derive_reports_dir,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
LEADS = "leads-0-1-2-3-4-5-6-7-8-9-10-11"

# The file names the paper dataset configs build. Train configs and the README
# quick check name these files, so a change here breaks the paper workflow.
PAPER_DATASET_FILENAMES = {
    "none/code100_none_all12_bp_norm.json": (
        f"code100__none__kall__wfdb_300hz__bp0p5-40hz-o4__norm-p95med__natural__{LEADS}.h5"
    ),
    "none/codetest_none_all12_raw_resample300_bp_norm.json": (
        "codetest__none__kall__raw_400hz__bp0p5-40hz-o4__resample-300hz__norm-p95med__"
        f"natural__{LEADS}.h5"
    ),
    "none/ptbxl_none_all12_500hz_resample300_bp_norm_code6strict.json": (
        "ptbxl__none__kall__500hz__code6strict__bp0p5-40hz-o4__resample-300hz__norm-p95med__"
        f"natural__{LEADS}.h5"
    ),
    "smoke/ptbxl_tp_smoke_200.json": (
        "ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__bp0p5-40hz-o4__"
        f"resample-300hz__norm-p95med__natural__{LEADS}__limit-200.h5"
    ),
    "turning_points/code100_tp_40hz_12lead_mgap80_bp_norm.json": (
        "code100__turning_points__40hz-original-mgap80ms__wfdb_300hz__bp0p5-40hz-o4__"
        f"norm-p95med__natural__{LEADS}.h5"
    ),
    "turning_points/codetest_tp_40hz_all12_mgap80_raw_resample300_bp_norm.json": (
        "codetest__turning_points__40hz-original-mgap80ms__raw_400hz__bp0p5-40hz-o4__"
        f"resample-300hz__norm-p95med__natural__{LEADS}.h5"
    ),
    "turning_points/ptbxl_tp_40hz_all12_mgap80_500hz_resample300_bp_norm_code6strict.json": (
        "ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__bp0p5-40hz-o4__"
        f"resample-300hz__norm-p95med__natural__{LEADS}.h5"
    ),
    "uniform/code100_uniform_k30_all12_bp_norm.json": (
        f"code100__uniform__k30__wfdb_300hz__bp0p5-40hz-o4__norm-p95med__natural__{LEADS}.h5"
    ),
    "uniform/codetest_uniform_k30_all12_bp_norm.json": (
        "codetest__uniform__k30__raw_400hz__bp0p5-40hz-o4__resample-300hz__norm-p95med__"
        f"natural__{LEADS}.h5"
    ),
}


def _dataset_config_paths() -> list[str]:
    root = REPO_ROOT / "configs" / "dataset"
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.json"))


def _built_paths() -> set[str]:
    return {
        str(
            derive_output_path(
                load_config(REPO_ROOT / "configs" / "dataset" / rel), repo_root=Path()
            )
        )
        for rel in _dataset_config_paths()
    }


def test_golden_table_covers_every_dataset_config():
    assert sorted(PAPER_DATASET_FILENAMES) == _dataset_config_paths()


@pytest.mark.parametrize("rel_path", sorted(PAPER_DATASET_FILENAMES))
def test_paper_dataset_config_filename(rel_path):
    cfg = load_config(REPO_ROOT / "configs" / "dataset" / rel_path)
    assert derive_dataset_filename(cfg) == PAPER_DATASET_FILENAMES[rel_path]


@pytest.mark.parametrize(
    "train_config",
    sorted(
        str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "configs" / "train").rglob("*.json")
    ),
)
def test_train_config_names_a_built_dataset(train_config):
    payload = json.loads((REPO_ROOT / train_config).read_text())
    built = _built_paths()
    assert payload["dataset_path"] in built
    assert payload["test_dataset_path"] in built


def test_readme_quick_check_names_the_smoke_build():
    readme = (REPO_ROOT / "README.md").read_text()
    match = re.search(r"^H5=(\S+)$", readme, re.MULTILINE)
    assert match is not None
    cfg = load_config(REPO_ROOT / "configs" / "dataset" / "smoke" / "ptbxl_tp_smoke_200.json")
    assert match.group(1) == str(derive_output_path(cfg, repo_root=Path()))


def _cfg(**overrides) -> DatasetConfig:
    base = {
        "dataset_source": "ptbxl",
        "source_variant": "500hz",
        "sampling_rate": 500,
        "graph_leads": [0, 1, 11],
        "node_reduction": {
            "method": "turning_points",
            "turning_points": {"cutoff_hz": 40.0, "max_gap_ms": 80.0},
        },
    }
    base.update(overrides)
    return DatasetConfig.model_validate(base)


def _tp(**params) -> dict:
    return {"method": "turning_points", "turning_points": params}


def test_turning_points_filename():
    assert derive_dataset_filename(_cfg()) == (
        "ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__natural__"
        "leads-0-1-11.h5"
    )


def test_uniform_filename():
    cfg = _cfg(node_reduction={"method": "uniform", "uniform": {"target_nodes": 30}})
    assert derive_dataset_filename(cfg) == (
        "ptbxl__uniform__k30__500hz__code6strict__natural__leads-0-1-11.h5"
    )


def test_code6strict_token_only_for_ptbxl():
    cfg = _cfg(dataset_source="codetest", source_variant="raw_400hz", sampling_rate=400)
    assert "code6strict" not in derive_dataset_filename(cfg)


def test_resample_token_present_after_variant():
    cfg = _cfg(resample_to_hz=300)
    assert derive_dataset_filename(cfg) == (
        "ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__resample-300hz__"
        "natural__leads-0-1-11.h5"
    )


def test_no_resample_omits_resample_token():
    assert "resample-" not in derive_dataset_filename(_cfg())


def test_signal_filter_token_present_after_variant_before_resample():
    cfg = _cfg(
        signal_filter={"highpass_hz": 0.5, "lowpass_hz": 40.0, "order": 4},
        resample_to_hz=300,
    )
    assert derive_dataset_filename(cfg) == (
        "ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__bp0p5-40hz-o4__"
        "resample-300hz__natural__leads-0-1-11.h5"
    )


def test_no_signal_filter_omits_token():
    assert "__bp" not in derive_dataset_filename(_cfg())


def test_normalization_token_present_without_filter_or_resample():
    cfg = _cfg(amplitude_normalization={"method": "p95_lead_median"})
    assert derive_dataset_filename(cfg) == (
        "ptbxl__turning_points__40hz-original-mgap80ms__500hz__code6strict__norm-p95med__"
        "natural__leads-0-1-11.h5"
    )


def test_no_normalization_omits_token():
    assert "__norm-" not in derive_dataset_filename(_cfg())


def test_none_uses_kall():
    cfg = _cfg(node_reduction={"method": "none"})
    assert "__kall__" in derive_dataset_filename(cfg)


def test_smoke_includes_limit_token():
    assert derive_dataset_filename(_cfg(limit_exams=100)).endswith("__limit-100.h5")


def test_no_limit_omits_limit_token():
    assert "limit-" not in derive_dataset_filename(_cfg())


def test_lead_order_does_not_change_filename():
    a = _cfg(graph_leads=[11, 0, 1])
    b = _cfg(graph_leads=[0, 1, 11])
    assert derive_dataset_filename(a) == derive_dataset_filename(b)


def test_different_method_different_name():
    a = _cfg()
    b = _cfg(node_reduction={"method": "uniform", "uniform": {"target_nodes": 30}})
    assert derive_dataset_filename(a) != derive_dataset_filename(b)


def test_different_target_different_name():
    a = _cfg(node_reduction={"method": "uniform", "uniform": {"target_nodes": 30}})
    b = _cfg(node_reduction={"method": "uniform", "uniform": {"target_nodes": 50}})
    assert derive_dataset_filename(a) != derive_dataset_filename(b)


def test_different_tp_params_different_name():
    a = _cfg(node_reduction=_tp(cutoff_hz=40.0))
    b = _cfg(node_reduction=_tp(cutoff_hz=20.0))
    assert derive_dataset_filename(a) != derive_dataset_filename(b)


def test_turning_points_no_max_gap_no_token():
    assert "mgap" not in derive_dataset_filename(_cfg(node_reduction=_tp(cutoff_hz=40.0)))


def test_different_max_gap_different_name():
    a = _cfg(node_reduction=_tp(cutoff_hz=40.0, max_gap_ms=80.0))
    b = _cfg(node_reduction=_tp(cutoff_hz=40.0, max_gap_ms=60.0))
    assert derive_dataset_filename(a) != derive_dataset_filename(b)


def test_filename_base_strips_extension():
    cfg = _cfg()
    base = derive_dataset_filename_base(cfg)
    assert not base.endswith(".h5")
    assert derive_dataset_filename(cfg) == base + ".h5"


def test_output_path_under_graph_datasets(tmp_path):
    cfg = _cfg()
    p = derive_output_path(cfg, repo_root=tmp_path)
    assert p == tmp_path / "graph_datasets" / "ptbxl" / derive_dataset_filename(cfg)


def test_output_path_explicit_override(tmp_path):
    custom = tmp_path / "custom" / "out.h5"
    cfg = _cfg(output_path=str(custom))
    assert derive_output_path(cfg, repo_root=tmp_path) == custom


def test_reports_dir_under_reports(tmp_path):
    cfg = _cfg()
    rd = derive_reports_dir(cfg, repo_root=tmp_path)
    assert rd == tmp_path / "reports" / derive_dataset_filename_base(cfg)
