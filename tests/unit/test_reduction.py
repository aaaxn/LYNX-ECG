"""Unit tests for the node-reduction strategy layer (``dataset.reduction``)."""

from __future__ import annotations

import numpy as np
import pytest

from config.models import NodeReductionConfig
from dataset.qc import QcContract, validate_selected_output
from dataset.reduction import (
    Beat,
    LeadPrep,
    LeadRejected,
    build_reducer,
)

pytestmark = pytest.mark.unit


def _cfg(d: dict) -> NodeReductionConfig:
    return NodeReductionConfig.model_validate(d)


def _qrs_like(n: int, r_idx: int | None = None) -> np.ndarray:
    r = n // 2 if r_idx is None else r_idx
    x = np.arange(n, dtype=np.float32)
    return np.exp(-((x - r) ** 2) / 8.0).astype(np.float32)


@pytest.mark.parametrize(
    "config, token",
    [
        ({"method": "none"}, "kall"),
        ({"method": "uniform", "uniform": {"target_nodes": 30}}, "k30"),
        ({"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}}, "40hz-original"),
        (
            {
                "method": "turning_points",
                "turning_points": {"cutoff_hz": 40.0, "max_gap_ms": 80.0},
            },
            "40hz-original-mgap80ms",
        ),
    ],
)
def test_filename_token(config, token):
    assert build_reducer(_cfg(config)).filename_token == token


def test_qc_contract_none_expects_all():
    beat = Beat(seg=_qrs_like(40), r_peak_local_idx=20)
    contract = build_reducer(_cfg({"method": "none"})).qc_contract(beat)
    assert contract == QcContract(expected_count=40)


def test_qc_contract_fixed_count_uses_min_of_target_and_n():
    reducer = build_reducer(_cfg({"method": "uniform", "uniform": {"target_nodes": 30}}))
    assert reducer.qc_contract(Beat(seg=_qrs_like(100), r_peak_local_idx=50)) == QcContract(
        expected_count=30
    )
    assert reducer.qc_contract(Beat(seg=_qrs_like(20), r_peak_local_idx=10)) == QcContract(
        expected_count=20
    )


def test_qc_contract_turning_points_is_variable_count():
    reducer = build_reducer(
        _cfg({"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}})
    )
    contract = reducer.qc_contract(Beat(seg=_qrs_like(100), r_peak_local_idx=50))
    assert contract == QcContract(expected_count=None, anchors=(50,))


def test_prepare_lead_default_is_noop():
    reducer = build_reducer(_cfg({"method": "uniform", "uniform": {"target_nodes": 30}}))
    prep = reducer.prepare_lead(_qrs_like(100), sampling_rate=300)
    assert isinstance(prep, LeadPrep)
    assert prep.filtered is None
    assert prep.max_gap_samples is None


def test_prepare_lead_turning_points_filters_and_resolves_gap():
    reducer = build_reducer(
        _cfg(
            {"method": "turning_points", "turning_points": {"cutoff_hz": 40.0, "max_gap_ms": 80.0}}
        )
    )
    signal = _qrs_like(300, r_idx=150)
    prep = reducer.prepare_lead(signal, sampling_rate=300)
    assert isinstance(prep, LeadPrep)
    assert prep.filtered is not None
    assert prep.filtered.shape == signal.shape
    assert prep.filtered.dtype == np.float32
    assert prep.max_gap_samples == 24


def test_prepare_lead_turning_points_rejects_short_lead():
    reducer = build_reducer(
        _cfg({"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}})
    )
    prep = reducer.prepare_lead(_qrs_like(8), sampling_rate=300)
    assert isinstance(prep, LeadRejected)
    assert prep.reason == "signal_too_short_for_filter"
    assert prep.details["signal_length"] == 8


def test_select_uniform_delegates_and_satisfies_its_contract():
    reducer = build_reducer(_cfg({"method": "uniform", "uniform": {"target_nodes": 10}}))
    beat = Beat(seg=_qrs_like(100), r_peak_local_idx=50)
    selected = reducer.select(beat)
    assert selected.dtype == np.int64
    assert selected.size == 10
    contract = reducer.qc_contract(beat)
    assert validate_selected_output(selected, n_beat_samples=100, contract=contract) is None


def test_select_turning_points_requires_filtered_seg():
    reducer = build_reducer(
        _cfg({"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}})
    )
    beat = Beat(seg=_qrs_like(50), r_peak_local_idx=25, filtered_seg=None)
    with pytest.raises(ValueError, match="requires a filtered signal"):
        reducer.select(beat)


def test_node_amplitude_default_uses_raw_segment():
    reducer = build_reducer(_cfg({"method": "uniform", "uniform": {"target_nodes": 10}}))
    seg = _qrs_like(100)
    beat = Beat(seg=seg, r_peak_local_idx=50)
    selected = np.array([0, 50, 99], dtype=np.int64)
    np.testing.assert_array_equal(reducer.node_amplitude(beat, selected), seg[selected])


def test_node_amplitude_turning_points_uses_original_not_lowpassed_segment():
    reducer = build_reducer(
        _cfg({"method": "turning_points", "turning_points": {"cutoff_hz": 40.0}})
    )
    seg = _qrs_like(50)
    filtered = seg * 0.5
    beat = Beat(seg=seg, r_peak_local_idx=25, filtered_seg=filtered)
    selected = np.array([0, 25, 49], dtype=np.int64)
    np.testing.assert_array_equal(reducer.node_amplitude(beat, selected), seg[selected])
