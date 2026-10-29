"""Unit tests for the QC reason functions and SkipEvent serialization."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.qc import (
    QcContract,
    SkipEvent,
    validate_beat_anchor,
    validate_beat_signal,
    validate_rpeaks,
    validate_selected_output,
)

pytestmark = pytest.mark.unit


def test_rpeaks_too_few():
    peaks = np.array([100], dtype=np.int64)
    assert (
        validate_rpeaks(
            peaks, signal_length=4000, sampling_rate=400, min_count=3, hr_range_bpm=(30, 240)
        )
        == "rpeak_too_few_peaks"
    )


def test_rpeaks_non_monotonic():
    peaks = np.array([100, 50, 200], dtype=np.int64)
    assert validate_rpeaks(peaks, 4000, 400, 3, (30, 240)) == "rpeak_non_monotonic"


def test_rpeaks_out_of_bounds_max():
    peaks = np.array([100, 200, 5000], dtype=np.int64)
    assert validate_rpeaks(peaks, 4000, 400, 3, (30, 240)) == "rpeak_out_of_bounds"


def test_rpeaks_out_of_bounds_min():
    peaks = np.array([-1, 100, 200], dtype=np.int64)
    assert validate_rpeaks(peaks, 4000, 400, 3, (30, 240)) == "rpeak_out_of_bounds"


def test_rpeaks_hr_too_low():
    peaks = np.array([0, 1600, 3200, 4800], dtype=np.int64)
    assert validate_rpeaks(peaks, 5000, 400, 3, (30, 240)) == "rpeak_hr_out_of_range"


def test_rpeaks_hr_too_high():
    peaks = np.array([0, 80, 160, 240], dtype=np.int64)
    assert validate_rpeaks(peaks, 4000, 400, 3, (30, 240)) == "rpeak_hr_out_of_range"


def test_rpeaks_valid_passes():
    peaks = np.array([200, 543, 886, 1229, 1572], dtype=np.int64)
    assert validate_rpeaks(peaks, 4000, 400, 3, (30, 240)) is None


def test_beat_signal_too_few_samples():
    y = np.array([0.1, 0.2], dtype=np.float32)
    assert validate_beat_signal(y, 0.005, (0.25, 2.0), 1e-4, min_samples=3) == (
        "beat_too_few_samples"
    )


def test_beat_signal_nan():
    y = np.array([0.1, np.nan, 0.3, 0.4], dtype=np.float32)
    assert validate_beat_signal(y, 0.5, (0.25, 2.0), 1e-4, 3) == "beat_nan_or_inf"


def test_beat_signal_inf():
    y = np.array([0.1, np.inf, 0.3, 0.4], dtype=np.float32)
    assert validate_beat_signal(y, 0.5, (0.25, 2.0), 1e-4, 3) == "beat_nan_or_inf"


def test_beat_signal_duration_too_short():
    y = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert validate_beat_signal(y, 0.10, (0.25, 2.0), 1e-4, 3) == "beat_duration_out_of_range"


def test_beat_signal_duration_too_long():
    y = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    assert validate_beat_signal(y, 5.0, (0.25, 2.0), 1e-4, 3) == "beat_duration_out_of_range"


def test_beat_signal_flat_amplitude():
    y = np.full(50, 0.5, dtype=np.float32)
    assert validate_beat_signal(y, 0.5, (0.25, 2.0), 1e-4, 3) == "beat_flat_amplitude"


def test_beat_signal_valid_passes():
    y = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    assert validate_beat_signal(y, 0.5, (0.25, 2.0), 1e-4, 3) is None


def test_beat_signal_duration_skipped_for_boundary_when_enabled():
    """Boundary beats (first/last/only) bypass the RR range check when flagged."""
    y = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    for bt in (1, 2, 3):
        assert (
            validate_beat_signal(
                y,
                5.0,
                (0.25, 2.0),
                1e-4,
                3,
                boundary_type=bt,
                skip_duration_check_for_boundary=True,
            )
            is None
        )


def test_beat_signal_duration_still_checked_for_middle_when_flag_set():
    """Middle beats (boundary_type=0) keep the duration check even when flag is on."""
    y = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    assert (
        validate_beat_signal(
            y,
            5.0,
            (0.25, 2.0),
            1e-4,
            3,
            boundary_type=0,
            skip_duration_check_for_boundary=True,
        )
        == "beat_duration_out_of_range"
    )


def test_beat_signal_boundary_still_checked_when_flag_off():
    """Default (flag off) preserves legacy behavior: boundary beats are duration-checked."""
    y = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    assert (
        validate_beat_signal(y, 5.0, (0.25, 2.0), 1e-4, 3, boundary_type=1)
        == "beat_duration_out_of_range"
    )


def test_beat_anchor_inside():
    assert validate_beat_anchor(start_sample=10, r_peak_sample=15, end_sample=20) is None


def test_beat_anchor_at_boundary():
    assert validate_beat_anchor(10, 10, 20) is None
    assert validate_beat_anchor(10, 20, 20) is None


def test_beat_anchor_below_start():
    assert validate_beat_anchor(10, 5, 20) == "beat_anchor_out_of_bounds"


def test_beat_anchor_above_end():
    assert validate_beat_anchor(10, 25, 20) == "beat_anchor_out_of_bounds"


def test_selected_output_fixed_count_match():
    selected = np.array([0, 11, 22, 33, 44, 55, 66, 77, 88, 99], dtype=np.int64)
    contract = QcContract(expected_count=10)
    assert validate_selected_output(selected, n_beat_samples=100, contract=contract) is None


def test_selected_output_underflow_count():
    selected = np.arange(15, dtype=np.int64)
    contract = QcContract(expected_count=15)
    assert validate_selected_output(selected, n_beat_samples=15, contract=contract) is None


def test_selected_output_wrong_count():
    selected = np.array([0, 33, 99], dtype=np.int64)
    contract = QcContract(expected_count=10)
    assert (
        validate_selected_output(selected, n_beat_samples=100, contract=contract)
        == "pip_invalid_output"
    )


def test_selected_output_anchors_present():
    selected = np.array([0, 20, 25, 47, 50, 53, 70, 78, 85, 99], dtype=np.int64)
    contract = QcContract(expected_count=10, anchors=(0, 50, 99))
    assert validate_selected_output(selected, n_beat_samples=100, contract=contract) is None


def test_selected_output_missing_anchor():
    selected = np.array([0, 20, 25, 47, 53, 70, 78, 85, 90, 99], dtype=np.int64)
    contract = QcContract(expected_count=10, anchors=(0, 50, 99))
    assert (
        validate_selected_output(selected, n_beat_samples=100, contract=contract)
        == "pip_invalid_output"
    )


def test_selected_output_dtype_must_be_int64():
    selected = np.array([0, 10, 20], dtype=np.int32)
    contract = QcContract(expected_count=30)
    assert (
        validate_selected_output(selected, n_beat_samples=30, contract=contract)
        == "pip_invalid_output"
    )


def test_selected_output_non_monotonic():
    selected = np.array([0, 50, 25, 99], dtype=np.int64)
    contract = QcContract(expected_count=4, anchors=(0, 99))
    assert (
        validate_selected_output(selected, n_beat_samples=100, contract=contract)
        == "pip_invalid_output"
    )


def test_selected_output_index_out_of_range():
    selected = np.array([0, 50, 99, 200], dtype=np.int64)
    contract = QcContract(expected_count=4, anchors=(0, 99))
    assert (
        validate_selected_output(selected, n_beat_samples=100, contract=contract)
        == "pip_invalid_output"
    )


def test_selected_output_no_exact_count_when_none():
    selected = np.array([0, 10, 20, 49], dtype=np.int64)
    contract = QcContract(expected_count=None, anchors=(10,))
    assert validate_selected_output(selected, n_beat_samples=50, contract=contract) is None


def test_selected_output_keeps_all_for_none_reducer():
    selected = np.arange(50, dtype=np.int64)
    contract = QcContract(expected_count=50)
    assert validate_selected_output(selected, n_beat_samples=50, contract=contract) is None


def test_selected_output_partial_fails_for_none_reducer():
    selected = np.array([0, 10, 20, 49], dtype=np.int64)
    contract = QcContract(expected_count=50)
    assert (
        validate_selected_output(selected, n_beat_samples=50, contract=contract)
        == "pip_invalid_output"
    )


def test_skip_event_to_dict_minimal():
    e = SkipEvent(level="exam", reason="exam_no_valid_leads")
    d = e.to_dict()
    assert d == {"level": "exam", "reason": "exam_no_valid_leads", "details": {}}


def test_skip_event_to_dict_full():
    e = SkipEvent(
        level="beat",
        reason="beat_flat_amplitude",
        exam_id="12345",
        lead=1,
        beat_index=7,
        details={"amplitude_range": 0.00003, "beat_min_amplitude": 0.0001},
    )
    d = e.to_dict()
    assert d["level"] == "beat"
    assert d["exam_id"] == "12345"
    assert d["lead"] == 1
    assert d["beat_index"] == 7
    assert d["details"]["amplitude_range"] == 0.00003
