"""Small integration coverage for the supported NeuroKit delineation adapters."""

from __future__ import annotations

import numpy as np
import pytest

from explain.delineation import DelineationMethod, delineate_lead

pytestmark = pytest.mark.integration


def test_dwt_and_prominence_adapters_accept_explicit_r_peaks_without_redetection(
    synthetic_ecg_clean: np.ndarray,
):
    import neurokit2 as nk

    sampling_rate = 400
    _signals, peak_info = nk.ecg_peaks(synthetic_ecg_clean, sampling_rate=sampling_rate)
    stored_r_peaks = np.asarray(peak_info["ECG_R_Peaks"], dtype=np.int64)

    dwt = delineate_lead(
        synthetic_ecg_clean,
        r_peak_samples=stored_r_peaks,
        sampling_rate=sampling_rate,
        method=DelineationMethod.DWT,
        candidate="dwt",
        polarity=1,
    )
    prominence = delineate_lead(
        synthetic_ecg_clean,
        r_peak_samples=stored_r_peaks,
        sampling_rate=sampling_rate,
        method=DelineationMethod.PROMINENCE,
        candidate="default",
        polarity=1,
        prominence_kwargs={
            "max_qrs_interval": 180,
            "max_pr_interval": 300,
            "max_r_rise_time": 120,
            "typical_st_segment": 150,
            "max_p_basepoint_interval": 100,
            "max_r_basepoint_interval": 100,
            "max_t_basepoint_interval": 200,
        },
    )

    expected = tuple(int(value) for value in stored_r_peaks)
    assert dwt.r_peak_samples == expected
    assert prominence.r_peak_samples == expected
    assert len(dwt.beats) == len(stored_r_peaks)
    assert len(prominence.beats) == len(stored_r_peaks)
    assert dwt.error_detail is None
    assert prominence.error_detail is None
    assert any(dwt.complete_mask)
    assert any(prominence.complete_mask)


def test_prominence_adapter_records_negative_polarity_correction_without_moving_indices(
    synthetic_ecg_inverted: np.ndarray,
):
    import neurokit2 as nk

    sampling_rate = 400
    _signals, peak_info = nk.ecg_peaks(-synthetic_ecg_inverted, sampling_rate=sampling_rate)
    stored_r_peaks = np.asarray(peak_info["ECG_R_Peaks"], dtype=np.int64)

    result = delineate_lead(
        synthetic_ecg_inverted,
        r_peak_samples=stored_r_peaks,
        sampling_rate=sampling_rate,
        method=DelineationMethod.PROMINENCE,
        candidate="default",
        polarity=-1,
    )

    assert result.r_peak_samples == tuple(int(value) for value in stored_r_peaks)
    assert result.polarity == -1
    assert result.polarity_corrected is True
    assert result.error_detail is None
    assert any(result.complete_mask)
