"""Public delineation contracts for complete PR-QRS-T triplets."""

from __future__ import annotations

import numpy as np
import pytest

from explain.delineation import (
    BeatDelineation,
    DelineationFailureReason,
    DelineationMethod,
    MorphologyLandmarks,
    adapt_neurokit_waves,
    ordering_violations,
    prepare_delineation_signal,
    region_availability,
)
from explain.ecg_taxonomy import ECGRegion

pytestmark = pytest.mark.unit

_REQUIRED_WAVES = {
    "ECG_P_Onsets": [60, 160],
    "ECG_R_Onsets": [80, 180],
    "ECG_R_Offsets": [120, 220],
    "ECG_T_Onsets": [140, 240],
    "ECG_T_Offsets": [170, 270],
}


def test_adapt_neurokit_waves_returns_complete_ordered_triplets():
    result = adapt_neurokit_waves(
        _REQUIRED_WAVES,
        r_peak_samples=np.array([100, 200]),
        signal_length=300,
        sampling_rate=100,
        method=DelineationMethod.DWT,
        candidate="dwt",
        polarity=-1,
        polarity_corrected=False,
    )

    assert result.r_peak_samples == (100, 200)
    assert result.polarity == -1
    assert result.polarity_corrected is False
    assert result.complete_mask == (True, True)
    assert result.beats[0].triplet is not None
    assert result.beats[0].triplet.p_onset_sample == 60
    assert result.beats[0].triplet.qrs_onset_sample == 80
    assert result.beats[0].triplet.qrs_offset_sample == 120
    assert result.beats[0].triplet.t_onset_sample == 140
    assert result.beats[0].triplet.t_offset_sample == 170
    assert result.beats[0].triplet.durations_ms(sampling_rate=100) == {
        "pr": 200.0,
        "qrs": 400.0,
        "t": 300.0,
    }


def test_adapt_neurokit_waves_reports_missing_and_ordering_reasons_without_imputation():
    waves = {key: list(values) for key, values in _REQUIRED_WAVES.items()}
    waves["ECG_P_Onsets"][0] = np.nan
    waves["ECG_R_Offsets"][1] = 170

    result = adapt_neurokit_waves(
        waves,
        r_peak_samples=np.array([100, 200]),
        signal_length=300,
        sampling_rate=100,
        method=DelineationMethod.PROMINENCE,
        candidate="default",
        polarity=1,
        polarity_corrected=False,
    )

    assert result.complete_mask == (False, False)
    assert result.beats[0].triplet is None
    assert result.beats[0].reasons == (DelineationFailureReason.MISSING_P_ONSET,)
    assert result.beats[1].triplet is None
    assert DelineationFailureReason.QRS_OFFSET_NOT_AFTER_R_PEAK in result.beats[1].reasons


def test_adapt_neurokit_waves_rejects_missing_keys_and_output_length_mismatch_per_field():
    waves = {key: list(values) for key, values in _REQUIRED_WAVES.items()}
    del waves["ECG_T_Onsets"]
    waves["ECG_P_Onsets"] = [60]

    result = adapt_neurokit_waves(
        waves,
        r_peak_samples=np.array([100, 200]),
        signal_length=300,
        sampling_rate=100,
        method=DelineationMethod.DWT,
        candidate="dwt",
        polarity=1,
        polarity_corrected=False,
    )

    for beat in result.beats:
        assert DelineationFailureReason.MISSING_T_ONSET_OUTPUT in beat.reasons
        assert DelineationFailureReason.P_ONSET_OUTPUT_LENGTH_MISMATCH in beat.reasons
        assert beat.triplet is None


def test_adapt_neurokit_waves_rejects_inconsistent_polarity_metadata():
    with pytest.raises(ValueError, match="polarity_corrected disagrees"):
        adapt_neurokit_waves(
            _REQUIRED_WAVES,
            r_peak_samples=np.array([100, 200]),
            signal_length=300,
            sampling_rate=100,
            method=DelineationMethod.DWT,
            candidate="dwt",
            polarity=-1,
            polarity_corrected=True,
        )


def test_prepare_delineation_signal_applies_polarity_only_to_prominence():
    signal = np.array([1.0, -2.0, 3.0])
    original = signal.copy()

    dwt_signal, dwt_corrected = prepare_delineation_signal(
        signal, method=DelineationMethod.DWT, polarity=-1
    )
    prominence_signal, prominence_corrected = prepare_delineation_signal(
        signal, method=DelineationMethod.PROMINENCE, polarity=-1
    )

    assert np.array_equal(dwt_signal, original)
    assert dwt_corrected is False
    assert np.array_equal(prominence_signal, -original)
    assert prominence_corrected is True
    assert np.array_equal(signal, original)


def test_region_availability_partial_landmarks_keep_other_regions():
    landmarks = MorphologyLandmarks(
        qrs_onset_sample=90,
        qrs_offset_sample=120,
        t_onset_sample=150,
        t_offset_sample=200,
    )
    availability = region_availability(landmarks, r_peak_sample=100)
    assert availability == {ECGRegion.PR: False, ECGRegion.QRS: True, ECGRegion.T: True}


def test_region_availability_qrs_t_overlap_blocks_t_not_qrs():
    overlapping = MorphologyLandmarks(
        p_onset_sample=10,
        qrs_onset_sample=90,
        qrs_offset_sample=160,
        t_onset_sample=150,
        t_offset_sample=200,
    )
    availability = region_availability(overlapping, r_peak_sample=100)
    assert availability[ECGRegion.QRS] is True
    assert availability[ECGRegion.T] is False
    without_qrs_offset = MorphologyLandmarks(
        p_onset_sample=10,
        qrs_onset_sample=90,
        t_onset_sample=150,
        t_offset_sample=200,
    )
    assert region_availability(without_qrs_offset, r_peak_sample=100)[ECGRegion.T] is True


def test_region_availability_full_chain_matches_completeness():
    ordered = MorphologyLandmarks(
        p_onset_sample=10,
        qrs_onset_sample=90,
        qrs_offset_sample=120,
        t_onset_sample=150,
        t_offset_sample=200,
    )
    beat = BeatDelineation(r_peak_sample=100, landmarks=ordered, reasons=())
    assert beat.complete
    assert all(beat.available_regions.values())
    misordered = MorphologyLandmarks(
        p_onset_sample=95,
        qrs_onset_sample=90,
        qrs_offset_sample=120,
        t_onset_sample=150,
        t_offset_sample=200,
    )
    partial_beat = BeatDelineation(r_peak_sample=100, landmarks=misordered, reasons=())
    assert not partial_beat.complete
    assert partial_beat.available_regions == {
        ECGRegion.PR: False,
        ECGRegion.QRS: True,
        ECGRegion.T: True,
    }


def test_ordering_violations_name_every_blocked_region():
    overlapping = MorphologyLandmarks(
        p_onset_sample=95,
        qrs_onset_sample=90,
        qrs_offset_sample=160,
        t_onset_sample=150,
        t_offset_sample=140,
    )
    reasons = ordering_violations(overlapping, r_peak_sample=100)
    assert reasons == (
        DelineationFailureReason.P_ONSET_NOT_BEFORE_QRS_ONSET,
        DelineationFailureReason.QRS_OFFSET_NOT_BEFORE_T_ONSET,
        DelineationFailureReason.T_ONSET_NOT_BEFORE_T_OFFSET,
    )
    availability = region_availability(overlapping, r_peak_sample=100)
    assert availability == {ECGRegion.PR: False, ECGRegion.QRS: True, ECGRegion.T: False}
