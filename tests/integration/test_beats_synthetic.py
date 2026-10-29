"""Integration tests for beat segmentation on synthetic ECG."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.beats import BoundaryType, segment_beats
from dataset.qc import validate_beat_anchor, validate_beat_signal
from dataset.rpeaks import RPeakResult, detect_rpeaks

pytestmark = pytest.mark.integration


def test_segment_basic_boundaries(synthetic_ecg_clean):
    res = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    assert isinstance(res, RPeakResult)
    spans = segment_beats(synthetic_ecg_clean.size, res.peaks)
    assert len(spans) == res.peaks.size

    assert spans[0].boundary_type == BoundaryType.FIRST
    assert spans[-1].boundary_type == BoundaryType.LAST
    for s in spans[1:-1]:
        assert s.boundary_type == BoundaryType.MIDDLE


def test_segment_first_beat_starts_at_zero(synthetic_ecg_clean):
    res = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    spans = segment_beats(synthetic_ecg_clean.size, res.peaks)
    assert spans[0].start_sample == 0


def test_segment_last_beat_ends_at_signal_end(synthetic_ecg_clean):
    res = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    spans = segment_beats(synthetic_ecg_clean.size, res.peaks)
    assert spans[-1].end_sample == synthetic_ecg_clean.size - 1


def test_segment_no_overlap_no_gap(synthetic_ecg_clean):
    """Adjacent beats must touch exactly: end_k + 1 == start_{k+1}."""
    res = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    spans = segment_beats(synthetic_ecg_clean.size, res.peaks)
    for k in range(len(spans) - 1):
        assert spans[k].end_sample + 1 == spans[k + 1].start_sample


def test_segment_anchors_inside_spans(synthetic_ecg_clean):
    res = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    spans = segment_beats(synthetic_ecg_clean.size, res.peaks)
    for s in spans:
        assert validate_beat_anchor(s.start_sample, s.r_peak_sample, s.end_sample) is None


def test_segment_local_idx_consistent(synthetic_ecg_clean):
    res = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    spans = segment_beats(synthetic_ecg_clean.size, res.peaks)
    for s in spans:
        assert s.r_peak_local_idx == s.r_peak_sample - s.start_sample
        assert 0 <= s.r_peak_local_idx < s.num_samples


def test_segment_zero_peaks_returns_empty():
    spans = segment_beats(signal_length=4000, peaks=np.array([], dtype=np.int64))
    assert spans == []


def test_segment_one_peak_yields_only_beat():
    spans = segment_beats(signal_length=4000, peaks=np.array([2000], dtype=np.int64))
    assert len(spans) == 1
    s = spans[0]
    assert s.boundary_type == BoundaryType.ONLY_BEAT
    assert s.start_sample == 0
    assert s.end_sample == 3999
    assert s.r_peak_sample == 2000
    assert s.r_peak_local_idx == 2000


def test_segment_two_peaks():
    spans = segment_beats(signal_length=4000, peaks=np.array([1000, 3000], dtype=np.int64))
    assert len(spans) == 2
    a, b = spans
    assert a.boundary_type == BoundaryType.FIRST
    assert b.boundary_type == BoundaryType.LAST
    assert a.start_sample == 0
    assert b.end_sample == 3999
    assert a.end_sample == (1000 + 3000) // 2 - 1
    assert b.start_sample == (1000 + 3000) // 2


def test_segment_beat_qc_passes_for_clean_signal(synthetic_ecg_clean):
    res = detect_rpeaks(synthetic_ecg_clean, sampling_rate=400)
    spans = segment_beats(synthetic_ecg_clean.size, res.peaks)
    sampling_rate = 400
    for s in spans:
        seg = synthetic_ecg_clean[s.start_sample : s.end_sample + 1]
        duration = (s.end_sample - s.start_sample + 1) / sampling_rate
        reason = validate_beat_signal(
            seg,
            duration_seconds=duration,
            rr_range_seconds=(0.25, 2.0),
            min_amplitude=1e-4,
            min_samples=3,
        )
        assert reason is None, f"middle beat failed QC: {reason}"


def test_segment_beat_qc_flat_segment_flagged():
    """A flat slice through validate_beat_signal must be flagged."""
    flat = np.full(200, 0.5, dtype=np.float32)
    reason = validate_beat_signal(flat, 0.5, (0.25, 2.0), 1e-4, 3)
    assert reason == "beat_flat_amplitude"


def test_segment_inverted_signal_indices_valid(synthetic_ecg_inverted):
    """Indices stay valid for the original signal regardless of which polarity won."""
    res = detect_rpeaks(synthetic_ecg_inverted, sampling_rate=400)
    assert res.polarity in (1, -1)
    spans = segment_beats(synthetic_ecg_inverted.size, res.peaks)
    for s in spans:
        assert 0 <= s.r_peak_sample < synthetic_ecg_inverted.size
        assert s.start_sample <= s.r_peak_sample <= s.end_sample
