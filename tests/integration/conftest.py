"""Integration-test fixtures: re-export synthetic ECG generators."""

from __future__ import annotations

from tests.fixtures.synthetic_ecg import (  # noqa: F401
    synthetic_ecg_500hz,
    synthetic_ecg_clean,
    synthetic_ecg_fast_hr,
    synthetic_ecg_flat,
    synthetic_ecg_inverted,
    synthetic_ecg_noisy,
)
