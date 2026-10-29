from __future__ import annotations

import pytest

from explain.ecg_taxonomy import (
    EXPECTED_REGION,
    ECGRegion,
)

pytestmark = pytest.mark.unit


def test_expected_region_pins_the_clinical_mapping() -> None:
    assert EXPECTED_REGION == {
        "1dAVb": ECGRegion.PR,
        "RBBB": ECGRegion.QRS,
        "LBBB": ECGRegion.QRS,
    }
