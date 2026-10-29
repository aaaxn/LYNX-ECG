"""Tests for the PTB-XL CODE-6 label mapping.

Covers the pure mapping functions: presence semantics, the AFIB-only /
1AVB-only / complete-BBB-only rules, and the canonical class order.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset.loaders import CODE_6CLASS_LABEL_NAMES
from dataset.loaders.ptbxl import code6_labels_from_scp, code6_scp_mapping

_IDX = {name: i for i, name in enumerate(CODE_6CLASS_LABEL_NAMES)}


def _label(scp: dict) -> np.ndarray:
    return code6_labels_from_scp(scp, code6_scp_mapping())


@pytest.mark.unit
def test_strict_mapping_table():
    m = code6_scp_mapping()
    assert m == {
        "1dAVb": ("1AVB",),
        "RBBB": ("CRBBB",),
        "LBBB": ("CLBBB",),
        "SB": ("SBRAD",),
        "AF": ("AFIB",),
        "ST": ("STACH",),
    }


@pytest.mark.unit
def test_presence_ignores_zero_likelihood():
    out = _label({"AFIB": 0.0, "SBRAD": 0.0, "STACH": 0.0})
    assert out[_IDX["AF"]] == 1
    assert out[_IDX["SB"]] == 1
    assert out[_IDX["ST"]] == 1


@pytest.mark.unit
def test_af_is_afib_only_not_aflt():
    assert _label({"AFIB": 100.0})[_IDX["AF"]] == 1
    assert _label({"AFLT": 100.0})[_IDX["AF"]] == 0


@pytest.mark.unit
def test_1davb_is_first_degree_only():
    assert _label({"1AVB": 0.0})[_IDX["1dAVb"]] == 1
    assert _label({"2AVB": 100.0})[_IDX["1dAVb"]] == 0
    assert _label({"3AVB": 100.0})[_IDX["1dAVb"]] == 0


@pytest.mark.unit
def test_incomplete_bbb_codes_do_not_count():
    assert _label({"IRBBB": 0.0})[_IDX["RBBB"]] == 0
    assert _label({"ILBBB": 0.0})[_IDX["LBBB"]] == 0
    assert _label({"CRBBB": 0.0})[_IDX["RBBB"]] == 1
    assert _label({"CLBBB": 0.0})[_IDX["LBBB"]] == 1


@pytest.mark.unit
def test_output_shape_dtype_and_order():
    out = _label({"CRBBB": 0.0, "STACH": 0.0})
    assert out.shape == (len(CODE_6CLASS_LABEL_NAMES),)
    assert out.dtype == np.uint8
    expected = np.zeros(6, dtype=np.uint8)
    expected[_IDX["RBBB"]] = 1
    expected[_IDX["ST"]] = 1
    np.testing.assert_array_equal(out, expected)


@pytest.mark.unit
def test_no_mapped_code_is_all_zero_undiagnosed():
    out = _label({"NORM": 100.0, "SR": 0.0, "LVOLT": 0.0})
    assert out.sum() == 0
