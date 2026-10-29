"""Unit tests for cli.build._exam_sort_key (numeric-aware exam ordering)."""

from __future__ import annotations

import pytest

from cli.build import _exam_sort_key

pytestmark = pytest.mark.unit


def test_numeric_ids_sort_numerically_not_lexicographically():
    ids = ["1", "10", "100", "2", "21837", "9999"]
    assert sorted(ids, key=_exam_sort_key) == ["1", "2", "10", "100", "9999", "21837"]


def test_non_numeric_ids_sort_after_numeric():
    ids = ["S10", "2", "10", "S2"]
    out = sorted(ids, key=_exam_sort_key)
    assert out == ["2", "10", "S10", "S2"]
