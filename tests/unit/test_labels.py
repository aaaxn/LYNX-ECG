"""Tests for the shared label contract."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.labels import (
    LabelContractError,
    validate_label_definition,
    validate_label_matrix,
    validate_label_vector,
)

pytestmark = pytest.mark.unit


def test_label_definition_rejects_empty_class_names() -> None:
    with pytest.raises(LabelContractError, match=r"source file.*at least one"):
        validate_label_definition((), "taxonomy", context="source file")


@pytest.mark.parametrize("names", ["A", object()])
def test_label_definition_rejects_bare_string_or_noniterable_names(names: object) -> None:
    with pytest.raises(LabelContractError, match=r"source file.*sequence of strings"):
        validate_label_definition(names, "taxonomy", context="source file")


def test_label_definition_rejects_empty_class_name() -> None:
    with pytest.raises(LabelContractError, match=r"source file.*non-empty"):
        validate_label_definition(("A", ""), "taxonomy", context="source file")


def test_label_definition_rejects_non_string_class_name() -> None:
    with pytest.raises(LabelContractError, match=r"source file.*strings"):
        validate_label_definition(("A", 2), "taxonomy", context="source file")


def test_label_definition_rejects_duplicate_class_names() -> None:
    with pytest.raises(LabelContractError, match=r"source file.*unique"):
        validate_label_definition(("A", "A"), "taxonomy", context="source file")


@pytest.mark.parametrize("taxonomy", ["", 1])
def test_label_definition_rejects_invalid_taxonomy(taxonomy: object) -> None:
    with pytest.raises(LabelContractError, match=r"source file.*taxonomy.*non-empty string"):
        validate_label_definition(("A",), taxonomy, context="source file")


def test_label_definition_preserves_exact_name_order() -> None:
    names, taxonomy = validate_label_definition(("B", "A"), "tax-v1", context="source file")

    assert names == ("B", "A")
    assert taxonomy == "tax-v1"


@pytest.mark.parametrize(
    "values",
    [
        np.array([0, 1], dtype=np.int64),
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([False, True], dtype=np.bool_),
    ],
)
def test_label_vector_accepts_numeric_binary_values_and_returns_uint8(values: np.ndarray) -> None:
    actual = validate_label_vector(values, num_classes=2, context="exam E1")

    np.testing.assert_array_equal(actual, np.array([0, 1], dtype=np.uint8))
    assert actual.dtype == np.uint8


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.0, 0.5], "binary"),
        ([0, 2], "binary"),
        ([0, -1], "binary"),
        ([0.0, np.nan], "finite"),
        ([0.0, np.inf], "finite"),
        (["0", "1"], "numeric"),
    ],
)
def test_label_vector_rejects_invalid_values(values: list[object], message: str) -> None:
    with pytest.raises(LabelContractError, match=rf"exam E1.*{message}"):
        validate_label_vector(values, num_classes=2, context="exam E1")


def test_label_vector_accepts_object_dtype_numeric_scalars_without_coercion() -> None:
    values = np.array([True, np.int64(1)], dtype=object)

    actual = validate_label_vector(values, num_classes=2, context="exam E1")

    np.testing.assert_array_equal(actual, np.array([1, 1], dtype=np.uint8))
    assert actual.dtype == np.uint8


def test_label_vector_rejects_object_dtype_strings() -> None:
    values = np.array(["0", "1"], dtype=object)

    with pytest.raises(LabelContractError, match=r"exam E1.*numeric"):
        validate_label_vector(values, num_classes=2, context="exam E1")


@pytest.mark.parametrize("values", [[[0, 1], [1]], [[0], [1, 0]]])
def test_label_matrix_rejects_ragged_values(values: list[list[int]]) -> None:
    with pytest.raises(LabelContractError, match=r"annotation file"):
        validate_label_matrix(values, num_exams=2, num_classes=2, context="annotation file")


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0], [1]], dtype=np.int64),
        np.array([0, 1, 1, 0], dtype=np.int64),
    ],
)
def test_label_vector_requires_exact_shape(values: np.ndarray) -> None:
    with pytest.raises(LabelContractError, match=r"exam E1.*shape.*\(2,\)"):
        validate_label_vector(values, num_classes=2, context="exam E1")


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((1, 2), dtype=np.int64),
        np.zeros((2, 3), dtype=np.int64),
        np.zeros(4, dtype=np.int64),
    ],
)
def test_label_matrix_requires_exact_row_and_column_counts(values: np.ndarray) -> None:
    with pytest.raises(LabelContractError, match=r"annotation file.*shape.*\(2, 2\)"):
        validate_label_matrix(
            values,
            num_exams=2,
            num_classes=2,
            context="annotation file",
        )


def test_label_matrix_returns_uint8_after_validation() -> None:
    values = np.array([[False, True], [1.0, 0.0]])

    actual = validate_label_matrix(
        values,
        num_exams=2,
        num_classes=2,
        context="annotation file",
    )

    np.testing.assert_array_equal(actual, np.array([[0, 1], [1, 0]], dtype=np.uint8))
    assert actual.dtype == np.uint8


def test_label_definition_rejects_expected_name_reorder() -> None:
    with pytest.raises(LabelContractError, match=r"exam E2.*label names.*expected order"):
        validate_label_definition(
            ("B", "A"),
            "tax-v1",
            expected_names=("A", "B"),
            expected_taxonomy="tax-v1",
            context="exam E2",
        )


def test_label_definition_rejects_expected_taxonomy_mismatch() -> None:
    with pytest.raises(LabelContractError, match=r"exam E2.*taxonomy.*expected"):
        validate_label_definition(
            ("A", "B"),
            "tax-v2",
            expected_names=("A", "B"),
            expected_taxonomy="tax-v1",
            context="exam E2",
        )
