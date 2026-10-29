"""Pure validation helpers for label definitions and binary label arrays.

The class-name order and taxonomy are semantic parts of the dataset contract.
Raw values are checked without a lossy dtype conversion; only valid binary
vectors and matrices are returned as ``np.uint8``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np

_NUMERIC_OBJECT_TYPES = (bool, np.bool_, int, np.integer, float, np.floating)


class LabelContractError(ValueError):
    """A label definition, shape, or raw value violates the shared contract."""


def validate_label_names(
    label_names: Sequence[object],
    *,
    context: str,
) -> tuple[str, ...]:
    """Validate class names and return their exact order as an immutable tuple."""
    if isinstance(label_names, (str, bytes)):
        raise LabelContractError(f"{context}: label names must be a sequence of strings")
    try:
        names = tuple(label_names)
    except TypeError as exc:
        raise LabelContractError(f"{context}: label names must be a sequence of strings") from exc
    if not names:
        raise LabelContractError(f"{context}: label names must contain at least one class")
    if not all(isinstance(name, str) for name in names):
        raise LabelContractError(f"{context}: label names must all be strings")
    string_names = cast("tuple[str, ...]", names)
    if any(name == "" for name in string_names):
        raise LabelContractError(f"{context}: label names must be non-empty strings")
    if len(set(string_names)) != len(string_names):
        raise LabelContractError(f"{context}: label names must be unique in their stored order")
    return string_names


def validate_label_taxonomy(label_taxonomy: object, *, context: str) -> str:
    """Validate and return an exact, non-empty taxonomy string."""
    if not isinstance(label_taxonomy, str) or label_taxonomy == "":
        raise LabelContractError(f"{context}: label taxonomy must be a non-empty string")
    return label_taxonomy


def validate_label_definition(
    label_names: Sequence[object],
    label_taxonomy: object,
    *,
    context: str,
    expected_names: Sequence[object] | None = None,
    expected_taxonomy: object | None = None,
) -> tuple[tuple[str, ...], str]:
    """Validate label metadata and optional exact semantic expectations."""
    names = validate_label_names(label_names, context=context)
    taxonomy = validate_label_taxonomy(label_taxonomy, context=context)

    if expected_names is not None:
        expected = validate_label_names(expected_names, context=f"{context} expected contract")
        if names != expected:
            raise LabelContractError(
                f"{context}: label names do not match the expected order and spelling"
            )
    if expected_taxonomy is not None:
        expected_tax = validate_label_taxonomy(
            expected_taxonomy,
            context=f"{context} expected contract",
        )
        if taxonomy != expected_tax:
            raise LabelContractError(f"{context}: label taxonomy does not match expected taxonomy")
    return names, taxonomy


def _validate_label_array(
    values: object,
    *,
    expected_shape: tuple[int, ...],
    context: str,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise LabelContractError(f"{context}: labels could not be inspected as an array") from exc

    if raw.shape != expected_shape:
        raise LabelContractError(
            f"{context}: label shape {raw.shape} does not match expected {expected_shape}"
        )
    if raw.dtype.kind == "O":
        for value in raw.flat:
            if not isinstance(value, _NUMERIC_OBJECT_TYPES):
                raise LabelContractError(
                    f"{context}: labels must contain boolean, integer, or floating numeric values"
                )
            try:
                finite = bool(np.isfinite(value))
            except TypeError as exc:
                raise LabelContractError(
                    f"{context}: labels must contain boolean, integer, or floating numeric values"
                ) from exc
            if not finite:
                raise LabelContractError(f"{context}: labels must contain only finite values")
            if value != 0 and value != 1:
                raise LabelContractError(
                    f"{context}: labels must contain only binary 0 or 1 values"
                )
    else:
        if raw.dtype.kind not in "buif":
            raise LabelContractError(
                f"{context}: labels must contain boolean, integer, or floating numeric values"
            )
        if not np.isfinite(raw).all():
            raise LabelContractError(f"{context}: labels must contain only finite values")
        if not ((raw == 0) | (raw == 1)).all():
            raise LabelContractError(f"{context}: labels must contain only binary 0 or 1 values")
    return raw.astype(np.uint8, copy=False)


def validate_label_vector(
    values: object,
    *,
    num_classes: int,
    context: str,
) -> np.ndarray:
    """Validate one exact ``(num_classes,)`` vector, then return ``uint8``."""
    return _validate_label_array(
        values,
        expected_shape=(num_classes,),
        context=context,
    )


def validate_label_matrix(
    values: object,
    *,
    num_exams: int,
    num_classes: int,
    context: str,
) -> np.ndarray:
    """Validate one exact ``(num_exams, num_classes)`` matrix, then return ``uint8``."""
    return _validate_label_array(
        values,
        expected_shape=(num_exams, num_classes),
        context=context,
    )
