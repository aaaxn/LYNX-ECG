"""Shared machine values for explainability report contracts.

These enums bind report validators and constructors to one serialized value.
They are string subclasses, so JSON payloads hold plain strings.
"""

from __future__ import annotations

from enum import StrEnum


class ReportProducer(StrEnum):
    """Stable producer identities for strict explainability artifacts."""

    REGION_DELINEATION_QC = "region_delineation_qc"
    PROMINENCE_SUPPORT_FACTORIAL = "prominence_support_factorial"
    PROMINENCE_AVAILABILITY_SHAPLEY = "prominence_availability_shapley"


class SharedReportValue(StrEnum):
    """Values that multiple report models use with identical semantics."""

    EXACT_EIGHT_STATE_DELETION = "exact_eight_state_pr_qrs_t_deletion"
    SCORE_DROP = "score_before_deletion_minus_score_after_deletion"
