"""Dependency-light names and orders shared by ECG explanation workflows.

This module defines the small domain taxonomy used across regional analyses and
visualizations.  It intentionally depends only on the standard library so core
region types and display names do not pull in Torch, Pydantic, or plotting code.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import combinations
from typing import Literal


class ECGRegion(StrEnum):
    """The three players in a PR-QRS-T regional intervention."""

    PR = "pr"
    QRS = "qrs"
    T = "t"


EXPECTED_REGION: dict[str, ECGRegion] = {
    "1dAVb": ECGRegion.PR,
    "RBBB": ECGRegion.QRS,
    "LBBB": ECGRegion.QRS,
}

MORPHOLOGY_CLASSES: tuple[str, ...] = tuple(EXPECTED_REGION)

REGION_ORDER: tuple[str, ...] = tuple(region.value for region in ECGRegion)
REGION_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(REGION_ORDER, 2))
SINGLE_PAIR_STATE_ORDER: tuple[str, ...] = (
    "intact",
    *REGION_ORDER,
    *(f"{first}+{second}" for first, second in REGION_PAIRS),
)
REGION_PAIR_STATE_ORDER: tuple[str, ...] = tuple(
    state for state in SINGLE_PAIR_STATE_ORDER if "+" in state
)

TWELVE_LEAD_DISPLAY_NAMES: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)

OutputScale = Literal["probability", "logit"]
DurationName = Literal[ECGRegion.PR, ECGRegion.QRS, ECGRegion.T]
Polarity = Literal[-1, 1]
