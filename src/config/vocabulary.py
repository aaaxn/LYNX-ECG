"""Shared literal vocabularies used by config and runtime modules.

This module intentionally depends only on :mod:`typing` so model, dataset, and
training code can import these aliases without pulling in Pydantic or PyTorch.
"""

from typing import Literal

RPeakPolarityMode = Literal["auto", "positive", "negative"]

__all__ = [
    "RPeakPolarityMode",
]
