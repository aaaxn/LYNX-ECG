"""Shared pytest fixtures.

Smoke fixtures live here; per-layer fixtures may also be defined in
`tests/<layer>/conftest.py` if scope-limited.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def ecg_datasets_root() -> Path:
    """Return the datasets root from ``$ECG_DATASETS_ROOT``, or skip.

    Skips when the variable is unset (e.g. CI without data) or when the
    resolved path is absent on disk, so data-dependent tests opt out cleanly
    rather than erroring.
    """
    root = os.environ.get("ECG_DATASETS_ROOT")
    if not root:
        pytest.skip("ECG_DATASETS_ROOT is not set")
    path = Path(root)
    if not path.exists():
        pytest.skip(f"datasets root not found on disk: {path}")
    return path
