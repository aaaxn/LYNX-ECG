"""Source-variant registry for raw ECG datasets.

Resolves ``(dataset_source, source_variant)`` to a path under
``$ECG_DATASETS_ROOT`` and validates ``sampling_rate`` consistency. The
registry is closed: any combination not listed here is a config error.

Loaders themselves live in module-specific files; this module is the
single source of truth for what variants exist and where their files are
expected on disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceVariant:
    """One row in the closed registry of accepted (source, variant) pairs."""

    dataset_source: str
    source_variant: str
    sampling_rate: int
    relative_path: str


_REGISTRY: tuple[SourceVariant, ...] = (
    SourceVariant(
        "ptbxl",
        "500hz",
        500,
        "ptbxl/physionet.org/files/ptb-xl/1.0.3/records500",
    ),
    SourceVariant(
        "codetest",
        "raw_400hz",
        400,
        "code15/codetest_raw.hdf5",
    ),
    SourceVariant(
        "code100",
        "wfdb_300hz",
        300,
        "code100",
    ),
)


def get_datasets_root() -> Path:
    """Return ``$ECG_DATASETS_ROOT``.

    The variable is required: there is no machine-local default. A hardcoded
    absolute path would silently resolve to the wrong (or a non-existent)
    location on any other machine or in CI, so its absence is a loud error.
    """
    root = os.environ.get("ECG_DATASETS_ROOT")
    if not root:
        raise RuntimeError(
            "ECG_DATASETS_ROOT is not set; point it at the directory holding the "
            "raw datasets (ptbxl/, code15/, code100/)."
        )
    return Path(root)


def list_variants(dataset_source: str | None = None) -> tuple[SourceVariant, ...]:
    """List all variants, optionally filtered by dataset_source."""
    if dataset_source is None:
        return _REGISTRY
    return tuple(sv for sv in _REGISTRY if sv.dataset_source == dataset_source)


def get_source_variant(dataset_source: str, source_variant: str) -> SourceVariant:
    """Look up a variant. Raises ValueError on unknown combinations."""
    for sv in _REGISTRY:
        if sv.dataset_source == dataset_source and sv.source_variant == source_variant:
            return sv
    valid = [sv.source_variant for sv in list_variants(dataset_source)]
    if not valid:
        raise ValueError(
            f"Unknown dataset_source={dataset_source!r}; "
            f"valid sources: {sorted({sv.dataset_source for sv in _REGISTRY})}"
        )
    raise ValueError(
        f"Unknown source_variant={source_variant!r} for dataset_source={dataset_source!r}; "
        f"valid variants: {valid}"
    )


def validate_source_variant(dataset_source: str, source_variant: str, sampling_rate: int) -> None:
    """Raise ValueError if the registry's sampling_rate does not match."""
    sv = get_source_variant(dataset_source, source_variant)
    if sv.sampling_rate != sampling_rate:
        raise ValueError(
            f"source_variant={source_variant!r} implies sampling_rate={sv.sampling_rate}, "
            f"got sampling_rate={sampling_rate}"
        )


def resolve_source_path(dataset_source: str, source_variant: str) -> Path:
    """Return the absolute path of the variant's primary signal file/directory."""
    sv = get_source_variant(dataset_source, source_variant)
    return get_datasets_root() / sv.relative_path


def resolve_metadata_dir(dataset_source: str, source_variant: str) -> Path:
    """Return the directory holding metadata sidecar files for this variant.

    For PTB-XL, that is the ``1.0.3/`` release directory containing
    ``ptbxl_database.csv`` and ``scp_statements.csv``. For CODE-test the
    metadata is embedded in the HDF5, so we return the file's parent.
    """
    sv = get_source_variant(dataset_source, source_variant)
    if sv.dataset_source == "ptbxl":
        return get_datasets_root() / "ptbxl/physionet.org/files/ptb-xl/1.0.3"
    if sv.dataset_source == "code100":
        return get_datasets_root() / sv.relative_path
    return (get_datasets_root() / sv.relative_path).parent
