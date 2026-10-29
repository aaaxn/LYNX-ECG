"""Deterministic dataset filename derivation.

Format:

    <source>__<method>__<reducer>__<variant>[__code6strict][__bp<lo>-<hi>hz-o<order>]
    [__resample-<hz>hz][__norm-p95med]__natural__leads-X-Y-Z[__limit-N].h5

Tokens are joined by ``__`` (double underscore) so they can be parsed back
unambiguously from filenames. Lead lists are flattened to a single token
``leads-<i>-<j>-<k>`` with sorted unique leads. ``limit_exams`` appears only
when the config sets a non-null limit, marking smoke / debug builds as
distinct from full builds.

``code6strict`` (PTB-XL label mapping) and ``natural`` (graph type) are
constant tokens: each names the only choice the code supports, and the paper's
dataset file names contain them.
"""

from __future__ import annotations

from pathlib import Path

from config.models import DatasetConfig
from dataset.reduction import build_reducer


def _reducer_token(config: DatasetConfig) -> str:
    return build_reducer(config.node_reduction).filename_token


def _hz(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _signal_filter_token(config: DatasetConfig) -> str | None:
    sf = config.signal_filter
    if sf is None:
        return None
    return f"bp{_hz(sf.highpass_hz)}-{_hz(sf.lowpass_hz)}hz-o{sf.order}"


def _normalization_token(config: DatasetConfig) -> str | None:
    norm = config.amplitude_normalization
    if norm is None:
        return None
    method_tokens = {"p95_lead_median": "p95med"}
    return f"norm-{method_tokens[norm.method]}"


def _leads_token(config: DatasetConfig) -> str:
    leads = sorted({int(lead) for lead in config.graph_leads})
    body = "-".join(str(lead) for lead in leads)
    return f"leads-{body}"


def _limit_token(config: DatasetConfig) -> str | None:
    if config.limit_exams is None:
        return None
    return f"limit-{int(config.limit_exams)}"


def derive_dataset_filename(config: DatasetConfig) -> str:
    """Return ``"<base>.h5"``. Same config → same filename, every time."""
    tokens: list[str] = [
        config.dataset_source,
        config.node_reduction.method,
        _reducer_token(config),
        config.source_variant,
    ]
    if config.dataset_source == "ptbxl":
        tokens.append("code6strict")
    sf_token = _signal_filter_token(config)
    if sf_token is not None:
        tokens.append(sf_token)
    if config.resample_to_hz is not None:
        tokens.append(f"resample-{config.resample_to_hz:g}hz")
    norm_token = _normalization_token(config)
    if norm_token is not None:
        tokens.append(norm_token)
    tokens += [
        "natural",
        _leads_token(config),
    ]
    limit = _limit_token(config)
    if limit is not None:
        tokens.append(limit)
    return "__".join(tokens) + ".h5"


def derive_dataset_filename_base(config: DatasetConfig) -> str:
    """Filename without the ``.h5`` extension, used for report directory names."""
    name = derive_dataset_filename(config)
    return name[:-3] if name.endswith(".h5") else name


def derive_output_path(config: DatasetConfig, *, repo_root: Path) -> Path:
    """Return the repo-relative path for the build's HDF5 output.

    Layout: ``graph_datasets/<source>/<filename>``.
    Honors an explicit ``config.output_path`` override.
    """
    if config.output_path is not None:
        return Path(config.output_path)
    return (
        Path(repo_root) / "graph_datasets" / config.dataset_source / derive_dataset_filename(config)
    )


def derive_reports_dir(config: DatasetConfig, *, repo_root: Path) -> Path:
    """Return ``<repo>/<reports_dir>/<filename_base>``."""
    return Path(repo_root) / config.reporting.reports_dir / derive_dataset_filename_base(config)
