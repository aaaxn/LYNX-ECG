"""Per-exam amplitude normalization for gain-invariant node features.

A pure, source-agnostic transform: given one exam's multi-lead signal, divide
every lead by a single per-exam scale derived from the configured
``graph_leads``. The result lives in a canonical amplitude scale independent of
the storage units / recording gain of the source, so a model's learned notion
of "a large QRS" transfers between sources whose raw amplitudes differ by a
constant factor.

Applying one scalar to the whole exam preserves inter-lead amplitude ratios,
and a positive uniform scale leaves the visibility-graph topology unchanged.
Only the stored amplitude feature value differs.

This runs as a source transform (after band-pass + resample, before R-peak
detection and graph construction); see :func:`dataset.loaders.prepare_exam_sample`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

P95_QUANTILE = 0.95


class DegenerateAmplitudeError(ValueError):
    """An exam's amplitude scale is zero or non-finite (e.g. a flat record).

    Raised instead of dividing by zero. The builder catches this and records a
    counted ``amplitude_norm_zero_scale`` exam-level skip.
    """


def p95_lead_median_scale(signal: np.ndarray, graph_leads: Sequence[int]) -> float | None:
    """Return the ``p95_lead_median`` per-exam amplitude scale, or ``None``.

    The scale is the median, across the in-bounds ``graph_leads``, of each
    lead's 95th-percentile absolute amplitude. Out-of-bounds lead indices are
    skipped (they are reported per-lead by the builder). Returns ``None`` when
    no ``graph_leads`` index is in bounds, so the caller leaves the signal
    untouched and the existing out-of-bounds reporting fires downstream.

    Raises:
        DegenerateAmplitudeError: the computed scale is zero or non-finite.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be [n_leads, n_samples], got shape {signal.shape}")
    n_leads = signal.shape[0]
    leads = [int(lead) for lead in graph_leads if 0 <= int(lead) < n_leads]
    if not leads:
        return None
    per_lead_p95 = np.array(
        [np.quantile(np.abs(signal[lead].astype(np.float64)), P95_QUANTILE) for lead in leads]
    )
    scale = float(np.median(per_lead_p95))
    if not np.isfinite(scale) or scale <= 0.0:
        raise DegenerateAmplitudeError(
            f"per-exam amplitude scale is degenerate ({scale!r}); "
            f"signal is flat or non-finite over graph_leads={leads}"
        )
    return scale


def normalize_amplitude(
    signal: np.ndarray, graph_leads: Sequence[int], *, method: str
) -> np.ndarray:
    """Return ``signal`` divided by its per-exam scale (float32).

    A no-op (returns the input scale-unchanged, cast to float32) when no
    ``graph_leads`` index is in bounds.

    Raises:
        DegenerateAmplitudeError: the exam's scale is zero or non-finite.
        ValueError: ``method`` is unknown.
    """
    if method == "p95_lead_median":
        scale = p95_lead_median_scale(signal, graph_leads)
    else:
        raise ValueError(f"unknown amplitude normalization method: {method!r}")
    if scale is None:
        return signal.astype(np.float32, copy=False)
    return (signal / scale).astype(np.float32, copy=False)
