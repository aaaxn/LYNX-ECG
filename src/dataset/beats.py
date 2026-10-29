"""Per-lead beat segmentation by midpoint between adjacent R-peaks.

Conventions:

- ``BeatSpan(start_sample, end_sample, ...)`` describes a closed interval
  ``[start_sample, end_sample]`` (both indices included). ``y[start:end+1]``
  yields the beat samples.
- The first beat starts at sample 0; the last beat ends at the final sample
  of the signal. Both are tagged with non-zero ``boundary_type``.
- ``r_peak_local_idx`` is the position of the R-peak inside the beat
  (``r_peak_sample - start_sample``); ``reduce_turning_points`` always keeps
  it as a node.
- Per-beat QC (``validate_beat_signal``, ``validate_beat_anchor``) is applied
  separately by the caller; this module does only the geometric segmentation.

``boundary_type`` codes (int8):

- ``0``: middle beat
- ``1``: first beat (boundary at signal start)
- ``2``: last beat (boundary at signal end)
- ``3``: only beat (single R-peak)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BoundaryType:
    """Numeric codes used by ``BeatSpan.boundary_type`` and ``graph_boundary_type`` in HDF5."""

    MIDDLE = 0
    FIRST = 1
    LAST = 2
    ONLY_BEAT = 3


@dataclass(frozen=True)
class BeatSpan:
    """One beat candidate: closed-interval span anchored at an R-peak."""

    start_sample: int
    end_sample: int
    r_peak_sample: int
    r_peak_local_idx: int
    boundary_type: int

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample + 1


def segment_beats(signal_length: int, peaks: np.ndarray) -> list[BeatSpan]:
    """Build ``BeatSpan`` objects from R-peak indices using the midpoint rule.

    For ``n_peaks >= 2``:

    - ``start_k = (peaks[k-1] + peaks[k]) // 2`` for ``k > 0``; ``start_0 = 0``.
    - ``end_k   = (peaks[k] + peaks[k+1]) // 2 - 1`` for ``k < n-1``;
      ``end_{n-1} = signal_length - 1``.

    No QC is applied here; degenerate spans (e.g. very close R-peaks) come
    out as-is and are caught by ``validate_beat_signal`` / ``validate_beat_anchor``.
    """
    if signal_length <= 0:
        return []
    p = np.asarray(peaks, dtype=np.int64).reshape(-1)
    n = int(p.size)
    if n == 0:
        return []
    if n == 1:
        peak = int(p[0])
        return [
            BeatSpan(
                start_sample=0,
                end_sample=signal_length - 1,
                r_peak_sample=peak,
                r_peak_local_idx=peak,
                boundary_type=BoundaryType.ONLY_BEAT,
            )
        ]

    spans: list[BeatSpan] = []
    for k in range(n):
        peak = int(p[k])
        start = 0 if k == 0 else int((int(p[k - 1]) + peak) // 2)
        end = signal_length - 1 if k == n - 1 else int((peak + int(p[k + 1])) // 2 - 1)
        if k == 0:
            bt = BoundaryType.FIRST
        elif k == n - 1:
            bt = BoundaryType.LAST
        else:
            bt = BoundaryType.MIDDLE
        spans.append(
            BeatSpan(
                start_sample=start,
                end_sample=end,
                r_peak_sample=peak,
                r_peak_local_idx=peak - start,
                boundary_type=bt,
            )
        )
    return spans
