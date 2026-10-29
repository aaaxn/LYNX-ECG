"""Bridge from current-contract HDF5 metadata to the beat-local timing the
clinical-alignment metric needs, aligned to the model's node ordering.

The model feature ``x[:, 1]`` is ``t_norm_beat`` (dimensionless); the
clinical-alignment metric needs ``node_t`` in beat-local *seconds* plus each
beat's R-peak time. Those live in the HDF5 metadata (``node_t``,
``graph_r_peak_sample``, ``graph_start_sample``), which the training dataset
reads but does not surface. This module reads them with the **same**
exam -> graph -> node slicing :class:`training.dataset.ExamGraphDataset` uses, so
the returned ``node_t`` aligns row-for-row with the model's ``x`` for that exam
and ``node_beat`` equals the model batch's ``subgraph_node_batch``. The
alignment is locked by ``tests/integration/test_explain_ecg_timing.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch


@dataclass
class ExamTiming:
    """Beat-local timing for one exam, aligned to the model's node ordering.

    - ``node_t`` ``[N]``: each node's time within its beat, in seconds from beat
      start, aligned row-for-row with the model's ``x`` for this exam.
    - ``node_beat`` ``[N]``: each node's beat (subgraph) id, 0-based within the
      exam (equals the batch ``subgraph_node_batch`` for a single-exam batch).
    - ``beat_r_peak_t`` ``[S]``: each beat's R-peak/QRS-anchor time, in seconds
      from beat start, indexable by ``node_beat``.
    - ``beat_rpeak_polarity`` ``[S]``: each beat's ``graph_rpeak_polarity``
      (``+1``/``-1``), indexable by ``node_beat``. ``+1`` means the anchor in
      ``beat_r_peak_t`` was detected on the raw signal; ``-1`` means it was
      detected on the inverted signal, so it is not guaranteed to be the R wave
      specifically. Exposed so callers can test whether downstream results
      depend on which polarity was selected.
    """

    node_t: torch.Tensor
    node_beat: torch.Tensor
    beat_r_peak_t: torch.Tensor
    beat_rpeak_polarity: torch.Tensor


def beat_local_timing(
    node_ptr: np.ndarray,
    r_peak_sample: np.ndarray,
    start_sample: np.ndarray,
    fs: float,
    rpeak_polarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure core shared by :func:`exam_timing` and callers that pre-slice their
    own graph-level arrays, for example to read the full ``graph_*`` arrays once
    and slice them per exam in memory instead of one small HDF5 read per exam).

    Parameters
    ----------
    node_ptr:
        ``[num_subgraphs + 1]`` CSR-style node pointer for this exam's beats.
        Only consecutive differences are used, so this may be either the
        absolute HDF5 offsets or a 0-based local pointer. Both give the same
        ``node_beat``.
    r_peak_sample, start_sample:
        ``[num_subgraphs]`` this exam's ``graph_r_peak_sample`` /
        ``graph_start_sample`` slices.
    fs:
        Sampling rate (Hz).
    rpeak_polarity:
        ``[num_subgraphs]`` this exam's ``graph_rpeak_polarity`` slice
        (``+1``/``-1``; see ``ExamTiming.beat_rpeak_polarity``).

    Returns
    -------
    node_beat : ndarray[int64] ``[N]``
        Beat id (0-based within exam) of each node, aligned to a node array
        sliced at ``[node_ptr[0] : node_ptr[-1]]``.
    beat_r_peak_t : ndarray[float64] ``[num_subgraphs]``
        Each beat's R-peak/QRS-anchor time, in seconds from its own start.
    beat_rpeak_polarity : ndarray[int8] ``[num_subgraphs]``
        Passthrough of ``rpeak_polarity`` (no computation needed, already
        one value per subgraph), returned alongside the other two so it stays
        indexable by the same ``node_beat``/beat-id convention.
    """
    num_subgraphs = r_peak_sample.shape[0]
    node_counts = np.diff(node_ptr)
    node_beat = np.repeat(np.arange(num_subgraphs, dtype=np.int64), node_counts)
    beat_r_peak_t = (r_peak_sample.astype(np.int64) - start_sample.astype(np.int64)) / fs
    beat_rpeak_polarity = rpeak_polarity.astype(np.int8)
    return node_beat, beat_r_peak_t, beat_rpeak_polarity


def exam_timing(
    h5: h5py.File,
    exam_idx: int,
    *,
    graph_exam_index: np.ndarray | None = None,
) -> ExamTiming:
    """Read beat-local timing for ``exam_idx`` from an open graph HDF5 file.

    Pass ``graph_exam_index`` (the full ``h5['graph_exam_index'][:]`` array) to
    avoid re-reading it on every call when iterating many exams.
    """
    fs = float(h5.attrs["sampling_rate"])
    if graph_exam_index is None:
        graph_exam_index = h5["graph_exam_index"][:]

    g_start = int(np.searchsorted(graph_exam_index, exam_idx, side="left"))
    g_end = int(np.searchsorted(graph_exam_index, exam_idx, side="right"))
    if g_end == g_start:
        raise ValueError(f"exam_idx {exam_idx} has no subgraphs in this file")

    node_ptr = h5["graph_node_ptr"][g_start : g_end + 1].astype(np.int64)
    n_start, n_end = int(node_ptr[0]), int(node_ptr[-1])
    node_t = h5["node_t"][n_start:n_end].astype(np.float64)

    r_peak = h5["graph_r_peak_sample"][g_start:g_end].astype(np.int64)
    start = h5["graph_start_sample"][g_start:g_end].astype(np.int64)
    polarity = h5["graph_rpeak_polarity"][g_start:g_end]
    node_beat, beat_r_peak_t, beat_rpeak_polarity = beat_local_timing(
        node_ptr, r_peak, start, fs, polarity
    )

    return ExamTiming(
        node_t=torch.from_numpy(node_t).float(),
        node_beat=torch.from_numpy(node_beat).long(),
        beat_r_peak_t=torch.from_numpy(beat_r_peak_t).float(),
        beat_rpeak_polarity=torch.from_numpy(beat_rpeak_polarity).to(torch.int8),
    )
