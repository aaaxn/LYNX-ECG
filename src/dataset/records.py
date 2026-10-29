"""Dataclasses bridging the build pipeline (workers) to the HDF5 writer.

These records are the canonical handoff format between
:mod:`dataset.builder` (where workers iterate exams, run R-peaks / beats /
reduction / VG) and :mod:`dataset.hdf5_writer`.

- :class:`ExamMetadata`. One row in the ``exam_*`` table.
- :class:`GraphRecord`. One ``(exam, lead, beat)`` graph inside the worker.
- :class:`PackedGraphBatch`. An exam's graphs packed into a few large arrays.
- :class:`LeadSummary`. Per-lead stats/skips inside an ExamResult.
- :class:`ExamResult`. What a worker returns to the main process.

The worker→main IPC contract is *packed*: per-beat :class:`GraphRecord`
objects never cross the process boundary. An exam's ~dozens of graphs would
otherwise pickle as hundreds of small numpy arrays, and unpickling them in
the single main process is the build's wall-clock cap at high job counts.
:func:`pack_graphs` concatenates them (in write order) into one
:class:`PackedGraphBatch`, the writer's native layout, inside the worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dataset.qc import SkipEvent
from dataset.schema import EDGE_INDEX_DTYPE, GRAPH_SCALAR_SPECS, MAX_NODES_PER_GRAPH


@dataclass(frozen=True)
class ExamMetadata:
    """Per-exam metadata that lands in the ``exam_*`` HDF5 table."""

    exam_id: str
    patient_id: str
    strat_fold: int
    labels: np.ndarray
    label_names: tuple[str, ...]
    label_taxonomy: str


@dataclass(frozen=True)
class GraphRecord:
    """One built graph: a (exam, lead, beat) triple ready for HDF5 write.

    Conventions:
    - ``edge_index`` stores **local** indices into this graph's nodes
      (``0..reduced_num_nodes-1``); the writer applies pointer offsets
      lazily at read time.
    - ``node_x`` shape ``[N, F]`` with column 0 == amplitude.
    - ``node_t`` shape ``[N]`` in seconds, beat-relative; ``node_t[0] == 0``.
    - Edge attributes are not carried: the dataset contract derives them at load time
      from ``node_t`` / ``node_x`` (see :mod:`dataset.edge_attrs`).
    """

    exam_id: str
    lead: int
    beat_index: int
    boundary_type: int
    start_sample: int
    end_sample: int
    r_peak_sample: int
    rpeak_polarity: int
    original_num_nodes: int

    node_x: np.ndarray
    node_t: np.ndarray
    node_original_sample: np.ndarray
    edge_index: np.ndarray

    @property
    def reduced_num_nodes(self) -> int:
        return int(self.node_x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass(frozen=True)
class LeadResult:
    """Per-lead pipeline output inside the worker (carries the graphs)."""

    lead: int
    valid: bool
    graphs: list[GraphRecord] = field(default_factory=list)
    skips: list[SkipEvent] = field(default_factory=list)
    rpeak_info: dict | None = None


@dataclass(frozen=True)
class PackedGraphBatch:
    """An ordered batch of graphs packed into a few large, typed arrays.

    This is the writer's native layout: per-node / per-edge arrays are
    concatenated in graph order with per-graph counts alongside (CSR-style),
    and per-graph scalar columns live in ``scalars`` keyed by their schema
    dataset name (:data:`dataset.schema.GRAPH_SCALAR_SPECS`). ``edge_index``
    is already validated and narrowed to the storage dtype (int16).
    """

    n_graphs: int
    node_x: np.ndarray
    node_t: np.ndarray
    node_original_sample: np.ndarray
    edge_index: np.ndarray
    node_counts: np.ndarray
    edge_counts: np.ndarray
    scalars: dict[str, np.ndarray]


def pack_graphs(graphs: list[GraphRecord]) -> PackedGraphBatch:
    """Pack ``graphs`` (in the given order) into a :class:`PackedGraphBatch`.

    Validates the schema's storage invariants while the per-graph context is
    still available: every graph has a bounded node count, exact ``[2, E]``
    edge shape, graph-local endpoints, and strictly increasing edge endpoints.
    """
    if not graphs:
        raise ValueError("pack_graphs requires at least one graph")

    edge_min = np.iinfo(EDGE_INDEX_DTYPE).min
    edge_max = np.iinfo(EDGE_INDEX_DTYPE).max
    edge_chunks: list[np.ndarray] = []
    edge_counts: list[int] = []
    for g in graphs:
        context = f"(exam_id={g.exam_id!r}, lead={g.lead}, beat={g.beat_index})"
        chunk = np.asarray(g.edge_index)
        if chunk.ndim != 2 or chunk.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E] {context}; got {chunk.shape}")

        num_nodes = g.reduced_num_nodes
        if num_nodes > MAX_NODES_PER_GRAPH:
            raise ValueError(
                f"Graph {context} has {num_nodes} nodes; edge_index storage uses int16, "
                f"which supports at most {MAX_NODES_PER_GRAPH} nodes per graph."
            )

        edge_count = int(chunk.shape[1])
        edge_counts.append(edge_count)
        if edge_count > 0:
            if not np.issubdtype(chunk.dtype, np.integer):
                raise ValueError(
                    f"edge_index endpoints must use an integer dtype {context}; got {chunk.dtype}."
                )
            chunk_min = int(chunk.min())
            chunk_max = int(chunk.max())
            if chunk_min < edge_min or chunk_max > edge_max:
                raise ValueError(
                    f"edge_index values outside int16 range "
                    f"[{chunk_min}, {chunk_max}] {context}; "
                    f"local node indices must be in [0, {edge_max}]."
                )
            if chunk_min < 0 or chunk_max >= num_nodes:
                raise ValueError(
                    f"edge_index values outside graph-local node range [0, {num_nodes}) "
                    f"{context}; values span [{chunk_min}, {chunk_max}]."
                )
            invalid_order = np.flatnonzero(chunk[0] >= chunk[1])
            if invalid_order.size:
                column = int(invalid_order[0])
                raise ValueError(
                    f"edge_index violates src < dst ordering {context} at column "
                    f"{column}: {int(chunk[0, column])}->{int(chunk[1, column])}."
                )
            edge_chunks.append(chunk.astype(EDGE_INDEX_DTYPE, copy=False))

    G = len(graphs)  # noqa: N806
    scalars: dict[str, np.ndarray] = {}
    for spec in GRAPH_SCALAR_SPECS:
        field = spec.graph_field
        assert field is not None
        scalars[spec.name] = np.fromiter(
            (getattr(g, field) for g in graphs), dtype=spec.dtype, count=G
        )
    return PackedGraphBatch(
        n_graphs=G,
        node_x=np.concatenate([np.asarray(g.node_x, dtype=np.float32) for g in graphs], axis=0),
        node_t=np.concatenate([np.asarray(g.node_t, dtype=np.float32).reshape(-1) for g in graphs]),
        node_original_sample=np.concatenate(
            [np.asarray(g.node_original_sample, dtype=np.int64).reshape(-1) for g in graphs]
        ),
        edge_index=(
            np.concatenate(edge_chunks, axis=1)
            if edge_chunks
            else np.empty((2, 0), dtype=EDGE_INDEX_DTYPE)
        ),
        node_counts=np.fromiter((g.reduced_num_nodes for g in graphs), dtype=np.int64, count=G),
        edge_counts=np.asarray(edge_counts, dtype=np.int64),
        scalars=scalars,
    )


@dataclass(frozen=True)
class LeadSummary:
    """Per-lead stats and skips returned to the main process (no graph data)."""

    lead: int
    valid: bool
    n_graphs: int
    skips: list[SkipEvent] = field(default_factory=list)
    rpeak_info: dict | None = None


@dataclass(frozen=True)
class ExamResult:
    """One exam's worker output (success or recoverable failure).

    ``packed`` carries all of the exam's graphs (valid leads, in
    ``graph_leads`` order, beats in order within each lead); it is ``None``
    when the exam produced no graphs.
    """

    exam_id: str
    success: bool
    metadata: ExamMetadata | None = None
    leads: list[LeadSummary] = field(default_factory=list)
    packed: PackedGraphBatch | None = None
    skips: list[SkipEvent] = field(default_factory=list)
    error: str | None = None

    def all_skips(self) -> list[SkipEvent]:
        events = list(self.skips)
        for ld in self.leads:
            events.extend(ld.skips)
        return events
