"""PyG Dataset over the current HDF5 contract with supergraph synthesis.

Each ``__getitem__(idx)`` returns one ``HierData`` per exam:

- ``x``, ``edge_index``, ``edge_attr``: all subgraphs of the exam, batched by
  PyG's CSR layout and rebased so ``edge_index`` indexes into the exam-local
  ``x`` block (not the per-subgraph local 0..K-1 the HDF5 stores).
  ``edge_attr`` is not stored; it is derived here from the
  exam's ``node_t`` / ``node_x`` slices (see :mod:`dataset.edge_attrs`).
- ``subgraph_node_batch``: ``[N]`` subgraph membership per node.
- ``super_edge_index``, ``super_edge_step``: synthesized per-(exam, lead) chain;
  signed bidirectional edges (``+1`` forward, ``-1`` reverse).
- ``subgraph_lead``: ``[S]`` lead id per subgraph in ``[0, num_leads)``.
- ``subgraph_exam``: ``[S]`` zero vector (one exam per Data; PyG Batch offsets it).
- ``y``: ``[num_classes]`` multi-hot float labels.

``HierData.__inc__`` overrides ensure PyG ``Batch.from_data_list`` offsets the
right tensors when collating multiple exams into one batch.
"""

from __future__ import annotations

import operator
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from config.train_models import SubgraphRepresentationConfig
from dataset.schema import require_dataset_contract
from training.subgraph_representation import materialize_subgraph_representation

_GRAPH_EXAM_INDEX_CHUNK_SIZE = 1_048_576


def _stream_graph_exam_boundaries(
    graph_exam_index: h5py.Dataset,
    num_exams: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate ownership and construct per-exam graph boundaries in one pass."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if graph_exam_index.ndim != 1 or np.dtype(graph_exam_index.dtype) != np.dtype(np.int64):
        raise ValueError(
            "graph_exam_index must be a 1-D int64 dataset; "
            f"got shape {graph_exam_index.shape}, dtype {graph_exam_index.dtype}"
        )

    starts = np.full(num_exams, -1, dtype=np.int64)
    ends = np.full(num_exams, -1, dtype=np.int64)
    first_value: int | None = None
    previous: int | None = None
    range_failure: tuple[int, int] | None = None
    order_failure: tuple[int, int, int] | None = None
    first_gap: int | None = None

    for lo in range(0, int(graph_exam_index.shape[0]), chunk_size):
        hi = min(lo + chunk_size, int(graph_exam_index.shape[0]))
        values = graph_exam_index[lo:hi]
        if values.size == 0:
            continue

        if first_value is None:
            first_value = int(values[0])

        invalid = (values < 0) | (values >= num_exams)
        if range_failure is None:
            invalid_positions = np.flatnonzero(invalid)
            if invalid_positions.size:
                offset = int(invalid_positions[0])
                range_failure = (lo + offset, int(values[offset]))

        cross_chunk_decrease = previous is not None and int(values[0]) < previous
        decreases = np.flatnonzero(values[1:] < values[:-1])
        if order_failure is None:
            if cross_chunk_decrease:
                order_failure = (lo, previous, int(values[0]))
            elif decreases.size:
                offset = int(decreases[0]) + 1
                order_failure = (lo + offset, int(values[offset - 1]), int(values[offset]))

        if first_gap is None and range_failure is None and order_failure is None:
            if previous is not None and int(values[0]) > previous + 1:
                first_gap = previous + 1
            else:
                jumps = np.flatnonzero(values[1:] > values[:-1] + 1)
                if jumps.size:
                    first_gap = int(values[int(jumps[0])]) + 1

        chunk_is_ordered = not cross_chunk_decrease and decreases.size == 0
        if not np.any(invalid) and chunk_is_ordered:
            exam_indices, run_starts, run_counts = np.unique(
                values, return_index=True, return_counts=True
            )
            for exam_index, run_start, run_count in zip(
                exam_indices, run_starts, run_counts, strict=True
            ):
                exam = int(exam_index)
                global_start = lo + int(run_start)
                if starts[exam] == -1:
                    starts[exam] = global_start
                ends[exam] = global_start + int(run_count)

        previous = int(values[-1])

    if range_failure is not None:
        pos, value = range_failure
        raise ValueError(f"graph_exam_index[{pos}]={value} is outside [0, {num_exams})")
    if order_failure is not None:
        pos, prior, value = order_failure
        raise ValueError(
            "graph_exam_index must be non-decreasing "
            f"(decreases at graph {pos}: {prior} -> {value})"
        )

    missing: int | None = None
    if num_exams > 0:
        if first_value is None or first_value > 0:
            missing = 0
        elif first_gap is not None:
            missing = first_gap
        elif previous is not None and previous < num_exams - 1:
            missing = previous + 1
    if missing is not None:
        raise ValueError(
            "graph_exam_index must cover every exam in "
            f"[0, {num_exams}); first missing exam index {missing}"
        )

    return starts, ends


@dataclass(frozen=True)
class GraphIndexArrays:
    """Per-HDF5 arrays shared by all ExamGraphDataset views over one file.

    Built once per unique HDF5 path and passed into every ExamGraphDataset that
    serves a subset of that file, so the ~7.4 GB of graph-index arrays (and the
    uint8 label array) are resident once per process instead of once per split.
    All fields are read-only; DataLoader workers inherit them copy-on-write via
    fork, and the dataclass is picklable for spawn-mode workers.
    """

    edge_attr_names: list[str]
    num_exams: int
    num_classes: int
    exam_labels: np.ndarray
    exam_id: np.ndarray
    exam_to_graph_start: np.ndarray
    exam_to_graph_end: np.ndarray
    graph_lead: np.ndarray
    graph_beat_index: np.ndarray
    graph_node_ptr: np.ndarray
    graph_edge_ptr: np.ndarray

    @classmethod
    def from_hdf5(cls, h5_path: str | Path) -> GraphIndexArrays:
        h5_path = str(h5_path)
        with h5py.File(h5_path, "r") as f:
            require_dataset_contract(f, Path(h5_path))
            num_exams = int(f["exam_id"].shape[0])
            exam_to_graph_start, exam_to_graph_end = _stream_graph_exam_boundaries(
                f["graph_exam_index"], num_exams, _GRAPH_EXAM_INDEX_CHUNK_SIZE
            )
            return cls(
                edge_attr_names=[str(n) for n in f.attrs["edge_attr_names"]],
                num_exams=num_exams,
                num_classes=int(f["exam_labels"].shape[1]),
                exam_labels=f["exam_labels"][:].astype(np.uint8),
                exam_id=f["exam_id"][:],
                exam_to_graph_start=exam_to_graph_start,
                exam_to_graph_end=exam_to_graph_end,
                graph_lead=f["graph_lead"][:],
                graph_beat_index=f["graph_beat_index"][:],
                graph_node_ptr=f["graph_node_ptr"][:],
                graph_edge_ptr=f["graph_edge_ptr"][:],
            )


class HierData(Data):
    """``Data`` with custom ``__inc__`` for hierarchical fields."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "super_edge_index":
            return int(self.subgraph_lead.size(0))
        if key == "subgraph_node_batch":
            return int(self.subgraph_lead.size(0))
        if key == "subgraph_exam":
            return 1
        if key == "subgraph_lead":
            return 0
        if key == "subgraph_beat":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def _build_super_edges(
    graph_lead: np.ndarray, graph_beat: np.ndarray, *, validate: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(exam, lead) sequential bidirectional super edges with signed step.

    ``graph_lead`` and ``graph_beat`` are length-S arrays giving the lead and
    beat index of each subgraph (in HDF5 order). Returns
    ``(super_edge_index [2, E], super_edge_step [E])``.

    The HDF5 build sorts by ``(exam, lead, beat)`` so subgraphs of the same
    lead are already contiguous and beat-ordered. ``validate=False`` skips the
    monotonicity re-check after the caller validates that view item.
    """
    if graph_lead.size <= 1:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)

    same_lead = graph_lead[:-1] == graph_lead[1:]
    if validate and not np.all(np.diff(graph_beat)[same_lead] > 0):
        bad = int(np.flatnonzero(same_lead & (np.diff(graph_beat) <= 0))[0])
        raise ValueError(
            f"graph_beat_index not monotonic for lead {int(graph_lead[bad])}; "
            "the schema contract requires sort by (exam, lead, beat)."
        )

    forward_src = np.flatnonzero(same_lead).astype(np.int64)
    if forward_src.size == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)

    forward_dst = forward_src + 1
    edge_index = np.empty((2, forward_src.size * 2), dtype=np.int64)
    edge_index[:, 0::2] = np.stack([forward_src, forward_dst], axis=0)
    edge_index[:, 1::2] = np.stack([forward_dst, forward_src], axis=0)
    step = np.empty(forward_src.size * 2, dtype=np.int64)
    step[0::2] = 1
    step[1::2] = -1
    return edge_index, step


class ExamGraphDataset(Dataset):
    """HDF5-backed per-exam dataset with supergraph synthesis.

    The HDF5 file is opened lazily per worker so DataLoader with
    ``num_workers > 0`` works correctly (h5py file handles are not safe to share
    across processes). Each served exam validates its stored graph edges and
    beat ordering once per process.

    The four per-graph index arrays (``graph_lead``, ``graph_beat_index``,
    ``graph_node_ptr``, ``graph_edge_ptr``) and the label array are held in a
    :class:`GraphIndexArrays` instance. When ``arrays`` is not provided, one is
    built from the HDF5 file automatically. When multiple dataset views share the
    same file (e.g. train / val splits), callers should pass the same
    ``GraphIndexArrays`` instance so the ~7.4 GB of index data is resident once
    per process rather than once per split. DataLoader workers inherit the arrays
    copy-on-write via fork, or pickle them for spawn-mode workers.

    Args:
        h5_path: path to an HDF5 satisfying the current dataset contract.
        exam_indices: indices in ``[0, num_exams)`` to expose. Derive these
            via the splits module from the fold information in
            :func:`training.hdf5_meta.read_hdf5_meta`.
        num_leads: total leads in the dataset (used by the model embedding).
            Validated here against the full ``graph_lead`` array. A mismatch
            would otherwise surface as an opaque embedding-lookup failure deep
            inside the model forward.
        arrays: pre-built :class:`GraphIndexArrays` for this HDF5 file. When
            ``None`` (default), one is constructed automatically. Pass a shared
            instance when constructing multiple views over the same file to
            avoid loading the index arrays multiple times.
        subgraph_representation: immutable load-time topology/edge-attribute
            policy. The default is the stored visibility adjacency
            with endpoint-derived attributes. Temporal chains are constructed
            independently inside each beat graph before any common
            symmetrization.
        symmetrize_edges: per-beat visibility-graph edges are stored directed
            (``src < dst``, enforced at build time by
            :func:`dataset.graph_ops.build_visibility_graph`) and nothing on
            this load path reverses them. ``GINEConv``'s default message flow
            aggregates ``src -> dst`` only, so with directed edges the earliest
            turning point of every beat receives zero neighbor messages at any
            depth, and every node's receptive field covers only earlier-in-time
            nodes. The supergraph, by contrast, is intentionally bidirectional
            (see :func:`_build_super_edges`). When ``True``, ``__getitem__``
            appends the reverse of every subgraph edge (with negated
            ``delta_t`` / ``delta_y``), making subgraph message passing
            undirected like the supergraph. Defaults to ``False`` (the stored
            directed edges).
    """

    def __init__(
        self,
        h5_path: str | Path,
        exam_indices: np.ndarray | list[int],
        num_leads: int = 12,
        *,
        arrays: GraphIndexArrays | None = None,
        symmetrize_edges: bool = False,
        subgraph_representation: SubgraphRepresentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.exam_indices = np.asarray(exam_indices, dtype=np.int64)
        self.num_leads = int(num_leads)
        self.symmetrize_edges = bool(symmetrize_edges)
        self.subgraph_representation = (
            subgraph_representation
            if subgraph_representation is not None
            else SubgraphRepresentationConfig()
        )
        self._h5: h5py.File | None = None
        self._datasets: dict[str, h5py.Dataset] | None = None
        self._validated_items = np.zeros(self.exam_indices.size, dtype=bool)
        self._validation_pid: int | None = None

        if arrays is None:
            arrays = GraphIndexArrays.from_hdf5(self.h5_path)
        self._arrays = arrays
        self.edge_attr_names = arrays.edge_attr_names
        self.num_exams = arrays.num_exams
        self.num_classes = arrays.num_classes
        self.exam_id = arrays.exam_id
        self.exam_to_graph_start = arrays.exam_to_graph_start
        self.exam_to_graph_end = arrays.exam_to_graph_end
        self.graph_lead = arrays.graph_lead
        self.graph_beat_index = arrays.graph_beat_index
        self.graph_node_ptr = arrays.graph_node_ptr
        self.graph_edge_ptr = arrays.graph_edge_ptr

        if self.graph_lead.size > 0:
            lead_min = int(self.graph_lead.min())
            lead_max = int(self.graph_lead.max())
            if lead_min < 0 or lead_max >= self.num_leads:
                raise ValueError(
                    f"{self.h5_path}: graph_lead values span [{lead_min}, {lead_max}] but "
                    f"num_leads={self.num_leads} expects [0, {self.num_leads}). This signals "
                    "a dataset/model num_leads mismatch. Fix the config rather than letting "
                    "the lead embedding fail (or silently clamp) inside the model forward."
                )

        if self.exam_indices.size == 0:
            raise ValueError("exam_indices is empty")
        if self.exam_indices.min() < 0 or self.exam_indices.max() >= self.num_exams:
            raise ValueError(
                f"exam_indices out of range [0, {self.num_exams}): "
                f"got [{self.exam_indices.min()}, {self.exam_indices.max()}]"
            )

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path,
                "r",
                rdcc_nbytes=64 * 1024 * 1024,
                rdcc_nslots=100_003,
            )
            self._datasets = {name: self._h5[name] for name in self._dataset_names()}
        return self._h5

    @staticmethod
    def _dataset_names() -> tuple[str, ...]:
        return ("node_x", "node_t", "edge_index")

    @property
    def datasets(self) -> dict[str, h5py.Dataset]:
        if self._datasets is None:
            _ = self.h5
        assert self._datasets is not None
        return self._datasets

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._datasets = None

    def __len__(self) -> int:
        return int(self.exam_indices.size)

    def effective_edges_per_exam(self) -> np.ndarray:
        """Effective edge totals after representation selection and direction policy."""
        if self.subgraph_representation.adjacency == "stored_visibility":
            graph_edge_counts = np.diff(self.graph_edge_ptr)
        else:
            graph_edge_counts = np.maximum(np.diff(self.graph_node_ptr) - 1, 0)
        prefix = np.empty(graph_edge_counts.size + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(graph_edge_counts, dtype=np.int64, out=prefix[1:])
        counts = prefix[self.exam_to_graph_end] - prefix[self.exam_to_graph_start]
        if self.symmetrize_edges:
            counts = counts * 2
        return counts

    def _needs_invariant_validation(self, item_idx: int) -> bool:
        """Return whether this view item needs validation in the current process."""
        current_pid = os.getpid()
        if self._validation_pid != current_pid:
            self._validated_items.fill(False)
            self._validation_pid = current_pid
        return not bool(self._validated_items[item_idx])

    def __getitem__(self, idx: int) -> HierData:
        item_idx = operator.index(idx)
        if item_idx < 0:
            item_idx += len(self)
        if item_idx < 0 or item_idx >= len(self):
            raise IndexError(f"dataset index out of range: {idx}")
        exam_idx = int(self.exam_indices[item_idx])
        validate = self._needs_invariant_validation(item_idx)
        dsets = self.datasets

        g_start = int(self.exam_to_graph_start[exam_idx])
        g_end = int(self.exam_to_graph_end[exam_idx])
        num_subgraphs = g_end - g_start
        if num_subgraphs == 0:
            raise RuntimeError(f"Exam {exam_idx} (id={self.exam_id[exam_idx]!r}) has no subgraphs")

        graph_lead = self.graph_lead[g_start:g_end].astype(np.int64)
        graph_beat = self.graph_beat_index[g_start:g_end].astype(np.int64)

        node_ptr = self.graph_node_ptr[g_start : g_end + 1]
        n_start = int(node_ptr[0])
        n_end = int(node_ptr[-1])
        node_x = dsets["node_x"][n_start:n_end]
        node_t = dsets["node_t"][n_start:n_end]
        local_node_ptr = node_ptr - n_start

        edge_ptr = self.graph_edge_ptr[g_start : g_end + 1]
        e_start = int(edge_ptr[0])
        e_end = int(edge_ptr[-1])
        edge_index = dsets["edge_index"][:, e_start:e_end].astype(np.int64)
        local_edge_ptr = edge_ptr - e_start

        edge_offsets = np.repeat(local_node_ptr[:-1], np.diff(local_edge_ptr))
        edge_index = edge_index + edge_offsets[None, :]

        edge_index, edge_attr = materialize_subgraph_representation(
            edge_index,
            local_node_ptr,
            node_t,
            node_x,
            self.edge_attr_names,
            self.subgraph_representation,
            local_edge_ptr=local_edge_ptr,
            symmetrize_edges=self.symmetrize_edges,
            validate=validate,
        )

        node_counts = np.diff(local_node_ptr)
        subgraph_node_batch = np.repeat(np.arange(num_subgraphs, dtype=np.int64), node_counts)

        super_edge_index, super_edge_step = _build_super_edges(
            graph_lead, graph_beat, validate=validate
        )
        if validate:
            self._validated_items[item_idx] = True

        return HierData(
            x=torch.from_numpy(np.ascontiguousarray(node_x)).float(),
            edge_index=torch.from_numpy(np.ascontiguousarray(edge_index)).long(),
            edge_attr=torch.from_numpy(np.ascontiguousarray(edge_attr)).float(),
            subgraph_node_batch=torch.from_numpy(subgraph_node_batch).long(),
            super_edge_index=torch.from_numpy(super_edge_index).long(),
            super_edge_step=torch.from_numpy(super_edge_step).long(),
            subgraph_lead=torch.from_numpy(graph_lead).long(),
            subgraph_exam=torch.zeros(num_subgraphs, dtype=torch.long),
            subgraph_beat=torch.from_numpy(graph_beat).long(),
            y=torch.from_numpy(self._arrays.exam_labels[exam_idx].astype(np.float32)),
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_datasets"] = None
        state["_validated_items"] = np.zeros_like(self._validated_items)
        state["_validation_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
