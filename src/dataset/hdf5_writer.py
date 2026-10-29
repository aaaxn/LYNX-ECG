"""Streaming writer for the current HDF5 dataset contract.

Layout (dual: per-exam table + per-graph indirection):

    exam_id              str   [E]
    exam_patient_id      str   [E]
    exam_strat_fold      i8    [E]
    exam_labels          u8    [E, C]

    graph_exam_index     i64   [G]
    graph_lead           i16   [G]
    graph_beat_index     i64   [G]
    graph_start_sample   i64   [G]
    graph_end_sample     i64   [G]
    graph_r_peak_sample  i64   [G]
    graph_boundary_type  i8    [G]
    graph_rpeak_polarity i8    [G]
    graph_original_num_nodes i64 [G]
    graph_reduced_num_nodes  i64 [G]
    graph_node_ptr       i64   [G + 1]
    graph_edge_ptr       i64   [G + 1]

    node_x               f32   [N, F]
    node_t               f32   [N]
    node_original_sample i64   [N]

    edge_index           i16   [2, M]   # local indices (src < dst)

``edge_index`` values are subgraph-local node indices, so int16 holds them
as long as every graph has at most ``INT16_MAX + 1`` nodes. The writer
enforces this per graph and fails loudly otherwise.

``edge_attr`` is NOT stored: the ``edge_attr_names`` attr declares which
attributes loaders must derive from ``node_t`` / ``node_x`` (see
:mod:`dataset.edge_attrs`).

Pointer arrays are CSR-style: graph ``g`` owns nodes
``[graph_node_ptr[g], graph_node_ptr[g+1])`` and edges
``[graph_edge_ptr[g], graph_edge_ptr[g+1])``.

Dataset names, dtypes, and required root attrs are declared once in
:mod:`dataset.schema`; this writer creates datasets from that table and
self-checks the attr contract at finalize.

Workflow::

    writer = DatasetWriter(path, config=cfg, label_taxonomy=..., label_names=...)
    for exam_batches in stream_of_packed_results:
        writer.add_packed_exam_batch(exam_batches)
    writer.finalize(summary=build_summary)
"""

from __future__ import annotations

import contextlib
import datetime
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from config.models import DatasetConfig
from dataset.labels import validate_label_definition, validate_label_vector
from dataset.records import ExamMetadata, PackedGraphBatch
from dataset.schema import (
    DATASET_SPECS,
    EDGE_ORDERING,
    GRAPH_SCALAR_SPECS,
    REQUIRED_ROOT_ATTRS,
)

_DEFAULT_CHUNK_NODES = 8192
_DEFAULT_CHUNK_EDGES = 8192
_DEFAULT_CHUNK_GRAPHS = 1024
_DEFAULT_CHUNK_EXAMS = 1024
_CHUNK_ROWS_BY_KIND = {
    "exam": _DEFAULT_CHUNK_EXAMS,
    "graph": _DEFAULT_CHUNK_GRAPHS,
    "node": _DEFAULT_CHUNK_NODES,
}


class DatasetWriter:
    """Append-only writer for the current HDF5 dataset contract."""

    def __init__(
        self,
        output_path: Path,
        *,
        config: DatasetConfig,
        label_taxonomy: str,
        label_names: tuple[str, ...],
        created_by_version: str,
        overwrite: bool = False,
    ) -> None:
        path = Path(output_path)
        validated_names, validated_taxonomy = validate_label_definition(
            label_names,
            label_taxonomy,
            context=f"writer {path}",
        )
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass overwrite=True or use --overwrite."
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        self.path = path
        self.config = config
        self.label_taxonomy = validated_taxonomy
        self.label_names = validated_names
        self.created_by_version = created_by_version

        self._num_label_classes = len(self.label_names)
        self._node_feature_names = list(config.node_features)
        self._node_feature_dim = len(self._node_feature_names)
        self._edge_attr_names = list(config.graph.edge_attr_names)

        widths = {
            "num_label_classes": self._num_label_classes,
            "node_feature_dim": self._node_feature_dim,
        }
        self._h5 = h5py.File(self.path, "w")
        bulk_kwargs: dict[str, Any] = {"compression": "lzf", "shuffle": True}
        for spec in DATASET_SPECS:
            if spec.kind == "graph_ptr":
                continue
            extra = bulk_kwargs if spec.kind in ("node", "edge") else {}
            if spec.kind == "edge":
                self._h5.create_dataset(
                    spec.name,
                    shape=(2, 0),
                    maxshape=(2, None),
                    dtype=spec.dtype,
                    chunks=(2, _DEFAULT_CHUNK_EDGES),
                    **extra,
                )
            elif spec.width_key is None:
                self._h5.create_dataset(
                    spec.name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=spec.dtype,
                    chunks=(_CHUNK_ROWS_BY_KIND[spec.kind],),
                    **extra,
                )
            else:
                width = widths[spec.width_key]
                self._h5.create_dataset(
                    spec.name,
                    shape=(0, width),
                    maxshape=(None, width),
                    dtype=spec.dtype,
                    chunks=(_CHUNK_ROWS_BY_KIND[spec.kind], max(width, 1)),
                    **extra,
                )
        self._graph_node_ptr: list[int] = [0]
        self._graph_edge_ptr: list[int] = [0]
        self._n_exams = 0
        self._n_graphs = 0
        self._n_nodes = 0
        self._n_edges = 0
        self._exam_id_to_index: dict[str, int] = {}

    @property
    def n_exams(self) -> int:
        return self._n_exams

    @property
    def n_graphs(self) -> int:
        return self._n_graphs

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def n_edges(self) -> int:
        return self._n_edges

    def add_packed_exam_batch(
        self, exam_batches: list[tuple[ExamMetadata, PackedGraphBatch]]
    ) -> None:
        """Append multiple exams with one resize/write batch per dataset.

        Each exam's graphs arrive pre-packed (and pre-validated, see
        :func:`dataset.records.pack_graphs`) in write order; the writer only
        concatenates across exams and appends.
        """
        items = list(exam_batches)
        if not items:
            return

        seen_in_batch: set[str] = set()
        labels_batch: list[np.ndarray] = []
        for metadata, packed in items:
            if metadata.exam_id in self._exam_id_to_index or metadata.exam_id in seen_in_batch:
                raise ValueError(f"Duplicate exam_id={metadata.exam_id!r} in writer")
            seen_in_batch.add(metadata.exam_id)
            if packed.n_graphs <= 0:
                raise ValueError(f"exam_id={metadata.exam_id!r} has zero graphs")

            context = f"writer exam {metadata.exam_id!r}"
            names, _taxonomy = validate_label_definition(
                metadata.label_names,
                metadata.label_taxonomy,
                context=context,
                expected_names=self.label_names,
                expected_taxonomy=self.label_taxonomy,
            )
            labels_batch.append(
                validate_label_vector(
                    metadata.labels,
                    num_classes=len(names),
                    context=f"{context} labels",
                )
            )

        start_exam_idx = self._n_exams
        self._append_1d(
            self._h5["exam_id"],
            np.asarray([metadata.exam_id for metadata, _packed in items], dtype=object),
        )
        self._append_1d(
            self._h5["exam_patient_id"],
            np.asarray([metadata.patient_id for metadata, _packed in items], dtype=object),
        )
        self._append_1d(
            self._h5["exam_strat_fold"],
            np.asarray([metadata.strat_fold for metadata, _packed in items], dtype=np.int8),
        )
        self._append_2d(self._h5["exam_labels"], np.stack(labels_batch, axis=0))

        for offset, (metadata, _packed) in enumerate(items):
            self._exam_id_to_index[metadata.exam_id] = start_exam_idx + offset
        self._n_exams += len(items)

        packs = [packed for _metadata, packed in items]
        graph_exam_index = np.repeat(
            np.arange(start_exam_idx, start_exam_idx + len(items), dtype=np.int64),
            [p.n_graphs for p in packs],
        )
        self._append_packed(packs, graph_exam_index)

    def _append_packed(self, packs: list[PackedGraphBatch], graph_exam_index: np.ndarray) -> None:
        self._append_1d(self._h5["graph_exam_index"], graph_exam_index)
        for spec in GRAPH_SCALAR_SPECS:
            self._append_1d(
                self._h5[spec.name], np.concatenate([p.scalars[spec.name] for p in packs])
            )

        self._append_2d(self._h5["node_x"], np.concatenate([p.node_x for p in packs], axis=0))
        self._append_1d(self._h5["node_t"], np.concatenate([p.node_t for p in packs]))
        self._append_1d(
            self._h5["node_original_sample"],
            np.concatenate([p.node_original_sample for p in packs]),
        )
        edge_batch = np.concatenate([p.edge_index for p in packs], axis=1)
        if edge_batch.shape[1] > 0:
            self._append_axis(self._h5["edge_index"], edge_batch, axis=1)

        node_counts = np.concatenate([p.node_counts for p in packs])
        edge_counts = np.concatenate([p.edge_counts for p in packs])
        self._graph_node_ptr.extend((self._n_nodes + np.cumsum(node_counts)).tolist())
        self._graph_edge_ptr.extend((self._n_edges + np.cumsum(edge_counts)).tolist())
        self._n_nodes += int(node_counts.sum())
        self._n_edges += int(edge_counts.sum())
        self._n_graphs += int(node_counts.size)

    def finalize(self, *, summary: dict[str, Any] | None = None) -> None:
        """Write pointer arrays and global attrs, then close the file."""
        self._h5.create_dataset(
            "graph_node_ptr",
            data=np.asarray(self._graph_node_ptr, dtype=np.int64),
        )
        self._h5.create_dataset(
            "graph_edge_ptr",
            data=np.asarray(self._graph_edge_ptr, dtype=np.int64),
        )

        attrs = self._h5.attrs
        attrs["created_by_version"] = self.created_by_version
        attrs["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()

        attrs["dataset_source"] = self.config.dataset_source
        attrs["source_variant"] = self.config.source_variant
        attrs["sampling_rate"] = int(self.config.effective_sampling_rate)
        attrs["source_sampling_rate"] = int(self.config.sampling_rate)

        if self.config.signal_filter is not None:
            sf = self.config.signal_filter
            attrs["signal_filter"] = sf.method
            attrs["signal_filter_highpass_hz"] = float(sf.highpass_hz)
            attrs["signal_filter_lowpass_hz"] = float(sf.lowpass_hz)
            attrs["signal_filter_order"] = int(sf.order)

        if self.config.amplitude_normalization is not None:
            attrs["amplitude_normalization"] = self.config.amplitude_normalization.method

        attrs["segmentation_mode"] = "per_lead"
        attrs["node_reduction_method"] = self.config.node_reduction.method
        _method = self.config.node_reduction.method
        _sub = getattr(self.config.node_reduction, _method, None) if _method != "none" else None
        if _sub is not None and hasattr(_sub, "target_nodes"):
            attrs["target_nodes"] = int(_sub.target_nodes)
        attrs["graph_leads"] = np.asarray(self.config.graph_leads, dtype=np.int16)

        attrs["node_feature_names"] = list(self._node_feature_names)
        attrs["edge_attr_names"] = list(self._edge_attr_names)
        attrs["edge_ordering"] = EDGE_ORDERING

        attrs["exam_label_taxonomy"] = self.label_taxonomy
        attrs["exam_label_names"] = list(self.label_names)

        attrs["rpeak_polarity_mode"] = self.config.rpeak_detection.polarity_mode

        attrs["config_json"] = self.config.model_dump_json()

        attrs["n_graphs_written"] = int(self._n_graphs)
        attrs["n_nodes_written"] = int(self._n_nodes)
        attrs["n_edges_written"] = int(self._n_edges)
        attrs["n_exams_kept"] = int(self._n_exams)

        if summary:
            for key, value in summary.items():
                if isinstance(value, dict):
                    attrs[key] = json.dumps(value)
                else:
                    attrs[key] = value

        missing_attrs = [name for name in REQUIRED_ROOT_ATTRS if name not in attrs]
        if missing_attrs:
            raise RuntimeError(
                f"DatasetWriter bug: required root attrs not written: {missing_attrs}"
            )

        self._h5.flush()
        self._h5.close()

    @staticmethod
    def _append_1d(ds: h5py.Dataset, data: np.ndarray) -> None:
        n_old = ds.shape[0]
        n_new = n_old + data.shape[0]
        ds.resize((n_new,))
        ds[n_old:n_new] = data

    @staticmethod
    def _append_2d(ds: h5py.Dataset, data: np.ndarray) -> None:
        n_old = ds.shape[0]
        n_new = n_old + data.shape[0]
        ds.resize((n_new, ds.shape[1]))
        ds[n_old:n_new, :] = data

    @staticmethod
    def _append_axis(ds: h5py.Dataset, data: np.ndarray, *, axis: int) -> None:
        new_shape = list(ds.shape)
        new_shape[axis] += data.shape[axis]
        ds.resize(tuple(new_shape))
        if axis == 0:
            ds[ds.shape[0] - data.shape[0] :] = data
        elif axis == 1:
            ds[:, ds.shape[1] - data.shape[1] :] = data
        else:  # pragma: no cover
            raise ValueError(f"Unsupported axis: {axis}")

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._h5:
            return
        try:
            if exc_type is None:
                self.finalize()
        finally:
            if self._h5:
                with contextlib.suppress(Exception):
                    self._h5.close()
