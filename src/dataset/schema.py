"""Single source of truth for the current HDF5 dataset contract.

Every dataset (name, dtype, axis kind), every required root attribute, and
every storage constraint are declared here once:

- :data:`DATASET_SPECS` drives dataset creation in
  :class:`dataset.hdf5_writer.DatasetWriter` and contract validation in
  :mod:`dataset.qa`, so the writer and validators cannot drift apart.
- :data:`REQUIRED_ROOT_ATTRS` is the metadata contract: the writer
  self-checks it at finalize and readers validate files against it.
- :func:`require_dataset_contract` performs the fail-closed structural and
  bounded label validation shared by training and metadata readers.
- :class:`GraphSlice` is the shared per-graph read pattern (CSR pointer
  lookup + node/edge slicing + edge-attr derivation) for consumers that
  inspect individual graphs (visualization, analysis scripts).

Generated datasets must satisfy this one contract. Rebuild a dataset that
does not.

Axis kinds (mirroring the layout docstring in :mod:`dataset.hdf5_writer`):

- ``"exam"``:      one row per exam            ``[E]`` / ``[E, C]``
- ``"graph"``:     one row per beat graph      ``[G]``
- ``"graph_ptr"``: CSR pointer arrays          ``[G + 1]``
- ``"node"``:      one row per node            ``[N]`` / ``[N, F]``
- ``"edge"``:      one column per edge         ``[2, M]``

This module is deliberately free of build-only dependencies (``ts2vg``,
``scipy``) so training-side consumers can import it cheaply.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from pydantic import ValidationError

from config.models import DatasetConfig
from dataset.edge_attrs import compute_edge_attrs
from dataset.labels import validate_label_definition, validate_label_matrix

VLEN_STR = h5py.string_dtype(encoding="utf-8")


def decode_vlen_str(value: Any) -> str:
    """Normalize one value read back from a :data:`VLEN_STR` dataset to ``str``.

    h5py yields ``bytes`` for variable-length UTF-8 datasets under plain
    indexing but ``str`` under an ``asstr()`` view, and NumPy scalar wrappers
    appear in either case. Readers of ``exam_id``/``exam_patient_id`` must not
    each re-derive that rule, so the module declaring the write dtype also
    declares the read-back normalization.
    """
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


EDGE_INDEX_DTYPE = np.int16
MAX_NODES_PER_GRAPH = np.iinfo(EDGE_INDEX_DTYPE).max + 1
EDGE_ORDERING = "src_lt_dst"

AxisKind = Literal["exam", "graph", "graph_ptr", "node", "edge"]


@dataclass(frozen=True)
class DatasetSpec:
    """Declaration of one HDF5 dataset in the schema.

    ``width_key`` names the build-time width of a 2-D dataset (resolved by
    the writer from the dataset config); ``None`` means 1-D. ``graph_field``
    names the :class:`dataset.records.GraphRecord` attribute a per-graph
    scalar column is read from; the writer appends those columns
    generically, so adding one is an edit here plus the record field.
    """

    name: str
    dtype: Any
    kind: AxisKind
    width_key: Literal["num_label_classes", "node_feature_dim"] | None = None
    graph_field: str | None = None
    required: bool = True


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("exam_id", VLEN_STR, "exam"),
    DatasetSpec("exam_patient_id", VLEN_STR, "exam", required=False),
    DatasetSpec("exam_strat_fold", np.int8, "exam", required=False),
    DatasetSpec("exam_labels", np.uint8, "exam", width_key="num_label_classes"),
    DatasetSpec("graph_exam_index", np.int64, "graph"),
    DatasetSpec("graph_lead", np.int16, "graph", graph_field="lead"),
    DatasetSpec("graph_beat_index", np.int64, "graph", graph_field="beat_index"),
    DatasetSpec("graph_start_sample", np.int64, "graph", graph_field="start_sample"),
    DatasetSpec("graph_end_sample", np.int64, "graph", graph_field="end_sample"),
    DatasetSpec("graph_r_peak_sample", np.int64, "graph", graph_field="r_peak_sample"),
    DatasetSpec("graph_boundary_type", np.int8, "graph", graph_field="boundary_type"),
    DatasetSpec("graph_rpeak_polarity", np.int8, "graph", graph_field="rpeak_polarity"),
    DatasetSpec("graph_original_num_nodes", np.int64, "graph", graph_field="original_num_nodes"),
    DatasetSpec("graph_reduced_num_nodes", np.int64, "graph", graph_field="reduced_num_nodes"),
    DatasetSpec("graph_node_ptr", np.int64, "graph_ptr"),
    DatasetSpec("graph_edge_ptr", np.int64, "graph_ptr"),
    DatasetSpec("node_x", np.float32, "node", width_key="node_feature_dim"),
    DatasetSpec("node_t", np.float32, "node"),
    DatasetSpec("node_original_sample", np.int64, "node"),
    DatasetSpec("edge_index", EDGE_INDEX_DTYPE, "edge"),
)

DATASET_NAMES: tuple[str, ...] = tuple(spec.name for spec in DATASET_SPECS if spec.required)

REQUIRED_DTYPES: dict[str, np.dtype] = {
    spec.name: np.dtype(spec.dtype) for spec in DATASET_SPECS if spec.dtype is not VLEN_STR
}

GRAPH_SCALAR_SPECS: tuple[DatasetSpec, ...] = tuple(
    spec for spec in DATASET_SPECS if spec.graph_field is not None
)

_LABEL_VALIDATION_CHUNK_ROWS = 1_048_576


REQUIRED_ROOT_ATTRS: tuple[str, ...] = (
    "created_by_version",
    "created_at",
    "dataset_source",
    "source_variant",
    "sampling_rate",
    "source_sampling_rate",
    "segmentation_mode",
    "node_reduction_method",
    "graph_leads",
    "node_feature_names",
    "edge_attr_names",
    "edge_ordering",
    "exam_label_taxonomy",
    "exam_label_names",
    "config_json",
    "rpeak_polarity_mode",
)


def require_dataset_contract(h5: h5py.File, h5_path: Path) -> DatasetConfig:
    """Validate the structural and bounded label contract, then return the config.

    This checks structure rather than a version label. It also validates the
    root label definition and every stored label value in bounded row chunks.
    Full graph-array invariants remain the responsibility of :mod:`dataset.qa`;
    this seam catches malformed labels before any consumer casts or uses them.
    """
    missing_datasets = [name for name in DATASET_NAMES if name not in h5]
    if missing_datasets:
        raise ValueError(f"{h5_path}: missing required datasets {missing_datasets}")
    missing_attrs = [name for name in REQUIRED_ROOT_ATTRS if name not in h5.attrs]
    if missing_attrs:
        raise ValueError(f"{h5_path}: missing required root attributes {missing_attrs}")
    for name, expected in REQUIRED_DTYPES.items():
        if name not in h5:
            continue
        actual = np.dtype(h5[name].dtype)
        if actual != expected:
            raise ValueError(f"{h5_path}: {name} dtype={actual}, expected {expected}")
    if h5["edge_index"].ndim != 2 or h5["edge_index"].shape[0] != 2:
        raise ValueError(
            f"{h5_path}: edge_index shape={h5['edge_index'].shape}, expected (2, n_edges)"
        )

    label_names, _label_taxonomy = validate_label_definition(
        h5.attrs["exam_label_names"],
        h5.attrs["exam_label_taxonomy"],
        context=f"{h5_path} root label contract",
    )
    labels_ds = h5["exam_labels"]
    if labels_ds.ndim != 2:
        raise ValueError(f"{h5_path}: exam_labels shape={labels_ds.shape}, expected rank 2")
    n_exams = int(h5["exam_id"].shape[0])
    expected_shape = (n_exams, len(label_names))
    if labels_ds.shape != expected_shape:
        raise ValueError(
            f"{h5_path}: exam_labels shape={labels_ds.shape}, expected {expected_shape}"
        )
    for start in range(0, n_exams, _LABEL_VALIDATION_CHUNK_ROWS):
        stop = min(start + _LABEL_VALIDATION_CHUNK_ROWS, n_exams)
        validate_label_matrix(
            labels_ds[start:stop],
            num_exams=stop - start,
            num_classes=len(label_names),
            context=f"{h5_path} exam_labels rows {start}:{stop}",
        )

    raw_config = h5.attrs["config_json"]
    try:
        payload = json.loads(str(raw_config))
        config = DatasetConfig.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"{h5_path}: config_json does not satisfy DatasetConfig: {exc}") from exc
    canonical_payload = json.loads(config.model_dump_json())
    if payload != canonical_payload:
        raise ValueError(
            f"{h5_path}: config_json is not the complete canonical DatasetConfig payload"
        )

    expected_attrs: tuple[tuple[str, object], ...] = (
        ("dataset_source", config.dataset_source),
        ("source_variant", config.source_variant),
        ("sampling_rate", config.effective_sampling_rate),
        ("source_sampling_rate", config.sampling_rate),
        ("segmentation_mode", "per_lead"),
        ("node_reduction_method", config.node_reduction.method),
        ("edge_ordering", EDGE_ORDERING),
        ("rpeak_polarity_mode", config.rpeak_detection.polarity_mode),
    )
    for name, expected in expected_attrs:
        actual = h5.attrs[name]
        if str(actual) != str(expected):
            raise ValueError(
                f"{h5_path}: root attr {name}={actual!r} disagrees with "
                f"the current contract value {expected!r}"
            )

    expected_optional_attrs: tuple[tuple[str, object | None], ...] = (
        (
            "amplitude_normalization",
            config.amplitude_normalization.method
            if config.amplitude_normalization is not None
            else None,
        ),
        (
            "signal_filter",
            config.signal_filter.method if config.signal_filter is not None else None,
        ),
        (
            "signal_filter_highpass_hz",
            config.signal_filter.highpass_hz if config.signal_filter is not None else None,
        ),
        (
            "signal_filter_lowpass_hz",
            config.signal_filter.lowpass_hz if config.signal_filter is not None else None,
        ),
        (
            "signal_filter_order",
            config.signal_filter.order if config.signal_filter is not None else None,
        ),
    )
    for name, expected in expected_optional_attrs:
        actual = h5.attrs.get(name)
        if expected is None and actual is not None:
            raise ValueError(f"{h5_path}: root attr {name}={actual!r} is absent from config_json")
        if expected is not None and (actual is None or str(actual) != str(expected)):
            raise ValueError(
                f"{h5_path}: root attr {name}={actual!r} disagrees with "
                f"config_json value {expected!r}"
            )

    graph_leads = [int(lead) for lead in h5.attrs["graph_leads"]]
    if graph_leads != config.graph_leads:
        raise ValueError(
            f"{h5_path}: graph_leads={graph_leads!r} disagrees with "
            f"config_json {config.graph_leads!r}"
        )
    node_features = [str(name) for name in h5.attrs["node_feature_names"]]
    if node_features != config.node_features:
        raise ValueError(
            f"{h5_path}: node_feature_names={node_features!r} disagrees with "
            f"config_json {config.node_features!r}"
        )
    edge_attrs = [str(name) for name in h5.attrs["edge_attr_names"]]
    if edge_attrs != config.graph.edge_attr_names:
        raise ValueError(
            f"{h5_path}: edge_attr_names={edge_attrs!r} disagrees with "
            f"config_json {config.graph.edge_attr_names!r}"
        )
    return config


@dataclass(frozen=True)
class GraphSlice:
    """One beat graph read out of a conforming HDF5 file.

    ``edge_index`` is returned as int64 **local** indices (``0..N-1`` within
    this graph), the canonical compute representation every consumer casts
    to anyway. ``edge_attr`` is derived per the file's ``edge_attr_names``
    attr (``None`` when the attr is empty).
    """

    graph_idx: int
    exam_idx: int
    lead: int
    beat_index: int
    boundary_type: int
    rpeak_polarity: int
    start_sample: int
    end_sample: int
    r_peak_sample: int
    original_num_nodes: int
    reduced_num_nodes: int
    node_x: np.ndarray
    node_t: np.ndarray
    node_original_sample: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray | None
    sampling_rate: int

    @classmethod
    def from_hdf5(cls, h5: h5py.File, graph_idx: int) -> GraphSlice:
        """Read one graph: CSR pointer lookup, node/edge slices, derived attrs."""
        n_lo, n_hi = (int(v) for v in h5["graph_node_ptr"][graph_idx : graph_idx + 2])
        e_lo, e_hi = (int(v) for v in h5["graph_edge_ptr"][graph_idx : graph_idx + 2])

        node_x = np.asarray(h5["node_x"][n_lo:n_hi])
        node_t = np.asarray(h5["node_t"][n_lo:n_hi])
        edge_index = np.asarray(h5["edge_index"][:, e_lo:e_hi], dtype=np.int64)

        edge_attr_names = [str(n) for n in h5.attrs["edge_attr_names"]]
        edge_attr = (
            compute_edge_attrs(edge_index, node_t, node_x, edge_attr_names)
            if edge_attr_names
            else None
        )

        return cls(
            graph_idx=int(graph_idx),
            exam_idx=int(h5["graph_exam_index"][graph_idx]),
            lead=int(h5["graph_lead"][graph_idx]),
            beat_index=int(h5["graph_beat_index"][graph_idx]),
            boundary_type=int(h5["graph_boundary_type"][graph_idx]),
            rpeak_polarity=int(h5["graph_rpeak_polarity"][graph_idx]),
            start_sample=int(h5["graph_start_sample"][graph_idx]),
            end_sample=int(h5["graph_end_sample"][graph_idx]),
            r_peak_sample=int(h5["graph_r_peak_sample"][graph_idx]),
            original_num_nodes=int(h5["graph_original_num_nodes"][graph_idx]),
            reduced_num_nodes=int(h5["graph_reduced_num_nodes"][graph_idx]),
            node_x=node_x,
            node_t=node_t,
            node_original_sample=np.asarray(h5["node_original_sample"][n_lo:n_hi]),
            edge_index=edge_index,
            edge_attr=edge_attr,
            sampling_rate=int(h5.attrs["sampling_rate"]),
        )
