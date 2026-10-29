"""Validation profiles for the current HDF5 dataset contract.

Profiles:

- ``"quick"``: contract checks, required keys, dtypes, metadata binding,
  pointer monotonicity, and basic shape consistency. Runs in O(num_graphs) time.
- ``"full"`` : ``quick`` + per-graph cross-checks: ``node_t`` formula
  consistency with ``node_original_sample / sampling_rate``, edge_index
  ``src < dst`` invariant, edge_index local-bound consistency, derived
  ``delta_t`` positivity, no NaN/Inf in ``node_x`` / ``node_t``. Every graph is validated, with no
  sampling.

Memory: validation streams every whole-array check in ``chunk_size``-element
chunks and reads per-graph slices directly from HDF5, so peak memory is
bounded by the chunk size, not the file size. Code100-scale files (billions
of nodes, tens of billions of edges) validate in a few hundred MB of RAM.

The schema contract being validated (dataset names, dtypes, required root
attrs) is declared in :mod:`dataset.schema`, the same table the writer
creates files from.

Returns ``QAResult`` with ``success: bool`` and a list of issues. CLI
exposes exit codes (0 success, 1 schema/contract violation, 2 execution
error).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

from dataset.edge_attrs import compute_edge_attrs
from dataset.schema import (
    DATASET_NAMES,
    DATASET_SPECS,
    REQUIRED_DTYPES,
    REQUIRED_ROOT_ATTRS,
    require_dataset_contract,
)

DEFAULT_CHUNK_SIZE = 16_777_216


@dataclass
class QAIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    success: bool
    profile: str
    issues: list[QAIssue]
    summary: dict[str, Any]

    def errors(self) -> list[QAIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "profile": self.profile,
            "issues": [
                {"severity": i.severity, "code": i.code, "message": i.message, "details": i.details}
                for i in self.issues
            ],
            "summary": self.summary,
        }


def _iter_slices(total: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def _ptr_is_non_decreasing(ds: h5py.Dataset, chunk_size: int) -> bool:
    """Streamed equivalent of ``np.all(np.diff(ds) >= 0)``."""
    prev: np.generic | None = None
    for lo, hi in _iter_slices(ds.shape[0], chunk_size):
        arr = ds[lo:hi]
        if prev is not None and arr[0] < prev:
            return False
        if arr.size > 1 and np.any(np.diff(arr) < 0):
            return False
        prev = arr[-1]
    return True


def _check_graph_exam_ownership(
    f: h5py.File,
    issues: list[QAIssue],
    n_exams: int,
    chunk_size: int,
) -> None:
    """Validate graph ownership range, order, and complete exam coverage."""
    ds = f["graph_exam_index"]
    if ds.ndim != 1 or np.dtype(ds.dtype) != np.dtype(np.int64):
        return
    first_value: int | None = None
    previous: int | None = None
    range_failure: tuple[int, int] | None = None
    order_failure: tuple[int, int, int] | None = None
    first_gap: int | None = None

    for lo, hi in _iter_slices(int(ds.shape[0]), chunk_size):
        values = ds[lo:hi]
        if values.size == 0:
            continue

        if first_value is None:
            first_value = int(values[0])

        if range_failure is None:
            invalid = (values < 0) | (values >= n_exams)
            invalid_positions = np.flatnonzero(invalid)
            if invalid_positions.size:
                offset = int(invalid_positions[0])
                range_failure = (lo + offset, int(values[offset]))

        if order_failure is None:
            if previous is not None and int(values[0]) < previous:
                order_failure = (lo, previous, int(values[0]))
            else:
                decreases = np.flatnonzero(values[1:] < values[:-1])
                if decreases.size:
                    offset = int(decreases[0]) + 1
                    order_failure = (
                        lo + offset,
                        int(values[offset - 1]),
                        int(values[offset]),
                    )

        if first_gap is None and range_failure is None and order_failure is None:
            if previous is not None and int(values[0]) > previous + 1:
                first_gap = previous + 1
            else:
                jumps = np.flatnonzero(values[1:] > values[:-1] + 1)
                if jumps.size:
                    first_gap = int(values[int(jumps[0])]) + 1

        previous = int(values[-1])

    if range_failure is not None:
        graph_pos, value = range_failure
        issues.append(
            QAIssue(
                "error",
                "graph_exam_index_out_of_range",
                f"graph_exam_index[{graph_pos}]={value} is outside [0, {n_exams})",
            )
        )
    if order_failure is not None:
        graph_pos, prior, value = order_failure
        issues.append(
            QAIssue(
                "error",
                "graph_exam_index_non_monotonic",
                f"graph_exam_index decreases at graph {graph_pos}: {prior} -> {value}",
            )
        )

    if range_failure is not None or order_failure is not None:
        return

    missing: int | None = None
    if n_exams > 0:
        if first_value is None or first_value > 0:
            missing = 0
        elif first_gap is not None:
            missing = first_gap
        elif previous is not None and previous < n_exams - 1:
            missing = previous + 1

    if missing is not None:
        issues.append(
            QAIssue(
                "error",
                "graph_exam_index_incomplete_coverage",
                "graph_exam_index does not cover every exam in "
                f"[0, {n_exams}); first missing exam index {missing}",
            )
        )


def _reduced_num_matches_ptr_delta(
    reduced_ds: h5py.Dataset, ptr_ds: h5py.Dataset, chunk_size: int
) -> bool:
    """Streamed equivalent of ``np.array_equal(reduced, np.diff(ptr))``."""
    n_graphs = reduced_ds.shape[0]
    for lo, hi in _iter_slices(n_graphs, chunk_size):
        if not np.array_equal(reduced_ds[lo:hi], np.diff(ptr_ds[lo : hi + 1])):
            return False
    return True


def _check_required_datasets(f: h5py.File, issues: list[QAIssue]) -> bool:
    """Verify every dataset in ``DATASET_NAMES`` is present. Emits ``missing_dataset``.

    Returns False if any dataset is missing. The caller must stop early,
    since every later check assumes the basic structure is intact.
    """
    for name in DATASET_NAMES:
        if name not in f:
            issues.append(QAIssue("error", "missing_dataset", f"Missing dataset: {name!r}"))
    return not any(i.code == "missing_dataset" for i in issues)


def _check_contract_attrs(f: h5py.File, issues: list[QAIssue]) -> None:
    """Verify required root attrs. Emits ``missing_attr``."""
    for attr in REQUIRED_ROOT_ATTRS:
        if attr not in f.attrs:
            issues.append(QAIssue("error", "missing_attr", f"Missing global attr: {attr!r}"))


def _check_dtypes(f: h5py.File, issues: list[QAIssue]) -> None:
    """Verify dataset dtypes match ``REQUIRED_DTYPES``. Emits ``dtype_mismatch``."""
    for ds_name, expected in REQUIRED_DTYPES.items():
        if ds_name in f and np.dtype(f[ds_name].dtype) != np.dtype(expected):
            issues.append(
                QAIssue(
                    "error",
                    "dtype_mismatch",
                    f"{ds_name!r} dtype={f[ds_name].dtype}, expected {np.dtype(expected)}",
                )
            )


def _axis_lengths(f: h5py.File, issues: list[QAIssue]) -> tuple[int, int, int, int | None]:
    """Compute (n_exams, n_graphs, n_nodes, n_edges) from the unconditionally-present
    axis-defining datasets. Emits ``shape_mismatch`` if ``edge_index`` isn't 2-D
    with a leading dimension of 2, in which case n_edges is returned as None.
    """
    n_exams = int(f["exam_id"].shape[0])
    n_graphs = int(f["graph_exam_index"].shape[0])
    n_nodes = int(f["node_x"].shape[0])

    ei = f["edge_index"]
    if ei.ndim == 2 and ei.shape[0] == 2:
        n_edges: int | None = int(ei.shape[1])
    else:
        issues.append(
            QAIssue(
                "error",
                "shape_mismatch",
                f"edge_index shape {ei.shape}, expected (2, n_edges)",
            )
        )
        n_edges = None

    return n_exams, n_graphs, n_nodes, n_edges


def _resolve_width(f: h5py.File, width_key: str) -> int | None:
    """Map a ``DatasetSpec.width_key`` to the number of names declared in the
    corresponding root attr. Returns None (never raises) when the attr is
    missing, so a shape check never fires when an attr check already flagged it.
    """
    attr_name = "exam_label_names" if width_key == "num_label_classes" else "node_feature_names"
    raw = f.attrs.get(attr_name)
    return len(raw) if raw is not None else None


def _check_shapes(
    f: h5py.File,
    issues: list[QAIssue],
    n_exams: int,
    n_graphs: int,
    n_nodes: int,
    n_edges: int | None,
) -> None:
    """Walk every ``DATASET_SPECS`` entry and compare the actual HDF5 shape
    against the shape predicted by its axis kind (exam/graph/graph_ptr/node/edge).
    Emits ``shape_mismatch``. Datasets already flagged as missing are skipped.
    """
    for spec in DATASET_SPECS:
        if spec.name not in f:
            continue

        ds = f[spec.name]
        shape = ds.shape

        if spec.kind == "exam":
            if spec.width_key is None:
                expected = (n_exams,)
                if shape != expected:
                    issues.append(
                        QAIssue(
                            "error",
                            "shape_mismatch",
                            f"{spec.name} shape {shape}, expected {expected}",
                        )
                    )
            else:
                width = _resolve_width(f, spec.width_key)
                if width is not None:
                    expected_2d = (n_exams, width)
                    if shape != expected_2d:
                        issues.append(
                            QAIssue(
                                "error",
                                "shape_mismatch",
                                f"{spec.name} shape {shape}, expected {expected_2d}",
                            )
                        )

        elif spec.kind == "graph":
            expected = (n_graphs,)
            if shape != expected:
                issues.append(
                    QAIssue(
                        "error",
                        "shape_mismatch",
                        f"{spec.name} shape {shape}, expected {expected}",
                    )
                )

        elif spec.kind == "graph_ptr":
            expected = (n_graphs + 1,)
            if shape != expected:
                issues.append(
                    QAIssue(
                        "error",
                        "shape_mismatch",
                        f"{spec.name} shape {shape}, expected {expected}",
                    )
                )

        elif spec.kind == "node":
            if spec.width_key is None:
                expected = (n_nodes,)
                if shape != expected:
                    issues.append(
                        QAIssue(
                            "error",
                            "shape_mismatch",
                            f"{spec.name} shape {shape}, expected {expected}",
                        )
                    )
            else:
                width = _resolve_width(f, spec.width_key)
                if width is not None:
                    expected_2d = (n_nodes, width)
                    if shape != expected_2d:
                        issues.append(
                            QAIssue(
                                "error",
                                "shape_mismatch",
                                f"{spec.name} shape {shape}, expected {expected_2d}",
                            )
                        )

        elif spec.kind == "edge":
            if n_edges is None:
                continue
            expected_edge = (2, n_edges)
            if shape != expected_edge:
                issues.append(
                    QAIssue(
                        "error",
                        "shape_mismatch",
                        f"{spec.name} shape {shape}, expected {expected_edge}",
                    )
                )


def _check_pointers(
    f: h5py.File,
    issues: list[QAIssue],
    n_graphs: int,
    n_nodes: int,
    n_edges: int | None,
    chunk_size: int,
) -> tuple[bool, bool]:
    """Verify ``graph_node_ptr``/``graph_edge_ptr`` start at zero, are monotonic,
    sum to n_nodes/n_edges, and that ``graph_reduced_num_nodes`` matches
    diff(graph_node_ptr). Emits ``ptr_starts_nonzero``, ``ptr_non_monotonic``,
    ``ptr_total_mismatch``, ``reduced_num_ptr_mismatch``.

    Returns (node_ptr_ok, edge_ptr_ok), the pointer shape flags. Value checks
    only run when the corresponding pointer array has the correct shape; a
    malformed shape already produced a ``shape_mismatch`` issue in
    ``_check_shapes``.
    """
    node_ptr = f["graph_node_ptr"]
    edge_ptr = f["graph_edge_ptr"]
    node_ptr_ok = node_ptr.shape == (n_graphs + 1,)
    edge_ptr_ok = edge_ptr.shape == (n_graphs + 1,)

    if node_ptr_ok and node_ptr.shape[0] > 0 and node_ptr[0] != 0:
        issues.append(QAIssue("error", "ptr_starts_nonzero", "graph_node_ptr[0] != 0"))
    if edge_ptr_ok and edge_ptr.shape[0] > 0 and edge_ptr[0] != 0:
        issues.append(QAIssue("error", "ptr_starts_nonzero", "graph_edge_ptr[0] != 0"))
    if node_ptr_ok and node_ptr.shape[0] > 1 and not _ptr_is_non_decreasing(node_ptr, chunk_size):
        issues.append(QAIssue("error", "ptr_non_monotonic", "graph_node_ptr not monotonic"))
    if edge_ptr_ok and edge_ptr.shape[0] > 1 and not _ptr_is_non_decreasing(edge_ptr, chunk_size):
        issues.append(QAIssue("error", "ptr_non_monotonic", "graph_edge_ptr not monotonic"))

    if node_ptr_ok and node_ptr.shape[0] > 0 and int(node_ptr[-1]) != n_nodes:
        issues.append(
            QAIssue(
                "error",
                "ptr_total_mismatch",
                f"graph_node_ptr[-1]={int(node_ptr[-1])} but node_x has {n_nodes} rows",
            )
        )
    if (
        edge_ptr_ok
        and n_edges is not None
        and edge_ptr.shape[0] > 0
        and int(edge_ptr[-1]) != n_edges
    ):
        issues.append(
            QAIssue(
                "error",
                "ptr_total_mismatch",
                f"graph_edge_ptr[-1]={int(edge_ptr[-1])} but edge_index has {n_edges} edges",
            )
        )

    reduced_num = f["graph_reduced_num_nodes"]
    if (
        node_ptr_ok
        and reduced_num.shape == (n_graphs,)
        and not _reduced_num_matches_ptr_delta(reduced_num, node_ptr, chunk_size)
    ):
        issues.append(
            QAIssue(
                "error",
                "reduced_num_ptr_mismatch",
                "graph_reduced_num_nodes does not match diff(graph_node_ptr)",
            )
        )

    return node_ptr_ok, edge_ptr_ok


def _check_edge_direction(
    f: h5py.File, issues: list[QAIssue], n_edges: int | None, chunk_size: int
) -> None:
    """Streamed check that ``edge_index`` has src < dst everywhere.
    Emits ``src_lt_dst_violation``.

    n_edges is None when the quick-profile shape check already flagged a
    malformed edge_index; the value check is skipped in that case.
    """
    edge_index = f["edge_index"]
    if n_edges is not None:
        for lo, hi in _iter_slices(n_edges, chunk_size):
            ei_chunk = edge_index[:, lo:hi]
            if not bool(np.all(ei_chunk[0] < ei_chunk[1])):
                issues.append(QAIssue("error", "src_lt_dst_violation", "edge_index has src >= dst"))
                break


def _check_node_finiteness(
    f: h5py.File, issues: list[QAIssue], n_nodes: int, chunk_size: int
) -> None:
    """Streamed finiteness check for ``node_t`` and ``node_x``.
    Emits ``node_t_non_finite``, ``node_x_non_finite``.
    """
    for ds_name, code in (("node_t", "node_t_non_finite"), ("node_x", "node_x_non_finite")):
        ds = f[ds_name]
        for lo, hi in _iter_slices(n_nodes, chunk_size):
            if not bool(np.all(np.isfinite(ds[lo:hi]))):
                issues.append(QAIssue("error", code, f"{ds_name} contains NaN/Inf"))
                break


def _iter_graph_chunks(
    node_ptr: h5py.Dataset,
    edge_ptr: h5py.Dataset,
    n_graphs: int,
    chunk_size: int,
) -> Iterator[tuple[int, int]]:
    """Yield ``(g_lo, g_hi)`` graph-index ranges covering ``range(n_graphs)``
    exactly once, greedily sized so each range's node span
    (``node_ptr[g_hi] - node_ptr[g_lo]``) and edge span
    (``edge_ptr[g_hi] - edge_ptr[g_lo]``) stay at most ``chunk_size``
    elements. A single graph whose own span already exceeds ``chunk_size`` is
    still yielded alone. A chunk always contains at least one graph.

    Streams ``node_ptr``/``edge_ptr`` in ``chunk_size``-sized reads and
    derives chunk boundaries incrementally, rather than loading either
    pointer array whole: at CODE-100 scale (~1e8 graphs) the full arrays
    would be ~800 MB each, which would blow the "few hundred MB" memory
    budget documented on :func:`validate_dataset`. This keeps peak memory
    bounded by ``chunk_size`` the same way every other check in this module
    is bounded.
    """
    if n_graphs == 0:
        return

    g_lo = 0
    node_lo = int(node_ptr[0])
    edge_lo = int(edge_ptr[0])
    for lo, hi in _iter_slices(n_graphs, chunk_size):
        node_block = node_ptr[lo : hi + 1]
        edge_block = edge_ptr[lo : hi + 1]
        for g in range(lo, hi):
            node_hi = int(node_block[g - lo + 1])
            edge_hi = int(edge_block[g - lo + 1])
            if g > g_lo and (node_hi - node_lo > chunk_size or edge_hi - edge_lo > chunk_size):
                yield g_lo, g
                g_lo = g
                node_lo = int(node_block[g - lo])
                edge_lo = int(edge_block[g - lo])
    yield g_lo, n_graphs


def _check_graph_chunk(
    issues: list[QAIssue],
    *,
    g_lo: int,
    g_hi: int,
    sampling_rate: int,
    edge_attr_names: list[str],
    edge_index_usable: bool,
    node_ptr: h5py.Dataset,
    edge_ptr: h5py.Dataset,
    node_t_ds: h5py.Dataset,
    node_x_ds: h5py.Dataset,
    node_orig_ds: h5py.Dataset,
    graph_start_ds: h5py.Dataset,
    graph_end_ds: h5py.Dataset,
    graph_rpeak_ds: h5py.Dataset,
    edge_index_ds: h5py.Dataset,
) -> None:
    """Vectorized deep checks for graphs ``[g_lo, g_hi)``, reading each
    dataset exactly once for the whole chunk (no per-graph HDF5 reads).
    Appends the same issue codes/messages ``_check_graphs_deep`` has always
    emitted, per graph.
    """
    node_ptr_chunk = np.asarray(node_ptr[g_lo : g_hi + 1], dtype=np.int64)
    edge_ptr_chunk = np.asarray(edge_ptr[g_lo : g_hi + 1], dtype=np.int64)
    n_lo, n_hi = int(node_ptr_chunk[0]), int(node_ptr_chunk[-1])
    e_lo, e_hi = int(edge_ptr_chunk[0]), int(edge_ptr_chunk[-1])
    local_node_ptr = node_ptr_chunk - n_lo
    local_edge_ptr = edge_ptr_chunk - e_lo
    node_counts = np.diff(local_node_ptr)
    edge_counts = np.diff(local_edge_ptr)
    n_g = g_hi - g_lo
    graph_ids = np.arange(g_lo, g_hi, dtype=np.int64)

    node_orig = np.asarray(node_orig_ds[n_lo:n_hi], dtype=np.int64)
    node_t = np.asarray(node_t_ds[n_lo:n_hi])
    node_x = np.asarray(node_x_ds[n_lo:n_hi])
    start = np.asarray(graph_start_ds[g_lo:g_hi], dtype=np.int64)
    end = np.asarray(graph_end_ds[g_lo:g_hi], dtype=np.int64)
    rpeak = np.asarray(graph_rpeak_ds[g_lo:g_hi], dtype=np.int64)

    graph_of_node = np.repeat(graph_ids, node_counts)

    start_per_node = np.repeat(start, node_counts)
    expected_t = (node_orig - start_per_node).astype(np.float64) / float(sampling_rate)
    actual_t = node_t.astype(np.float64)
    diff = np.abs(actual_t - expected_t)
    formula_bad_mask = diff > 1e-5
    if np.any(formula_bad_mask):
        for g in np.unique(graph_of_node[formula_bad_mask]):
            max_err = float(diff[graph_of_node == g].max())
            issues.append(
                QAIssue(
                    "error",
                    "node_t_formula",
                    f"graph {g}: node_t inconsistent with "
                    f"(node_original_sample - graph_start_sample) / sampling_rate "
                    f"(max abs error {max_err:.3e})",
                )
            )

    if actual_t.size > 1:
        d = np.diff(actual_t)
        boundary_positions = local_node_ptr[1:-1] - 1
        boundary_positions = boundary_positions[
            (boundary_positions >= 0) & (boundary_positions < d.size)
        ]
        valid_d_mask = np.ones(d.size, dtype=bool)
        valid_d_mask[boundary_positions] = False
        mono_bad_mask = (d < 0) & valid_d_mask
        if np.any(mono_bad_mask):
            for g in np.unique(graph_of_node[:-1][mono_bad_mask]):
                issues.append(
                    QAIssue("error", "node_t_non_monotonic", f"graph {g}: node_t not monotonic")
                )

    has_nodes_mask = node_counts > 0
    first_idx = local_node_ptr[:-1][has_nodes_mask]
    first_vals = actual_t[first_idx]
    first_bad_mask = np.abs(first_vals) > 1e-6
    if np.any(first_bad_mask):
        for g, val in zip(
            graph_ids[has_nodes_mask][first_bad_mask], first_vals[first_bad_mask], strict=True
        ):
            issues.append(
                QAIssue(
                    "error",
                    "node_t_first_nonzero",
                    f"graph {g}: node_t[0]={float(val)} != 0",
                )
            )

    valid_graph_bool = np.zeros(n_g, dtype=bool)
    if edge_index_usable and e_hi > e_lo:
        ei_chunk = np.asarray(edge_index_ds[:, e_lo:e_hi], dtype=np.int64)
        src_local, dst_local = ei_chunk[0], ei_chunk[1]
        graph_of_edge = np.repeat(graph_ids, edge_counts)
        local_graph_idx = graph_of_edge - g_lo
        num_local_nodes_per_edge = np.repeat(node_counts, edge_counts)

        oob_mask = (
            (src_local < 0)
            | (src_local >= num_local_nodes_per_edge)
            | (dst_local < 0)
            | (dst_local >= num_local_nodes_per_edge)
        )
        bad_bounds_per_graph = np.zeros(n_g, dtype=bool)
        np.logical_or.at(bad_bounds_per_graph, local_graph_idx, oob_mask)
        for g in graph_ids[bad_bounds_per_graph]:
            nln = int(node_counts[g - g_lo])
            issues.append(
                QAIssue(
                    "error",
                    "edge_index_local_bounds",
                    f"graph {g}: edge_index references node outside [0, {nln})",
                )
            )

        bad_src_dst_mask = (~oob_mask) & (src_local >= dst_local)
        bad_src_dst_per_graph = np.zeros(n_g, dtype=bool)
        np.logical_or.at(bad_src_dst_per_graph, local_graph_idx, bad_src_dst_mask)

        has_edges_per_graph = edge_counts > 0
        valid_graph_bool = has_edges_per_graph & ~bad_bounds_per_graph & ~bad_src_dst_per_graph

        if edge_attr_names and np.any(valid_graph_bool):
            include_mask = valid_graph_bool[local_graph_idx]
            node_offset_per_edge = np.repeat(local_node_ptr[:-1], edge_counts)
            rebased_src = src_local + node_offset_per_edge
            rebased_dst = dst_local + node_offset_per_edge
            inc_edge_index = np.vstack([rebased_src[include_mask], rebased_dst[include_mask]])
            inc_graph_of_edge = graph_of_edge[include_mask]

            derived = compute_edge_attrs(
                inc_edge_index, node_t, node_x, edge_attr_names, validate=False
            )
            non_finite_mask = ~np.all(np.isfinite(derived), axis=1)
            for g in np.unique(inc_graph_of_edge[non_finite_mask]):
                issues.append(
                    QAIssue(
                        "error",
                        "derived_edge_attr_non_finite",
                        f"graph {g}: derived edge_attr contains NaN/Inf",
                    )
                )
            if "delta_t" in edge_attr_names:
                col = edge_attr_names.index("delta_t")
                bad_delta_mask = derived[:, col] <= 0
                for g in np.unique(inc_graph_of_edge[bad_delta_mask]):
                    issues.append(
                        QAIssue(
                            "error",
                            "delta_t_non_positive",
                            f"graph {g}: derived delta_t has non-positive values",
                        )
                    )

    anchor_oob_mask = ~((start <= rpeak) & (rpeak <= end))
    if np.any(anchor_oob_mask):
        for g, r, s, e in zip(
            graph_ids[anchor_oob_mask],
            rpeak[anchor_oob_mask],
            start[anchor_oob_mask],
            end[anchor_oob_mask],
            strict=True,
        ):
            issues.append(
                QAIssue(
                    "error",
                    "graph_anchor_out_of_bounds",
                    f"graph {g}: r_peak_sample={int(r)} outside [{int(s)}, {int(e)}]",
                )
            )


def _check_graphs_deep(
    f: h5py.File,
    issues: list[QAIssue],
    n_graphs: int,
    chunk_size: int,
    node_ptr_ok: bool,
    edge_ptr_ok: bool,
) -> int:
    """Per-graph deep checks, chunked and vectorized over HDF5 reads. Every
    graph is validated, no sampling: ``node_t`` formula consistency with
    ``(node_original_sample - graph_start_sample) / sampling_rate``, ``node_t``
    monotonicity and first-zero, ``edge_index`` local-bound consistency,
    derived edge_attr finiteness and ``delta_t`` positivity, and the R-peak
    inside its beat span.

    Emits ``node_t_formula``, ``node_t_non_monotonic``, ``node_t_first_nonzero``,
    ``edge_index_local_bounds``, ``derived_edge_attr_non_finite``,
    ``delta_t_non_positive``, ``graph_anchor_out_of_bounds``.

    Graphs are processed in ``_iter_graph_chunks`` chunks: each chunk reads
    every dataset exactly once (whole-chunk slices), then derives per-graph
    results with vectorized numpy ops (``np.repeat``/``np.unique``/
    ``ufunc.at`` segment reductions) instead of a per-graph HDF5 read plus a
    per-graph ``compute_edge_attrs`` call. This is the same numeric contract
    as the original per-graph loop. See ``_check_graph_chunk``.

    ``node_ptr``/``edge_ptr`` must be correctly shaped; malformed shapes were
    already reported by the schema shape checks, so ``node_ptr_ok``/
    ``edge_ptr_ok`` gate the loop as a guard, NOT an early return. The
    caller still runs the file-level config_json check regardless of what
    this function returns.

    Returns the number of graphs deep-checked: 0 if pointers are malformed,
    ``n_graphs`` otherwise (for ``summary["num_graphs_deep_checked"]``).
    """
    if not (node_ptr_ok and edge_ptr_ok):
        return 0

    sampling_rate = int(f.attrs["sampling_rate"])
    edge_attr_names = [str(n) for n in f.attrs.get("edge_attr_names", [])]
    node_ptr = f["graph_node_ptr"]
    edge_ptr = f["graph_edge_ptr"]
    node_t_ds = f["node_t"]
    node_x_ds = f["node_x"]
    node_orig_ds = f["node_original_sample"]
    graph_start_ds = f["graph_start_sample"]
    graph_end_ds = f["graph_end_sample"]
    graph_rpeak_ds = f["graph_r_peak_sample"]
    edge_index_ds = f["edge_index"]
    edge_index_usable = edge_index_ds.shape[0] == 2

    for g_lo, g_hi in _iter_graph_chunks(node_ptr, edge_ptr, n_graphs, chunk_size):
        _check_graph_chunk(
            issues,
            g_lo=g_lo,
            g_hi=g_hi,
            sampling_rate=sampling_rate,
            edge_attr_names=edge_attr_names,
            edge_index_usable=edge_index_usable,
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            node_t_ds=node_t_ds,
            node_x_ds=node_x_ds,
            node_orig_ds=node_orig_ds,
            graph_start_ds=graph_start_ds,
            graph_end_ds=graph_end_ds,
            graph_rpeak_ds=graph_rpeak_ds,
            edge_index_ds=edge_index_ds,
        )

    return n_graphs


def _check_metadata_binding(
    f: h5py.File,
    h5_path: Path,
    issues: list[QAIssue],
) -> None:
    """Verify structure and duplicated root metadata against ``config_json``."""
    try:
        require_dataset_contract(f, h5_path)
    except ValueError as exc:
        issues.append(QAIssue("error", "contract_mismatch", str(exc)))


def validate_open_dataset_quick(
    f: h5py.File,
    h5_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> QAResult:
    """Run the quick schema checks through an already-open HDF5 handle."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    issues: list[QAIssue] = []
    summary: dict[str, Any] = {"path": str(h5_path), "profile": "quick"}
    if not _check_required_datasets(f, issues):
        return _result(issues, "quick", summary)
    _check_contract_attrs(f, issues)
    _check_dtypes(f, issues)
    n_exams, n_graphs, n_nodes, n_edges = _axis_lengths(f, issues)
    _check_shapes(f, issues, n_exams, n_graphs, n_nodes, n_edges)
    _check_graph_exam_ownership(f, issues, n_exams, chunk_size)
    _check_pointers(f, issues, n_graphs, n_nodes, n_edges, chunk_size)
    _check_metadata_binding(f, h5_path, issues)
    summary.update(
        num_exams=n_exams,
        num_graphs=n_graphs,
        num_nodes=n_nodes,
        num_edges=n_edges,
    )
    return _result(issues, "quick", summary)


def validate_dataset(
    h5_path: Path,
    *,
    profile: Literal["quick", "full"] = "full",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> QAResult:
    """Run validation. Returns QAResult; never raises for schema errors.

    ``chunk_size`` bounds the elements per HDF5 read in the streamed
    whole-array checks; the default keeps peak memory in the low hundreds
    of MB regardless of file size.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    issues: list[QAIssue] = []
    summary: dict[str, Any] = {"path": str(h5_path), "profile": profile}

    if not Path(h5_path).exists():
        return QAResult(
            success=False,
            profile=profile,
            issues=[QAIssue("error", "missing_file", f"HDF5 not found: {h5_path}")],
            summary=summary,
        )

    with h5py.File(h5_path, "r") as f:
        if not _check_required_datasets(f, issues):
            return QAResult(
                success=False,
                profile=profile,
                issues=issues,
                summary=summary,
            )

        _check_contract_attrs(f, issues)
        _check_dtypes(f, issues)

        n_exams, n_graphs, n_nodes, n_edges = _axis_lengths(f, issues)
        _check_shapes(f, issues, n_exams, n_graphs, n_nodes, n_edges)
        _check_graph_exam_ownership(f, issues, n_exams, chunk_size)
        node_ptr_ok, edge_ptr_ok = _check_pointers(
            f, issues, n_graphs, n_nodes, n_edges, chunk_size
        )
        _check_metadata_binding(f, h5_path, issues)

        summary["num_exams"] = n_exams
        summary["num_graphs"] = n_graphs
        summary["num_nodes"] = n_nodes
        summary["num_edges"] = n_edges

        if profile == "quick":
            return _result(issues, profile, summary)

        _check_edge_direction(f, issues, n_edges, chunk_size)
        _check_node_finiteness(f, issues, n_nodes, chunk_size)

        summary["num_graphs_deep_checked"] = _check_graphs_deep(
            f, issues, n_graphs, chunk_size, node_ptr_ok, edge_ptr_ok
        )

    return _result(issues, profile, summary)


def _result(issues: list[QAIssue], profile: str, summary: dict[str, Any]) -> QAResult:
    success = all(i.severity != "error" for i in issues)
    return QAResult(success=success, profile=profile, issues=issues, summary=summary)
