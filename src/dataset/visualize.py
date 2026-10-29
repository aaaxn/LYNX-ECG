"""Figure rendering for built dataset HDF5s.

All render functions are pure: they take an open ``h5py.File`` and return a
``matplotlib.figure.Figure``. The backend is forced to ``Agg`` at import time
so headless servers / pytest captured stdout don't crash on missing display.

Per-graph reads go through :class:`dataset.schema.GraphSlice` (the shared
CSR read pattern); this module only draws.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import h5py
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from dataset.schema import GraphSlice


def _exam_id_str(h5: h5py.File, exam_idx: int) -> str:
    return h5["exam_id"].asstr()[exam_idx]


def _plot_beat(ax, graph: GraphSlice, *, title: str | None = None) -> None:
    """Plot a single reduced beat: line + dots + R-peak marker."""
    t = graph.node_t
    amp = graph.node_x[:, 0]
    sampling_rate = float(graph.sampling_rate)
    rpeak_t = (graph.r_peak_sample - graph.start_sample) / sampling_rate

    ax.plot(t, amp, "-", color="#1f77b4", linewidth=1.0, alpha=0.6)
    ax.plot(t, amp, "o", color="#d62728", markersize=4)
    ax.axvline(
        rpeak_t,
        color="#2ca02c",
        linestyle="--",
        alpha=0.6,
        linewidth=0.8,
        label=f"R-peak @ {rpeak_t:.3f}s",
    )
    ax.set_xlabel("t within beat (s)")
    ax.set_ylabel("amplitude")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_visibility_graph(ax, graph: GraphSlice) -> None:
    """Draw the visibility graph directly on the signal (edges as chords).

    Uses a single ``LineCollection`` with per-edge color/alpha keyed to edge
    span (viridis): short chain edges stay faint and recede, long-range
    visibility bridges stand out. The signal trace + nodes + R-peak guide are
    overlaid on top.
    """
    import matplotlib.pyplot as plt

    t = np.asarray(graph.node_t, dtype=float)
    amp = np.asarray(graph.node_x[:, 0], dtype=float)
    edge_index = graph.edge_index
    n_edges = int(edge_index.shape[1])
    sampling_rate = float(graph.sampling_rate)
    rpeak_t = (graph.r_peak_sample - graph.start_sample) / sampling_rate

    if n_edges > 0:
        src = edge_index[0].astype(int)
        dst = edge_index[1].astype(int)
        segs = np.stack(
            [
                np.stack([t[src], amp[src]], axis=-1),
                np.stack([t[dst], amp[dst]], axis=-1),
            ],
            axis=1,
        )

        spans = t[dst] - t[src]
        max_span = float(spans.max()) if spans.size else 1.0
        norm_spans = spans / max_span if max_span > 0 else np.zeros_like(spans)
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm_spans)
        colors[:, 3] = 0.10 + 0.55 * norm_spans

        lc = LineCollection(segs, colors=colors, linewidths=0.6, zorder=2)
        ax.add_collection(lc)

    ax.plot(t, amp, "-", color="#1f77b4", linewidth=1.0, alpha=0.7, zorder=3)
    ax.plot(t, amp, "o", color="#d62728", markersize=3, zorder=4)
    ax.axvline(rpeak_t, color="#2ca02c", linestyle="--", alpha=0.6, linewidth=0.8, zorder=3)

    n_orig = int(graph.original_num_nodes)
    n_red = int(graph.reduced_num_nodes)
    ratio = (n_red / n_orig) if n_orig > 0 else 0.0
    ax.text(
        0.99,
        0.97,
        f"nodes: {n_red} kept / {n_orig} original ({ratio:.1%})\nedges: {n_edges}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    ax.set_xlim(t.min() - 0.01, t.max() + 0.01)
    ax.set_xlabel("t within beat (s)")
    ax.set_ylabel("amplitude")
    ax.set_title("Visibility graph on signal")
    ax.grid(True, alpha=0.3)


def render_beat_figure(
    h5: h5py.File,
    *,
    graph_idx: int,
    method: str | None = None,
) -> Figure:
    """Two panels: reduced beat + visibility graph + metadata strip."""
    import matplotlib.pyplot as plt

    graph = GraphSlice.from_hdf5(h5, graph_idx)
    stored_method = str(h5.attrs["node_reduction_method"])
    method = method or stored_method
    exam_id = _exam_id_str(h5, graph.exam_idx)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [1, 1]})

    title = (
        f"exam={exam_id} lead={graph.lead} beat={graph.beat_index} "
        f"method={method} K={graph.reduced_num_nodes}/{graph.original_num_nodes} "
        f"polarity={'+' if graph.rpeak_polarity > 0 else '-'}"
    )
    _plot_beat(axes[0], graph, title=title)
    _plot_visibility_graph(axes[1], graph)

    fig.tight_layout()
    return fig


def render_smoke_figure(h5: h5py.File) -> Figure:
    """Single-graph figure for build-time smoke check (uses graph_idx=0)."""
    if int(h5["graph_exam_index"].shape[0]) == 0:
        raise ValueError("dataset has zero graphs; smoke render impossible")
    return render_beat_figure(h5, graph_idx=0)
