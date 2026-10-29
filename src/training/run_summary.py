"""Run-start summary: print a banner and persist ``run_summary.json`` + ``preset.json``.

Pure functions so they can be unit-tested without touching HDF5 / DDP:

- :func:`build_run_summary` takes pre-loaded arrays + the resolved preset and
  returns a plain ``dict``.
- :func:`format_run_summary` renders the dict as a human-readable banner.
- :func:`write_run_summary` and :func:`write_preset_snapshot` are the IO sinks
  (rank 0 only, the caller's responsibility).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from config.train_models import TrainPreset

BANNER_WIDTH = 100
SECTION_LABEL_WIDTH = 10


def _normalize_class_name(name: Any) -> str:
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _pos_weight(pos: int, total: int) -> float | None:
    if pos <= 0:
        return None
    return float(total - pos) / float(pos)


def _split_prevalence(
    exam_labels: np.ndarray,
    indices: np.ndarray,
) -> list[dict[str, float | int]]:
    """Per-class positive count + pct for one split."""
    if indices.size == 0:
        return [{"count": 0, "pct": 0.0} for _ in range(exam_labels.shape[1])]
    sub = exam_labels[indices]
    total = int(indices.size)
    counts = sub.sum(axis=0).astype(np.int64)
    return [
        {"count": int(c), "pct": float(c) / total * 100.0 if total > 0 else 0.0} for c in counts
    ]


def build_run_summary(
    *,
    preset: TrainPreset,
    dataset_path: str | Path,
    sampling_rate: int,
    total_exams: int,
    total_graphs: int,
    num_leads: int,
    class_names: Sequence[Any],
    exam_labels: np.ndarray,
    graphs_per_exam: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str,
    world_size: int,
    node_reduction: dict[str, Any],
    edges_per_exam: np.ndarray | None = None,
    model_parameter_count: int | None = None,
    trainable_parameter_count: int | None = None,
) -> dict[str, Any]:
    """Compute the run-summary dict.

    All inputs are already loaded; this function does not touch IO. The shape
    of the output is what gets written to ``run_summary.json``.
    """
    names = [_normalize_class_name(n) for n in class_names]
    num_classes = len(names)
    if exam_labels.shape[1] != num_classes:
        raise ValueError(
            f"exam_labels.shape[1]={exam_labels.shape[1]} != len(class_names)={num_classes}"
        )

    if edges_per_exam is not None and edges_per_exam.shape != graphs_per_exam.shape:
        raise ValueError("edges_per_exam and graphs_per_exam must have the same per-exam shape")
    splits: dict[str, dict[str, int]] = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        splits[name] = {
            "exams": int(idx.size),
            "graphs": int(graphs_per_exam[idx].sum()) if idx.size else 0,
        }
        if edges_per_exam is not None:
            splits[name]["subgraph_edges"] = int(edges_per_exam[idx].sum()) if idx.size else 0

    train_prev = _split_prevalence(exam_labels, train_idx)
    val_prev = _split_prevalence(exam_labels, val_idx)
    test_prev = _split_prevalence(exam_labels, test_idx)

    prevalence: dict[str, dict[str, Any]] = {}
    for cls_idx, cls_name in enumerate(names):
        train_count = int(train_prev[cls_idx]["count"])
        prevalence[cls_name] = {
            "train": train_prev[cls_idx],
            "val": val_prev[cls_idx],
            "test": test_prev[cls_idx],
            "pos_weight_train": _pos_weight(train_count, int(train_idx.size)),
        }

    model_summary = preset.model.model_dump()
    if model_parameter_count is not None or trainable_parameter_count is not None:
        if model_parameter_count is None or trainable_parameter_count is None:
            raise ValueError("model and trainable parameter counts must be provided together")
        if not 0 <= trainable_parameter_count <= model_parameter_count:
            raise ValueError(
                "parameter counts must satisfy 0 <= trainable <= total, got "
                f"trainable={trainable_parameter_count}, total={model_parameter_count}"
            )
        model_summary["parameter_count"] = {
            "total": int(model_parameter_count),
            "trainable": int(trainable_parameter_count),
        }

    summary: dict[str, Any] = {
        "dataset": {
            "path": str(dataset_path),
            "exams": int(total_exams),
            "beat_graphs": int(total_graphs),
            "sampling_rate_hz": int(sampling_rate),
            "leads": int(num_leads),
            "node_reduction": dict(node_reduction),
        },
        "subgraph_representation": {
            **preset.subgraph_representation.model_dump(),
            "symmetrize_edges": bool(preset.symmetrize_edges),
        },
        "splits": splits,
        "classes": {"count": num_classes, "names": names},
        "prevalence": prevalence,
        "model": model_summary,
        "runtime": {
            "device": str(device),
            "world_size": int(world_size),
            "batch_size": int(preset.optim.batch_size),
            "num_workers": int(preset.optim.num_workers),
            "seed": int(preset.optim.seed),
        },
    }

    is_pos = exam_labels[train_idx].sum(axis=1) > 0
    n_pos = int(is_pos.sum())
    n_neg_epoch = int(preset.sampling.n_neg)
    summary["sampling"] = {
        "pool": {"n_pos": n_pos, "n_neg": int((~is_pos).sum())},
        "per_epoch": {
            "n_pos": n_pos,
            "n_neg": n_neg_epoch,
            "total": n_pos + n_neg_epoch,
        },
    }

    return summary


def _fmt_pos_weight(pw: float | None) -> str:
    if pw is None:
        return "inf"
    return f"{pw:.2f}"


def _fmt_count_pct(cell: dict[str, float | int]) -> str:
    return f"{int(cell['count']):>5d} ({float(cell['pct']):4.1f}%)"


def format_run_summary(summary: dict[str, Any]) -> str:
    """Render the banner string."""
    lines: list[str] = []
    rule = "═" * BANNER_WIDTH
    label = " run summary "
    pad = (BANNER_WIDTH - len(label)) // 2
    header = "═" * pad + label + "═" * (BANNER_WIDTH - pad - len(label))
    lines.append(header)

    def section(name: str, body: list[str]) -> None:
        prefix = f"{name:<{SECTION_LABEL_WIDTH}}"
        cont = " " * SECTION_LABEL_WIDTH
        for i, line in enumerate(body):
            lines.append((prefix if i == 0 else cont) + line)

    ds = summary["dataset"]
    section(
        "dataset",
        [
            ds["path"],
            (
                f"exams={ds['exams']}  beat-graphs={ds['beat_graphs']}  "
                f"sampling_rate={ds['sampling_rate_hz']}Hz  leads={ds['leads']}"
            ),
        ],
    )
    lines.append("")
    section(
        "provenance",
        [f"node_reduction={json.dumps(ds['node_reduction'], sort_keys=True)}"],
    )
    lines.append("")

    representation = summary["subgraph_representation"]
    section(
        "subgraph",
        [
            (
                f"adjacency={representation['adjacency']}  "
                f"edge_attributes={representation['edge_attributes']}  "
                f"symmetrize_edges={representation['symmetrize_edges']}"
            )
        ],
    )
    lines.append("")

    sp = summary["splits"]
    total_exams = sp["train"]["exams"] + sp["val"]["exams"] + sp["test"]["exams"]

    def pct(n: int) -> str:
        return f"{(n / total_exams * 100.0) if total_exams else 0.0:4.1f}%"

    split_lines = []
    for split_name in ("train", "val", "test"):
        cell = sp[split_name]
        edge_suffix = (
            f"  subgraph-edges={cell['subgraph_edges']}" if "subgraph_edges" in cell else ""
        )
        split_lines.append(
            f"{split_name:<5}={cell['exams']:>5d} ({pct(cell['exams'])})  "
            f"graphs={cell['graphs']}{edge_suffix}"
        )
    section("splits", split_lines)
    lines.append("")

    cls = summary["classes"]
    section("classes", [f"{cls['count']}: {cls['names']}"])
    lines.append("")

    name_w = max((len(n) for n in cls["names"]), default=8)
    name_w = max(name_w, 8)
    cell_w = 14
    header_row = (
        f"  {'class':<{name_w}}  "
        f"{'train':>{cell_w}}  {'val':>{cell_w}}  {'test':>{cell_w}}  pos_weight(train)"
    )
    section("prevalence", [header_row])
    for name in cls["names"]:
        row = summary["prevalence"][name]
        lines.append(
            " " * SECTION_LABEL_WIDTH
            + f"  {name:<{name_w}}  "
            + f"{_fmt_count_pct(row['train']):>{cell_w}}  "
            + f"{_fmt_count_pct(row['val']):>{cell_w}}  "
            + f"{_fmt_count_pct(row['test']):>{cell_w}}  "
            + f"{_fmt_pos_weight(row['pos_weight_train']):>10}"
        )
    lines.append("")

    m = summary["model"]
    section(
        "model",
        [
            (
                f"HierGNNModel | hidden={m['hidden_channels']}  "
                f"beat_encoder_layers={m['beat_encoder']['layers']}  "
                f"gatv2_heads={m['gatv2_heads']}  gatv2_layers={m['gatv2_layers']}"
                + (f"  params={m['parameter_count']['total']:,}" if "parameter_count" in m else "")
            ),
        ],
    )
    lines.append("")

    pool = summary["sampling"]["pool"]
    epoch = summary["sampling"]["per_epoch"]
    section(
        "sampling",
        [
            f"pool: n_pos={pool['n_pos']}  n_neg={pool['n_neg']}",
            (f"per_epoch: n_pos={epoch['n_pos']}  n_neg={epoch['n_neg']}  total={epoch['total']}"),
        ],
    )
    lines.append("")

    rt = summary["runtime"]
    section(
        "runtime",
        [
            (
                f"device={rt['device']}  world_size={rt['world_size']}  "
                f"batch_size={rt['batch_size']}  num_workers={rt['num_workers']}  seed={rt['seed']}"
            ),
        ],
    )
    lines.append(rule)
    return "\n".join(lines)


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so JSON stays strict."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_run_summary(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_json_safe(summary), f, indent=2)


def write_preset_snapshot(preset: TrainPreset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(preset.model_dump(), f, indent=2)
