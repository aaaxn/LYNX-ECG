"""``evaluate`` CLI: score an existing checkpoint on an HDF5 dataset.

Exit codes:
- ``0``: evaluation completed successfully.
- ``2``: execution / usage error (missing file, unreadable checkpoint,
  dataset-checkpoint incompatibility).

``1`` is deliberately not used so that the CLI exit code is unambiguous:
``2`` always means "something went wrong before inference", never a
classifier-performance outcome.

Usage examples::

    # Threshold 0.5 for every class (default):
    uv run evaluate --ckpt results/my_run/best_f1_val.pt --dataset path/to/data.h5

    # Write metrics to JSON:
    uv run evaluate --ckpt ... --dataset ... --output metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from torch_geometric.loader import DataLoader as PyGDataLoader

from training.checkpoint import (
    reconstruct_model_from_ckpt,
    resolve_subgraph_representation,
    resolve_symmetrize_edges,
)
from training.dataset import ExamGraphDataset
from training.evaluate import evaluate_loader, require_checkpoint_epoch, require_final_metrics
from training.hdf5_meta import require_compatible_dataset
from training.loaders import select_device


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description=(
            "Score an existing HierGNNModel checkpoint against an HDF5 dataset. "
            "Outputs per-class F1 / AUROC / Brier / ECE and full confusion counts, "
            "with an optional JSON report via --output."
        ),
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Path to the checkpoint (.pt) to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the HDF5 dataset to score against.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold applied to every class (default 0.5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="DataLoader batch size (default 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default 4; use 0 for in-process loading).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Compute device ('auto' selects CUDA when available, default 'auto').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``evaluate`` CLI.

    Composes :func:`reconstruct_model_from_ckpt` and :func:`evaluate_loader`
    in a single-process forward pass, with no DDP init required.

    Returns 0 on success, 2 on any execution or compatibility error.
    """
    args = _parse_args(argv)
    console = Console()

    device = select_device(args.device)

    try:
        model, state = reconstruct_model_from_ckpt(args.ckpt, device=device)
    except FileNotFoundError:
        console.print(f"[red]evaluate error:[/red] checkpoint not found: {args.ckpt}")
        return 2
    except ValueError as exc:
        console.print(f"[red]evaluate error:[/red] {exc}")
        return 2
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]evaluate error:[/red] unexpected error loading checkpoint: {exc}")
        return 2

    try:
        checkpoint_epoch = require_checkpoint_epoch(state, label=f"checkpoint {args.ckpt}")
    except ValueError as exc:
        console.print(f"[red]evaluate error:[/red] {exc}")
        return 2

    try:
        meta = require_compatible_dataset(state, args.dataset, allow_symmetrized=True)
    except FileNotFoundError:
        console.print(f"[red]evaluate error:[/red] dataset not found: {args.dataset}")
        return 2
    except ValueError as exc:
        console.print(f"[red]evaluate error:[/red] {exc}")
        return 2
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]evaluate error:[/red] unexpected error reading dataset: {exc}")
        return 2

    symmetrize_edges = resolve_symmetrize_edges(state)
    try:
        representation = resolve_subgraph_representation(state)
        ds = ExamGraphDataset(
            args.dataset,
            np.arange(meta.total_exams),
            num_leads=model.num_leads,
            symmetrize_edges=symmetrize_edges,
            subgraph_representation=representation,
        )
    except ValueError as exc:
        console.print(f"[red]evaluate error:[/red] {exc}")
        return 2
    loader = PyGDataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    try:
        eval_report = evaluate_loader(
            model,
            loader,
            device,
            class_names=meta.class_names,
            threshold=args.threshold,
            desc="evaluate",
        )
        metrics = require_final_metrics(eval_report.metrics)
    except ValueError as exc:
        console.print(f"[red]evaluate error:[/red] {exc}")
        return 2
    finally:
        ds.close()

    epoch_str = str(checkpoint_epoch)
    console.print(
        f"\n[bold]Checkpoint:[/bold] {args.ckpt}  "
        f"[bold]epoch:[/bold] {epoch_str}  "
        f"[bold]avg_loss:[/bold] {eval_report.avg_loss:.4f}  "
        f"[bold]threshold:[/bold] {args.threshold}"
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Class", style="dim")
    table.add_column("F1", justify="right")
    table.add_column("AUROC", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("TN", justify="right")

    def _fmt(v: float) -> str:
        return "NaN" if not np.isfinite(v) else f"{v:.4f}"

    for cls_name in meta.class_names:
        table.add_row(
            cls_name,
            _fmt(metrics[f"f1/{cls_name}"]),
            _fmt(metrics[f"auroc/{cls_name}"]),
            _fmt(metrics[f"precision/{cls_name}"]),
            _fmt(metrics[f"recall/{cls_name}"]),
            _fmt(metrics[f"brier/{cls_name}"]),
            _fmt(metrics[f"ece/{cls_name}"]),
            str(int(metrics[f"confusion/{cls_name}/tp"])),
            str(int(metrics[f"confusion/{cls_name}/fp"])),
            str(int(metrics[f"confusion/{cls_name}/fn"])),
            str(int(metrics[f"confusion/{cls_name}/tn"])),
        )

    table.add_section()
    table.add_row(
        "[bold]macro[/bold]",
        _fmt(metrics["f1/macro"]),
        _fmt(metrics["auroc/macro"]),
        _fmt(metrics["precision/macro"]),
        _fmt(metrics["recall/macro"]),
        _fmt(metrics["brier/macro"]),
        _fmt(metrics["ece/macro"]),
        "n/a",
        "n/a",
        "n/a",
        "n/a",
    )
    console.print(table)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "ckpt_path": str(args.ckpt),
            "dataset": str(args.dataset),
            "threshold": args.threshold,
            "epoch_at_save": checkpoint_epoch,
            "metrics": metrics,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        console.print(f"[green]✓[/green] report written to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
