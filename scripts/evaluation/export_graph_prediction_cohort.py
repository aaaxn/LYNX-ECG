"""Score checkpoints and export an exact multi-seed paired prediction cohort."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from cli.run_arguments import seed_path_type
from ecg_paired_predictions import write_paired_prediction_cohort
from training.checkpoint import resolve_symmetrize_edges
from training.inference import load_checkpoint_and_predict, resolve_device


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one exact evaluation cohort through multiple checkpoints and "
            "publish aligned probabilities with checkpoint provenance."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Evaluation graph HDF5.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Cohort identity shared with the baseline export (for example CODE-Test).",
    )
    parser.add_argument("--model-name", default="Ours")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=seed_path_type("SEED=CHECKPOINT"),
        metavar="SEED=PATH",
        help="Checkpoint seed binding; repeat once per reported run.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-symmetrized-checkpoints",
        action="store_true",
        help=(
            "Explicitly allow checkpoints trained with symmetrize_edges=True. "
            "Inference reconstructs each dataset using the checkpoint's persisted setting."
        ),
    )
    return parser.parse_args()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(values)
    positive = values >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    return probabilities


def main() -> None:
    args = _parse_args()
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")
    if not args.model_name:
        raise ValueError("--model-name must be non-empty")

    checkpoints = tuple(args.checkpoint)
    seeds = tuple(run.seed for run in checkpoints)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--checkpoint contains duplicate seeds: {seeds}")

    with h5py.File(args.dataset, "r") as h5:
        ids = h5["exam_id"].asstr()[:]

    device = resolve_device(args.device)
    probabilities_by_seed: list[np.ndarray] = []
    sources: list[dict[str, object]] = []
    labels: tuple[str, ...] | None = None
    y_true: np.ndarray | None = None
    for run in checkpoints:
        seed, checkpoint = run.seed, run.path
        result = load_checkpoint_and_predict(
            checkpoint,
            args.dataset,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc=f"{args.model_name} seed {seed}",
            allow_symmetrized=args.allow_symmetrized_checkpoints,
        )
        seed_labels = tuple(result.meta.class_names)
        seed_truth = np.asarray(result.predictions.labels)
        seed_probabilities = _sigmoid(result.predictions.logits)
        expected_shape = (len(ids), len(seed_labels))
        if seed_truth.shape != expected_shape:
            raise ValueError(
                f"seed {seed}: gathered labels must have shape {expected_shape}, "
                f"got {seed_truth.shape}"
            )
        if seed_probabilities.shape != expected_shape:
            raise ValueError(
                f"seed {seed}: probabilities must have shape {expected_shape}, "
                f"got {seed_probabilities.shape}"
            )
        if labels is None:
            labels = seed_labels
            y_true = seed_truth.astype(np.uint8, copy=False)
        else:
            if seed_labels != labels:
                raise ValueError(f"seed {seed}: class order differs from the first checkpoint")
            if not np.array_equal(seed_truth, y_true):
                raise ValueError(f"seed {seed}: target matrix differs from the first checkpoint")

        probabilities_by_seed.append(seed_probabilities)
        sources.append(
            {
                "model": args.model_name,
                "seed": seed,
                "kind": "graph_checkpoint_prediction",
                "provenance": {
                    "checkpoint": {
                        "path": str(checkpoint.resolve()),
                        "sha256": result.checkpoint_sha256,
                        "epoch": int(result.state["epoch"]),
                    },
                    "dataset": {
                        "path": str(args.dataset.resolve()),
                        "fingerprint": asdict(result.meta.fingerprint),
                    },
                    "symmetrize_edges": resolve_symmetrize_edges(result.state),
                },
            }
        )

    assert labels is not None
    assert y_true is not None
    cohort = write_paired_prediction_cohort(
        args.output,
        dataset=args.dataset_name,
        ids=ids,
        labels=labels,
        y_true=y_true,
        models=(args.model_name,),
        seeds=seeds,
        y_prob=np.stack(probabilities_by_seed, axis=0)[None, ...],
        sources=sources,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dataset": cohort.dataset,
                "model": args.model_name,
                "seeds": list(cohort.seeds),
                "exams": len(cohort.ids),
                "device": str(device),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
