"""``train`` CLI: build dataset, model, run train + evaluate.

Single-process:
    uv run train --config configs/train/code100_tp.json --seed 1111

DDP (auto-relaunches under ``torchrun`` for the user):
    uv run train --config configs/train/code100_tp.json --launcher ddp --gpus 2

When ``--launcher ddp`` is set and the process is NOT already under
``torchrun`` (no ``RANK`` / ``WORLD_SIZE`` in env), the CLI re-exec's itself
under ``torchrun --nproc_per_node=<gpus>``. Once re-entered, RANK is set and
the normal DDP path runs. ``--ddp-devices "0,1"`` sets ``CUDA_VISIBLE_DEVICES``
on the relaunched process tree.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from config.train_models import TrainPreset, load_train_preset
from training import ddp
from training.evaluate import evaluate_test, require_final_metrics
from training.hdf5_meta import assert_model_meta_compatible, read_hdf5_meta
from training.loaders import build_loaders, select_device
from training.run_summary import (
    build_run_summary,
    format_run_summary,
    write_preset_snapshot,
    write_run_summary,
)
from training.splits import resolve_splits
from training.train import train, write_history_json


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hierarchical ECG graph classifier.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Path to a JSON training preset.")
    src.add_argument("--preset", type=str, help="Name of a preset under configs/train/<name>.json.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Override the dataset HDF5 path (else read from preset).",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=Path,
        default=None,
        help="Override the test dataset HDF5 path (else read from preset).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override the root output directory (else read from preset).",
    )
    parser.add_argument(
        "--launcher",
        choices=("single", "ddp"),
        default=None,
        help="Override preset.ddp.launcher.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override preset.ddp.gpus (sanity check vs torchrun).",
    )
    parser.add_argument(
        "--ddp-devices",
        type=str,
        default=None,
        help="Comma-separated CUDA visible devices for DDP (informational; "
        "actual device assignment uses torchrun's LOCAL_RANK).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override preset.optim.seed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device selection (default 'auto' picks CUDA if available).",
    )
    return parser.parse_args(argv)


def _resolve_preset_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return Path(args.config)
    return Path("configs/train") / f"{args.preset}.json"


def _apply_overrides(preset: TrainPreset, args: argparse.Namespace) -> TrainPreset:
    payload = preset.model_dump()
    if args.dataset_path is not None:
        payload["dataset_path"] = str(args.dataset_path)
    if args.test_dataset_path is not None:
        payload["test_dataset_path"] = str(args.test_dataset_path)
    if args.results_dir is not None:
        payload["results_dir"] = str(args.results_dir)
    if args.launcher is not None:
        payload["ddp"]["launcher"] = args.launcher
    if args.gpus is not None:
        payload["ddp"]["gpus"] = int(args.gpus)
    if args.seed is not None:
        payload["optim"]["seed"] = int(args.seed)
    return TrainPreset.model_validate(payload)


def _relaunch_under_torchrun(
    args: argparse.Namespace, preset: TrainPreset, argv: Sequence[str] | None
) -> None:
    """Replace this process with ``torchrun --nproc_per_node=<gpus> -m cli.train <argv>``.

    Forwards every CLI argument the user originally passed (so the second pass
    of ``main`` sees the same flags). Sets ``CUDA_VISIBLE_DEVICES`` from
    ``--ddp-devices`` when provided. Does not return, since ``execvpe`` replaces the
    process image; if the exec itself fails (e.g. ``torchrun`` not on PATH),
    raises ``FileNotFoundError``.
    """
    forwarded = list(argv) if argv is not None else sys.argv[1:]
    cmd = [
        "torchrun",
        f"--nproc_per_node={int(preset.ddp.gpus)}",
        "-m",
        "cli.train",
        *forwarded,
    ]
    env = os.environ.copy()
    if args.ddp_devices is not None:
        devices = [device.strip() for device in args.ddp_devices.split(",")]
        if any(not device for device in devices):
            raise ValueError(
                f"--ddp-devices contains an empty device id: {args.ddp_devices!r}. "
                "Pass a comma-separated list with no empty entries."
            )
        if len(set(devices)) != len(devices):
            raise ValueError(
                f"--ddp-devices contains duplicate device ids: {args.ddp_devices!r}. "
                "Each torchrun rank requires a distinct visible GPU."
            )
        if len(devices) != int(preset.ddp.gpus):
            raise ValueError(
                f"--ddp-devices lists {len(devices)} device id(s) "
                f"({args.ddp_devices!r}) but preset.ddp.gpus={int(preset.ddp.gpus)}. "
                f"torchrun would spawn {int(preset.ddp.gpus)} ranks over "
                f"{len(devices)} visible GPU(s), causing rank/device collisions or "
                f"idle GPUs. Pass exactly {int(preset.ddp.gpus)} comma-separated "
                f"device ids, or align --gpus / ddp.gpus with the device list."
            )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    print(f"[train] re-launching under torchrun: {' '.join(cmd)}", file=sys.stderr)
    os.execvpe(cmd[0], cmd, env)


def _assert_world_size_matches(preset: TrainPreset) -> None:
    """Fail fast when torchrun's ``WORLD_SIZE`` disagrees with ``preset.ddp.gpus``.

    Called on every rank once the process is running under torchrun, before any
    dataset/model setup and before the process group is created. ``WORLD_SIZE``
    is read straight from the environment (the value torchrun sets and
    :func:`training.ddp.init_distributed` keys on), so the check needs no live
    process group. A mismatch means the run would shard loaders and record
    ``run_summary`` across a different rank count than the preset assumes.
    Single-node is the only supported topology, so ``WORLD_SIZE`` must equal
    ``preset.ddp.gpus`` exactly.
    """
    actual = int(os.environ["WORLD_SIZE"])
    expected = int(preset.ddp.gpus)
    if actual != expected:
        raise ValueError(
            f"DDP world-size mismatch: torchrun set WORLD_SIZE={actual} but "
            f"preset.ddp.gpus={expected}. On a single node these MUST be equal. "
            f"Relaunch with `torchrun --nproc_per_node={expected} -m cli.train ...` "
            f"(or drop the manual torchrun and let `--launcher ddp` self-relaunch), "
            f"or set ddp.gpus={actual} in the preset (equivalently pass --gpus {actual}). "
            f"Refusing to start: loaders would shard across {actual} ranks while the "
            f"run config and summary claim {expected}. Multi-node / elastic runs "
            f"(WORLD_SIZE != per-node gpus) are not supported."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    preset_path = _resolve_preset_path(args)
    preset = _apply_overrides(load_train_preset(preset_path), args)

    already_under_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if preset.ddp.launcher == "ddp" and not already_under_torchrun:
        _relaunch_under_torchrun(args, preset, argv)
        return 2  # pragma: no cover

    if preset.ddp.launcher == "ddp" and already_under_torchrun:
        _assert_world_size_matches(preset)

    distributed = ddp.init_distributed() if preset.ddp.launcher == "ddp" else False

    device = select_device(args.device)

    meta = read_hdf5_meta(Path(preset.dataset_path))
    assert_model_meta_compatible(preset.model, meta)
    num_classes = meta.num_classes
    num_leads = meta.num_leads
    class_names = meta.class_names

    splits = resolve_splits(preset, meta)
    train_idx, val_idx, test_idx = splits.train_idx, splits.val_idx, splits.test_idx
    test_meta = splits.test_meta
    test_dataset_path = Path(preset.test_dataset_path) if preset.test_dataset_path else None

    train_loader, val_loader, test_loader = build_loaders(
        dataset_path=Path(preset.dataset_path),
        num_leads=num_leads,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        exam_labels=meta.exam_labels,
        sampling=preset.sampling,
        batch_size=preset.optim.batch_size,
        num_workers=preset.optim.num_workers,
        distributed=distributed,
        seed=preset.optim.seed,
        device=device,
        test_dataset_path=test_dataset_path,
        shuffle_block_exams=preset.optim.shuffle_block_exams,
        eval_batch_size=preset.optim.eval_batch_size,
        symmetrize_edges=preset.symmetrize_edges,
        subgraph_representation=preset.subgraph_representation,
    )

    torch.manual_seed(preset.optim.seed)
    base_model = preset.model.build(num_classes=num_classes, num_leads=num_leads).to(device)
    model = ddp.wrap_and_compile_model(
        base_model,
        use_compile=preset.optim.compile,
    )

    run_dir = Path(preset.results_dir) / f"{preset.run_name}_s{preset.optim.seed}"
    if ddp.is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)

    if ddp.is_main_process():
        train_ds = train_loader.dataset
        graphs_per_exam = (train_ds.exam_to_graph_end - train_ds.exam_to_graph_start).astype(
            np.int64
        )
        edges_per_exam = train_ds.effective_edges_per_exam()
        summary_test_idx = np.empty(0, dtype=np.int64) if test_meta is not None else test_idx
        summary = build_run_summary(
            preset=preset,
            dataset_path=preset.dataset_path,
            sampling_rate=meta.sampling_rate,
            total_exams=meta.total_exams,
            total_graphs=meta.total_graphs,
            num_leads=num_leads,
            class_names=class_names,
            exam_labels=meta.exam_labels,
            graphs_per_exam=graphs_per_exam,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=summary_test_idx,
            device=str(device),
            world_size=ddp.world_size(),
            node_reduction=meta.node_reduction,
            edges_per_exam=edges_per_exam,
            model_parameter_count=sum(parameter.numel() for parameter in base_model.parameters()),
            trainable_parameter_count=sum(
                parameter.numel()
                for parameter in base_model.parameters()
                if parameter.requires_grad
            ),
        )
        if test_meta is not None:
            test_ds = test_loader.dataset
            test_graphs = (test_ds.exam_to_graph_end - test_ds.exam_to_graph_start).astype(np.int64)
            summary["splits"]["test"] = {
                "exams": test_meta.total_exams,
                "graphs": int(test_graphs.sum()),
                "subgraph_edges": int(test_ds.effective_edges_per_exam().sum()),
            }
            summary["test_dataset"] = {
                "path": str(preset.test_dataset_path),
                "exams": test_meta.total_exams,
                "beat_graphs": test_meta.total_graphs,
            }
            test_labels = test_meta.exam_labels
            test_total = test_meta.total_exams
            for cls_idx, cls_name in enumerate(class_names):
                count = int(test_labels[:, cls_idx].sum())
                summary["prevalence"][cls_name]["test"] = {
                    "count": count,
                    "pct": float(count) / test_total * 100.0 if test_total > 0 else 0.0,
                }
        write_run_summary(summary, run_dir / "run_summary.json")
        write_preset_snapshot(preset, run_dir / "preset.json")
        print(format_run_summary(summary), flush=True)

    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optim=preset.optim,
        threshold=preset.threshold,
        class_names=class_names,
        dataset_fingerprint=meta.fingerprint,
        device=device,
        ckpt_dir=run_dir,
        symmetrize_edges=preset.symmetrize_edges,
        subgraph_representation=preset.subgraph_representation,
    )

    if ddp.is_main_process():
        write_history_json(result.history, run_dir / "history.json")

    ddp.barrier()

    ckpt_path = run_dir / "best_f1_val.pt"
    if not ckpt_path.exists():
        if ddp.is_main_process():
            print(
                f"[train] WARN: {ckpt_path} not produced (val f1/macro never finite-improved); "
                "skipping final test evaluation.",
                file=sys.stderr,
            )
        ddp.shutdown_distributed()
        return 1

    test_metrics = evaluate_test(
        model=model,
        test_loader=test_loader,
        ckpt_path=ckpt_path,
        class_names=class_names,
        device=device,
        threshold=preset.threshold.value,
        out_json=run_dir / "test_metrics.json",
    )
    test_metrics = require_final_metrics(test_metrics, label="test metrics")
    if ddp.is_main_process():
        print(
            f"[train] best val f1/macro={result.best_f1_val:.4f}, "
            f"best val pr_auc/macro={result.best_pr_auc_val:.4f}; "
            f"test f1/macro={test_metrics['f1/macro']:.4f}, "
            f"test pr_auc/macro={test_metrics['pr_auc/macro']:.4f}, "
            f"test auroc/macro={test_metrics['auroc/macro']:.4f}"
        )

    ddp.shutdown_distributed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
