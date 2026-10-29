"""``dataset`` CLI: build an HDF5 satisfying the current dataset contract.

Defaults: ``--jobs 16``; writer in main thread. Results are written as each
bounded chunk finishes so processed graph arrays are not globally buffered.

Throughput layout: the loader yields raw native-rate samples and runs on a
prefetch thread (next chunk loads while the current one is processed); the
config-driven band-pass / resample transforms execute inside the parallel
workers as the first step of ``process_exam``, keeping the serial loader
path to pure source I/O, which is the build's wall-clock cap at high job
counts.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from collections.abc import Callable, Sequence, Sized
from contextlib import nullcontext
from itertools import batched, chain
from pathlib import Path
from typing import cast

import joblib
from joblib import Parallel, delayed
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config.models import DatasetConfig, load_config
from dataset import __version__
from dataset.builder import process_exam
from dataset.filename import (
    derive_dataset_filename,
    derive_output_path,
    derive_reports_dir,
)
from dataset.hdf5_writer import DatasetWriter
from dataset.labels import (
    LabelContractError,
    validate_label_definition,
    validate_label_vector,
)
from dataset.loaders import get_loader
from dataset.prefetch import prefetch_iter
from dataset.qa import validate_dataset
from dataset.report import BuildReport
from ecg_artifact_io import staged_artifact


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dataset",
        description="Build a LYNX-ECG HDF5 satisfying the current dataset contract.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a DatasetConfig JSON.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel workers (-1 = all cores). Default: config.build.jobs (16).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Exams per load/process/write chunk. Default: config.build.chunk_size (4096).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing HDF5 output.")
    parser.add_argument(
        "--output", type=Path, default=None, help="Override the derived output filename."
    )
    parser.add_argument(
        "--limit-exams",
        type=int,
        default=None,
        help="Override config.limit_exams; appears in filename token.",
    )
    parser.add_argument(
        "--no-sort-output",
        action="store_true",
        help="Do not sort exams by exam_id within each processing chunk.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root for derived output / reports paths (default: current working directory).",
    )
    return parser.parse_args(argv)


def _apply_overrides(config: DatasetConfig, args: argparse.Namespace) -> DatasetConfig:
    """Apply CLI overrides on top of the config's defaults."""
    updates: dict = {}
    if args.limit_exams is not None:
        updates["limit_exams"] = int(args.limit_exams)
    if args.output is not None:
        updates["output_path"] = str(args.output)
    if updates:
        config = config.model_copy(update=updates)

    build_updates: dict = {}
    if args.jobs is not None:
        build_updates["jobs"] = int(args.jobs)
    if args.chunk_size is not None:
        build_updates["chunk_size"] = int(args.chunk_size)
    if args.overwrite:
        build_updates["overwrite"] = True
    if args.no_sort_output:
        build_updates["sort_output"] = False
    if build_updates:
        new_build = config.build.model_copy(update=build_updates)
        config = config.model_copy(update={"build": new_build})

    return DatasetConfig.model_validate(config.model_dump())


@contextlib.contextmanager
def _joblib_progress(advance: Callable[[int], None] | None):
    """Patch joblib so each completed task advances the active progress display."""

    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            if advance is not None:
                advance(self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old


def _process_with_jobs(
    samples,
    config: DatasetConfig,
    *,
    advance: Callable[[int], None] | None = None,
):
    """Yield process_exam results over samples, sequential or parallel as configured."""
    jobs = int(config.build.jobs)
    if jobs == 1:
        for sample in samples:
            yield process_exam(sample, config)
            if advance is not None:
                advance(1)
        return

    pre_dispatch = "2*n_jobs" if jobs == -1 else max(1, 2 * jobs)
    with _joblib_progress(advance):
        yield from Parallel(
            n_jobs=jobs,
            backend="loky",
            return_as="generator",
            pre_dispatch=pre_dispatch,
        )(delayed(process_exam)(s, config) for s in samples)


def _iter_loader_batches(loader, batch_size: int):
    iter_batches = getattr(loader, "iter_batches", None)
    if callable(iter_batches):
        yield from iter_batches(batch_size=batch_size)
        return

    yield from (list(b) for b in batched(loader, batch_size))


def _validate_label_chunk(
    samples: Sequence,
    *,
    expected_names: tuple[str, ...] | None = None,
    expected_taxonomy: str | None = None,
    chunk_context: str,
) -> tuple[tuple[str, ...], str]:
    """Validate one loader chunk against one exact label contract."""
    if not samples:
        raise LabelContractError(f"{chunk_context}: loader yielded an empty exam chunk")

    if expected_names is None:
        contract_names, contract_taxonomy = validate_label_definition(
            samples[0].label_names,
            samples[0].label_taxonomy,
            context=f"{chunk_context} exam {samples[0].exam_id}",
        )
    else:
        if expected_taxonomy is None:  # pragma: no cover
            raise LabelContractError(f"{chunk_context}: expected taxonomy is missing")
        contract_names = expected_names
        contract_taxonomy = expected_taxonomy

    for sample in samples:
        context = f"{chunk_context} exam {sample.exam_id}"
        names, _taxonomy = validate_label_definition(
            sample.label_names,
            sample.label_taxonomy,
            context=context,
            expected_names=contract_names,
            expected_taxonomy=contract_taxonomy,
        )
        validate_label_vector(
            sample.labels,
            num_classes=len(names),
            context=f"{context} labels",
        )
    return contract_names, contract_taxonomy


def _exam_sort_key(exam_id: str) -> tuple[int, object]:
    """Stable sort key for exam_ids that orders numeric ids numerically.

    ``ExamSample.exam_id`` is always a ``str``; a plain string sort would place
    ``'10'`` before ``'2'``. Numeric ids sort numerically; any non-numeric id
    sorts lexicographically after all numeric ones (the leading 0/1 keeps the
    two groups from being compared as int-vs-str).

    Uses ``str.isdecimal`` (not ``isdigit``): ``isdigit`` is also true for chars
    like superscripts that ``int()`` cannot parse, which would raise here.
    """
    return (0, int(exam_id)) if exam_id.isdecimal() else (1, exam_id)


class _BuildNotPublishError(Exception):
    """Abort a build transaction without publishing its staged artifact."""

    def __init__(self, return_code: int) -> None:
        super().__init__(return_code)
        self.return_code = return_code


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()

    try:
        config = load_config(args.config)
    except Exception as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        return 2

    config = _apply_overrides(config, args)

    repo_root = args.repo_root or Path.cwd()
    output_path = derive_output_path(config, repo_root=repo_root)
    reports_dir = derive_reports_dir(config, repo_root=repo_root)
    filename = derive_dataset_filename(config)

    console.print(f"[bold]Building[/bold] {filename}")
    console.print(f"  config:      {args.config}")
    console.print(f"  output:      {output_path}")
    console.print(f"  reports dir: {reports_dir}")
    console.print(f"  jobs:        {config.build.jobs}")
    console.print(f"  source:      {config.dataset_source}/{config.source_variant}")
    if config.limit_exams is not None:
        console.print(f"  limit_exams: {config.limit_exams}")

    if output_path.exists() and not config.build.overwrite:
        console.print(
            f"[red]Output already exists:[/red] {output_path}\n  Pass --overwrite to replace it."
        )
        return 2

    try:
        loader = get_loader(
            config.dataset_source,
            config.source_variant,
            limit_exams=config.limit_exams,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]Loader error:[/red] {exc}")
        return 2

    chunk_size = int(config.build.chunk_size)

    console.print("Processing exams ...")
    try:
        batch_iter = prefetch_iter(_iter_loader_batches(loader, batch_size=chunk_size), depth=1)
        try:
            first_chunk = next(batch_iter)
        except StopIteration:
            console.print("[red]No exams found in source.[/red]")
            return 2
        label_names, label_taxonomy = _validate_label_chunk(
            first_chunk,
            chunk_context="initial source chunk",
        )
        total = len(cast("Sized", loader)) if hasattr(loader, "__len__") else None
    except LabelContractError as exc:
        console.print(f"[red]Label contract error:[/red] {exc}")
        return 2
    except FileNotFoundError as exc:
        console.print(f"[red]Loader error:[/red] {exc}")
        return 2

    try:
        with staged_artifact(output_path, overwrite=config.build.overwrite) as staged_path:
            with (
                BuildReport(
                    reports_dir,
                    config=config,
                    dataset_filename=filename,
                    hdf5_path=output_path,
                ) as rep,
                DatasetWriter(
                    staged_path,
                    config=config,
                    label_taxonomy=label_taxonomy,
                    label_names=label_names,
                    created_by_version=__version__,
                    overwrite=True,
                ) as writer,
            ):
                is_tty = sys.stdout.isatty()
                progress_cm = (
                    Progress(
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(),
                        TextColumn(
                            "{task.completed}/{task.total}"
                            if total is not None
                            else "{task.completed}"
                        ),
                        TimeElapsedColumn(),
                        TimeRemainingColumn(),
                        console=console,
                    )
                    if is_tty
                    else nullcontext()
                )
                with progress_cm as progress:
                    load_task = (
                        progress.add_task("Loading", total=total) if progress is not None else None
                    )
                    process_task = (
                        progress.add_task("Processing", total=total)
                        if progress is not None
                        else None
                    )
                    advance_load = (
                        (lambda n: progress.advance(load_task, advance=n))
                        if progress is not None and load_task is not None
                        else None
                    )
                    advance = (
                        (lambda n: progress.advance(process_task, advance=n))
                        if progress is not None and process_task is not None
                        else None
                    )
                    started = time.monotonic()
                    exams_kept = 0
                    exams_skipped = 0

                    def _handle_result(result):
                        nonlocal exams_kept, exams_skipped
                        rep.append_skips(result.all_skips())
                        rep.increment_exam(kept=bool(result.success), leads_seen=len(result.leads))
                        if result.success:
                            exams_kept += 1
                        else:
                            exams_skipped += 1
                        for ld in result.leads:
                            rep.record_lead(
                                valid=ld.valid,
                                candidate_beats=ld.n_graphs
                                + sum(1 for sk in ld.skips if sk.level == "beat"),
                                kept_beats=ld.n_graphs,
                            )
                        if (
                            result.success
                            and result.metadata is not None
                            and result.packed is not None
                        ):
                            return result.metadata, result.packed
                        return None

                    def _process_chunk(chunk_samples: list) -> None:
                        if config.build.sort_output:
                            chunk_samples.sort(key=lambda s: _exam_sort_key(s.exam_id))
                        exam_batches = []
                        for result in _process_with_jobs(chunk_samples, config, advance=advance):
                            exam_batch = _handle_result(result)
                            if exam_batch is not None:
                                exam_batches.append(exam_batch)
                        if exam_batches:
                            writer.add_packed_exam_batch(exam_batches)

                    for chunk_index, chunk in enumerate(chain([first_chunk], batch_iter)):
                        _validate_label_chunk(
                            chunk,
                            expected_names=label_names,
                            expected_taxonomy=label_taxonomy,
                            chunk_context=f"source chunk {chunk_index}",
                        )
                        if advance_load is not None:
                            advance_load(len(chunk))
                        _process_chunk(chunk)

                rep.record_writer_totals(
                    n_graphs=writer.n_graphs,
                    n_nodes=writer.n_nodes,
                    n_edges=writer.n_edges,
                )

            n_graphs_written = writer.n_graphs

            if n_graphs_written == 0:
                console.print(
                    "[red]Build aborted:[/red] no valid graphs produced "
                    f"(see {reports_dir / 'skipped.jsonl'} for per-skip details)."
                )
                raise _BuildNotPublishError(1)

            qa_result = validate_dataset(staged_path, profile="quick")
            if not qa_result.success:
                console.print(
                    "[red]Build aborted:[/red] quick-profile QA failed on partial output:"
                )
                for issue in qa_result.errors():
                    console.print(f"  [{issue.code}] {issue.message}")
                raise _BuildNotPublishError(1)
    except _BuildNotPublishError as exc:
        return exc.return_code
    except LabelContractError as exc:
        console.print(f"[red]Label contract error:[/red] {exc}")
        return 2
    except FileExistsError:
        console.print(
            f"[red]Output already exists:[/red] {output_path}\n  Pass --overwrite to replace it."
        )
        return 2

    elapsed = time.monotonic() - started
    summary = {
        "exams_kept": exams_kept,
        "exams_skipped": exams_skipped,
        "elapsed_sec": round(elapsed, 2),
    }
    console.print(
        f"[green]Done[/green] {summary['exams_kept']} kept / "
        f"{summary['exams_skipped']} skipped in {summary['elapsed_sec']}s"
    )
    console.print(f"  HDF5:        {output_path}")
    console.print(f"  report.json: {reports_dir / 'report.json'}")
    if (reports_dir / "skipped.jsonl").exists():
        console.print(f"  skipped:     {reports_dir / 'skipped.jsonl'}")

    if config.visualization.smoke_during_build and n_graphs_written > 0:
        smoke_path = reports_dir / "figures" / f"smoke.{config.visualization.format}"
        try:
            import h5py

            from dataset.visualize import render_smoke_figure

            with h5py.File(output_path, "r") as f:
                fig = render_smoke_figure(f)
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(smoke_path, dpi=config.visualization.dpi, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)
            console.print(f"  smoke:       {smoke_path}")
        except Exception as exc:  # pragma: no cover
            console.print(f"[yellow]![/yellow] smoke figure not generated: {exc}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
