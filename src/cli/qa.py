"""``dataset-qa`` CLI: validate a built HDF5 against the current dataset contract.

Exit codes:
- ``0``: validation passed.
- ``1``: schema / contract violation (one or more error-severity issues).
- ``2``: execution error (file missing, unreadable, internal exception).

``--strict-warnings`` upgrades warnings to errors for CI gating.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from dataset.qa import validate_dataset


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dataset-qa",
        description="Validate a LYNX-ECG HDF5 against the current dataset contract.",
    )
    parser.add_argument("--h5", required=True, type=Path, help="Path to the HDF5 dataset.")
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="full",
        help="Validation profile (default: full).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the QAResult as JSON.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as errors (exit 1 if any warning).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()

    try:
        result = validate_dataset(args.h5, profile=args.profile)
    except Exception as exc:  # pragma: no cover
        console.print(f"[red]QA execution error:[/red] {exc}")
        return 2

    if result.success and not result.issues:
        console.print(
            f"[green]✓[/green] QA [bold]{args.profile}[/bold] passed "
            f"({result.summary.get('num_graphs', 'n/a')} graphs)."
        )
    else:
        for issue in result.issues:
            tag = "[red]✗[/red]" if issue.severity == "error" else "[yellow]![/yellow]"
            console.print(f"  {tag} {issue.severity}: {issue.code}: {issue.message}")
        if result.success:
            console.print("[yellow]Warnings only; dataset is valid.[/yellow]")
        else:
            console.print(
                f"[red]✗ QA {args.profile} FAILED[/red] "
                f"({len(result.errors())} errors, {len(result.issues) - len(result.errors())} warnings)."
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    if not result.success:
        return 1
    if args.strict_warnings and any(i.severity == "warning" for i in result.issues):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
