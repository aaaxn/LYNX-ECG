"""Shared ``SEED=PATH`` run-declaration parsing for workflow scripts."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path


@dataclass(frozen=True)
class SeedPathArgument:
    """One declared training seed and its per-seed source path."""

    seed: int
    path: Path


def parse_seed_path(value: str, *, expected: str) -> SeedPathArgument:
    """Parse one ``SEED=PATH`` declaration; ``expected`` names the metavar."""
    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text or not path_text:
        raise argparse.ArgumentTypeError(f"expected {expected}")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid seed {seed_text!r}") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return SeedPathArgument(seed=seed, path=Path(path_text))


def seed_path_type(expected: str) -> Callable[[str], SeedPathArgument]:
    """Return an argparse ``type=`` callable for one ``SEED=PATH`` metavar."""
    return partial(parse_seed_path, expected=expected)
