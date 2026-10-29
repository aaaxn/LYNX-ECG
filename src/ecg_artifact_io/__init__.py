"""Shared primitives for hash-attested, atomically published artifact files.

Dataset builds, checkpoints, and analysis reports are written to a temporary
file next to their final path and then published atomically. This package is
the single home for that idiom and for streaming sha256 hashing.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

_CHUNK_BYTES = 1024 * 1024


def sha256_stream(stream: BinaryIO) -> str:
    """Hash bytes from the stream's current position to EOF in bounded chunks."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Stream ``path`` and return the sha256 hex digest of its bytes."""
    with path.open("rb") as stream:
        return sha256_stream(stream)


@contextlib.contextmanager
def staged_artifact(output_path: Path, *, overwrite: bool = False) -> Iterator[Path]:
    """Yield a same-directory temporary path, atomically published on success.

    The body writes the complete artifact to the yielded path. On clean exit
    it is published to ``output_path`` with ``os.replace`` when ``overwrite``
    is set, otherwise with ``os.link``, an atomic create-if-absent operation on the
    same filesystem that cannot clobber an output created by a racing
    process. On any exception nothing becomes visible at ``output_path``.
    The temporary file is always removed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path: Path | None = Path(temporary.name)
    try:
        yield temporary_path
        if overwrite:
            os.replace(temporary_path, output_path)
            temporary_path = None
        else:
            os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def is_regular_artifact_file(path: Path) -> bool:
    """Return whether a path entry is a regular file without following symlinks."""
    return stat.S_ISREG(path.lstat().st_mode)


def _file_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino


def _is_directory_entry(path: Path) -> bool:
    """Return whether the path entry itself is a directory, never a symlink."""
    return stat.S_ISDIR(path.lstat().st_mode)


@contextlib.contextmanager
def _package_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir.parent / f".{output_dir.name}.artifact-package.lock"
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def preflight_artifact_package(
    output_dir: Path,
    names: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Validate package destinations before expensive source loading/rendering."""
    if os.path.lexists(output_dir) and not _is_directory_entry(output_dir):
        raise ValueError(f"output directory path is not a directory entry: {output_dir}")
    final_paths = [output_dir / name for name in names]
    invalid = [
        path for path in final_paths if os.path.lexists(path) and not is_regular_artifact_file(path)
    ]
    if invalid:
        raise ValueError(f"package destinations must be regular files: {invalid}")
    existing = [path for path in final_paths if os.path.lexists(path)]
    if not overwrite and existing:
        raise FileExistsError(f"refusing to overwrite existing package files: {existing}")
    return final_paths


def publish_artifact_package(
    staged_dir: Path,
    output_dir: Path,
    names: tuple[str, ...],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a locked set of regular files with complete rollback."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with _package_lock(output_dir):
        if os.path.lexists(output_dir):
            if not _is_directory_entry(output_dir):
                raise ValueError(f"output directory path is not a directory entry: {output_dir}")
        else:
            output_dir.mkdir(parents=True)
        destinations = [output_dir / name for name in names]
        for destination in destinations:
            if os.path.lexists(destination) and not is_regular_artifact_file(destination):
                raise ValueError(f"package destination is not a regular file: {destination}")

        published: dict[Path, tuple[int, int]] = {}
        if not overwrite:
            try:
                for name, destination in zip(names, destinations, strict=True):
                    source = staged_dir / name
                    source_identity = _file_identity(source)
                    os.link(source, destination)
                    published[destination] = source_identity
            except OSError:
                for destination, identity in published.items():
                    if os.path.lexists(destination) and _file_identity(destination) == identity:
                        destination.unlink()
                raise
            return

        existing_paths = [path for path in destinations if os.path.lexists(path)]
        backup_dir = staged_dir / ".artifact-package-backups"
        backup_dir.mkdir()
        backups: dict[Path, Path] = {}
        try:
            for index, destination in enumerate(existing_paths):
                backup = backup_dir / str(index)
                os.link(destination, backup, follow_symlinks=False)
                backups[destination] = backup
            for name, destination in zip(names, destinations, strict=True):
                source = staged_dir / name
                source_identity = _file_identity(source)
                os.replace(source, destination)
                published[destination] = source_identity
        except (OSError, ValueError):
            for destination, identity in published.items():
                if not os.path.lexists(destination) or _file_identity(destination) != identity:
                    continue
                if backup := backups.get(destination):
                    os.replace(backup, destination)
                else:
                    destination.unlink()
            raise
