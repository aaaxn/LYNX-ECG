"""Resource-bounded PyTorch checkpoint deserialization.

The weights-only unpickler prevents arbitrary Python reconstruction, but it does
not bound ZIP expansion, container fan-out, tensor metadata, or storage size.
This module validates the PyTorch ZIP directory without extracting members,
then loads from the same open regular-file handle and validates the resulting
object graph and tensor resources.

This module is also the single owner of the checkpoint payload tree shape
(tensors and primitive ``dict``/``list``/``tuple`` containers with scalar
leaves): :func:`validate_tensor_payload` enforces it fail-closed, and
:func:`map_tensor_leaves` is the canonical walker over it.
"""

from __future__ import annotations

import os
import stat
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ecg_artifact_io import sha256_stream

_GIB = 1024**3
_SUPPORTED_DTYPES = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
)


class CheckpointFormatError(ValueError, RuntimeError):
    """Checkpoint structure or resource use exceeds the accepted contract."""


@dataclass(frozen=True)
class CheckpointLimits:
    max_file_bytes: int = 2 * _GIB
    max_members: int = 10_000
    max_member_bytes: int = 2 * _GIB
    max_total_member_bytes: int = 2 * _GIB
    max_metadata_bytes: int = 16 * 1024**2
    max_depth: int = 64
    max_container_items: int = 100_000
    max_tensors: int = 4_096
    max_storage_bytes: int = 2 * _GIB
    max_tensor_rank: int = 8
    max_tensor_dim: int = 1_000_000_000
    max_tensor_numel: int = 1_000_000_000


DEFAULT_LIMITS = CheckpointLimits()
_ALLOWED_FIXED_MEMBERS = frozenset(
    {
        "data.pkl",
        "version",
        "byteorder",
        ".format_version",
        ".storage_alignment",
        ".data/serialization_id",
    }
)
_ALLOWED_ZIP_FLAGS = 0x08 | 0x800
_ALLOWED_KEY_TYPES = (str, int, bool, float)


def _snapshot(fd: int) -> tuple[int, int, int, int, int]:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointFormatError("checkpoint input must be a regular file")
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_mode,
    )


def _validate_zip_directory(handle: Any, limits: CheckpointLimits) -> None:
    try:
        archive = zipfile.ZipFile(handle)
    except zipfile.BadZipFile as exc:
        raise CheckpointFormatError("checkpoint is not a supported PyTorch ZIP archive") from exc
    with archive:
        members = archive.infolist()
        if not members or len(members) > limits.max_members:
            raise CheckpointFormatError("checkpoint ZIP member count exceeds the limit")
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise CheckpointFormatError("checkpoint ZIP contains duplicate member names")
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(roots) != 1:
            raise CheckpointFormatError("checkpoint ZIP must have one serialization root")
        root = roots.pop()
        total = 0
        relative_names: set[str] = set()
        for member in members:
            name = member.filename
            if member.is_dir() or "\\" in name or name.startswith("/"):
                raise CheckpointFormatError("checkpoint ZIP contains an unsupported member path")
            parts = name.split("/")
            if any(part in ("", ".", "..") for part in parts) or parts[0] != root:
                raise CheckpointFormatError("checkpoint ZIP contains an unsafe member path")
            relative = "/".join(parts[1:])
            if relative not in _ALLOWED_FIXED_MEMBERS and not (
                relative.startswith("data/") and relative[5:].isdigit()
            ):
                raise CheckpointFormatError(f"unsupported checkpoint ZIP member: {relative}")
            if member.flag_bits & ~_ALLOWED_ZIP_FLAGS:
                raise CheckpointFormatError("checkpoint ZIP uses unsupported member flags")
            if (
                member.compress_type != zipfile.ZIP_STORED
                or member.file_size != member.compress_size
            ):
                raise CheckpointFormatError("compressed checkpoint ZIP members are not accepted")
            if member.file_size < 0 or member.file_size > limits.max_member_bytes:
                raise CheckpointFormatError("checkpoint ZIP member exceeds the size limit")
            if relative == "data.pkl" and member.file_size > limits.max_metadata_bytes:
                raise CheckpointFormatError("checkpoint metadata exceeds the size limit")
            total += member.file_size
            if total > limits.max_total_member_bytes:
                raise CheckpointFormatError("checkpoint ZIP aggregate size exceeds the limit")
            relative_names.add(relative)
        if "data.pkl" not in relative_names or "version" not in relative_names:
            raise CheckpointFormatError("checkpoint ZIP lacks required PyTorch metadata")
    handle.seek(0)


def map_tensor_leaves(value: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    """Rebuild a primitive-container structure, applying ``fn`` to each tensor leaf.

    Walks the checkpoint payload tree once, replacing every tensor with
    ``fn(tensor)`` while preserving the container shape (a ``list`` stays a
    ``list``, a ``tuple`` becomes a plain ``tuple``); non-tensor leaves pass
    through untouched. Purely structural: unknown object types also pass
    through, so callers that need the fail-closed contract must still run
    :func:`validate_tensor_payload` on the result.
    """
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {key: map_tensor_leaves(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [map_tensor_leaves(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(map_tensor_leaves(item, fn) for item in value)
    return value


def validate_tensor_payload(
    value: Any,
    *,
    limits: CheckpointLimits = DEFAULT_LIMITS,
    path: str = "<root>",
) -> None:
    """Validate bounded primitive containers and dense CPU tensors."""

    items = 0
    tensors = 0
    storage_bytes = 0
    seen_storages: set[tuple[int, int]] = set()
    stack: list[tuple[Any, str, int]] = [(value, path, 0)]
    while stack:
        current, current_path, depth = stack.pop()
        if depth > limits.max_depth:
            raise CheckpointFormatError(
                f"checkpoint container depth exceeds limit at {current_path}"
            )
        if torch.is_tensor(current):
            tensors += 1
            if tensors > limits.max_tensors:
                raise CheckpointFormatError("checkpoint tensor count exceeds the limit")
            if current.device.type != "cpu" or current.layout is not torch.strided:
                raise CheckpointFormatError(f"unsupported tensor device/layout at {current_path}")
            if current.is_quantized or current.dtype not in _SUPPORTED_DTYPES:
                raise CheckpointFormatError(f"unsupported tensor dtype at {current_path}")
            if current.ndim > limits.max_tensor_rank:
                raise CheckpointFormatError(f"tensor rank exceeds the limit at {current_path}")
            if any(
                dimension < 0 or dimension > limits.max_tensor_dim for dimension in current.shape
            ):
                raise CheckpointFormatError(f"tensor dimension exceeds the limit at {current_path}")
            if current.numel() > limits.max_tensor_numel:
                raise CheckpointFormatError(
                    f"tensor element count exceeds the limit at {current_path}"
                )
            storage = current.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            if identity not in seen_storages:
                seen_storages.add(identity)
                storage_bytes += storage.nbytes()
                if storage_bytes > limits.max_storage_bytes:
                    raise CheckpointFormatError(
                        "checkpoint tensor storage exceeds the aggregate limit"
                    )
            continue
        if current is None or type(current) in (bool, int, float, str):
            continue
        if isinstance(current, dict):
            items += len(current)
            if items > limits.max_container_items:
                raise CheckpointFormatError("checkpoint container item count exceeds the limit")
            for key, child in current.items():
                if key is not None and type(key) not in _ALLOWED_KEY_TYPES:
                    raise CheckpointFormatError(
                        f"unsupported checkpoint key type {type(key).__name__} at {current_path}"
                    )
                stack.append((child, f"{current_path}[{key!r}]", depth + 1))
            continue
        if isinstance(current, (list, tuple)):
            items += len(current)
            if items > limits.max_container_items:
                raise CheckpointFormatError("checkpoint container item count exceeds the limit")
            stack.extend(
                (child, f"{current_path}[{index}]", depth + 1)
                for index, child in enumerate(current)
            )
            continue
        raise CheckpointFormatError(
            f"unsupported checkpoint object of type {type(current).__name__} at {current_path}"
        )


def load_bounded_weights_stream(
    handle: Any,
    *,
    limits: CheckpointLimits = DEFAULT_LIMITS,
) -> Any:
    """Load and validate a bounded PyTorch ZIP from an already-open snapshot."""

    before = _snapshot(handle.fileno())
    if before[2] > limits.max_file_bytes:
        raise CheckpointFormatError("checkpoint file exceeds the size limit")
    handle.seek(0)
    _validate_zip_directory(handle, limits)
    try:
        value = torch.load(handle, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointFormatError(
            "failed to safely deserialize bounded checkpoint; unsafe fallback "
            "is intentionally not available"
        ) from exc
    if _snapshot(handle.fileno()) != before:
        raise CheckpointFormatError("checkpoint file changed while it was being loaded")
    validate_tensor_payload(value, limits=limits)
    return value


def load_bounded_weights_checkpoint_with_sha256(
    path: str | os.PathLike[str],
    *,
    limits: CheckpointLimits = DEFAULT_LIMITS,
) -> tuple[Any, str]:
    """Hash and bounded-load the exact same opened regular-file snapshot."""

    checkpoint_path = Path(path)
    with checkpoint_path.open("rb") as handle:
        info = _snapshot(handle.fileno())
        if info[2] > limits.max_file_bytes:
            raise CheckpointFormatError("checkpoint file exceeds the size limit")
        digest = sha256_stream(handle)
        if _snapshot(handle.fileno()) != info:
            raise CheckpointFormatError("checkpoint file changed while it was being hashed")
        handle.seek(0)
        value = load_bounded_weights_stream(handle, limits=limits)
    return value, digest
