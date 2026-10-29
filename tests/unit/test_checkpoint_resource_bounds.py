"""Adversarial resource-bound tests for the shared checkpoint loader."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
import torch

from ecg_checkpoint_io import (
    CheckpointFormatError,
    CheckpointLimits,
    load_bounded_weights_checkpoint_with_sha256,
    validate_tensor_payload,
)
from training.checkpoint import load_checkpoint

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]


def _canonical_payload(**extra) -> dict:
    payload = {
        "model_state": {"weight": torch.ones(2, dtype=torch.float32)},
        "epoch": 1,
        "metrics": {},
        "class_names": ["A"],
        "dataset_fingerprint": {},
        "symmetrize_edges": False,
        "subgraph_representation": {
            "adjacency": "stored_visibility",
            "edge_attributes": "derived",
        },
    }
    payload.update(extra)
    return payload


def _rewrite_zip(source: Path, output: Path, *, compress_data_pickle: bool = False) -> None:
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output, "w") as rewritten:
        for info in original.infolist():
            payload = original.read(info)
            compression = (
                zipfile.ZIP_DEFLATED
                if compress_data_pickle and info.filename.endswith("/data.pkl")
                else zipfile.ZIP_STORED
            )
            rewritten.writestr(info.filename, payload, compress_type=compression)


def test_shared_loader_rejects_compressed_zip_member_before_deserialization(tmp_path) -> None:
    source = tmp_path / "source.pt"
    compressed = tmp_path / "compressed.pt"
    torch.save(_canonical_payload(), source)
    _rewrite_zip(source, compressed, compress_data_pickle=True)

    with pytest.raises((CheckpointFormatError, ValueError), match="compressed"):
        load_checkpoint(compressed)


def test_shared_loader_rejects_oversized_metadata_member(tmp_path) -> None:
    source = tmp_path / "source.pt"
    oversized = tmp_path / "oversized.pt"
    torch.save(_canonical_payload(), source)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(oversized, "w") as rewritten:
        for info in original.infolist():
            payload = original.read(info)
            if info.filename.endswith("/data.pkl"):
                payload = b"x" * (16 * 1024**2 + 1)
            rewritten.writestr(info.filename, payload, compress_type=zipfile.ZIP_STORED)

    with pytest.raises((CheckpointFormatError, ValueError), match="metadata"):
        load_checkpoint(oversized)


def test_shared_loader_rejects_excessive_tensor_count(tmp_path) -> None:
    path = tmp_path / "many.pt"
    torch.save(_canonical_payload(many=[torch.empty(0) for _ in range(4097)]), path)

    with pytest.raises((CheckpointFormatError, ValueError), match="tensor count"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    "bad_tensor, message",
    [
        (torch.empty((1,) * 9), "rank"),
        (torch.ones(1, dtype=torch.complex64), "dtype"),
        (torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (1,)), "layout"),
    ],
)
def test_shared_loader_rejects_pathological_tensor_metadata(
    tmp_path, bad_tensor: torch.Tensor, message: str
) -> None:
    path = tmp_path / f"bad-{message}.pt"
    torch.save(_canonical_payload(model_state={"weight": bad_tensor}), path)

    with pytest.raises((CheckpointFormatError, ValueError), match=message):
        load_checkpoint(path)


def test_tensor_storage_and_container_depth_limits_are_enforced() -> None:
    with pytest.raises(CheckpointFormatError, match="storage"):
        validate_tensor_payload(
            {"weight": torch.ones(2, dtype=torch.float32)},
            limits=CheckpointLimits(max_storage_bytes=4),
        )

    nested: object = None
    for _ in range(66):
        nested = [nested]
    with pytest.raises(CheckpointFormatError, match="depth"):
        validate_tensor_payload(nested)


def test_loader_rejects_non_pytorch_and_extra_zip_members(tmp_path) -> None:
    plain = tmp_path / "plain.pt"
    plain.write_bytes(b"not a zip")
    with pytest.raises(CheckpointFormatError, match="PyTorch ZIP"):
        load_bounded_weights_checkpoint_with_sha256(plain)

    source = tmp_path / "source.pt"
    extra = tmp_path / "extra.pt"
    torch.save(_canonical_payload(), source)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(extra, "w") as rewritten:
        for info in original.infolist():
            rewritten.writestr(info.filename, original.read(info), compress_type=zipfile.ZIP_STORED)
        root = original.infolist()[0].filename.split("/", 1)[0]
        rewritten.writestr(f"{root}/unexpected.bin", b"x")
    with pytest.raises(CheckpointFormatError, match="unsupported"):
        load_bounded_weights_checkpoint_with_sha256(extra)
