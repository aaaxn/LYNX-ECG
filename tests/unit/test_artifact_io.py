"""Atomic-publish and hashing primitives (ecg_artifact_io)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from ecg_artifact_io import (
    preflight_artifact_package,
    publish_artifact_package,
    sha256_file,
    staged_artifact,
)

pytestmark = pytest.mark.unit


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    path = tmp_path / "payload.bin"
    payload = bytes(range(256)) * 5000
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_staged_artifact_publishes_on_success_and_leaves_no_temp(tmp_path) -> None:
    output = tmp_path / "nested" / "artifact.bin"

    with staged_artifact(output) as temporary_path:
        assert temporary_path.parent == output.parent
        temporary_path.write_bytes(b"published")

    assert output.read_bytes() == b"published"
    assert list(output.parent.glob("*.tmp")) == []


def test_staged_artifact_refuses_to_clobber_without_overwrite(tmp_path) -> None:
    output = tmp_path / "artifact.bin"
    output.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError), staged_artifact(output) as temporary_path:
        temporary_path.write_bytes(b"racer")

    assert output.read_bytes() == b"keep-me"
    assert list(tmp_path.glob("*.tmp")) == []


def test_staged_artifact_overwrite_replaces_existing_output(tmp_path) -> None:
    output = tmp_path / "artifact.bin"
    output.write_bytes(b"replace-me")

    with staged_artifact(output, overwrite=True) as temporary_path:
        temporary_path.write_bytes(b"replacement")
        assert output.read_bytes() == b"replace-me"

    assert output.read_bytes() == b"replacement"
    assert list(tmp_path.glob("*.tmp")) == []


def test_staged_artifact_publishes_nothing_on_body_failure(tmp_path) -> None:
    output = tmp_path / "artifact.bin"

    with (
        pytest.raises(RuntimeError, match="body failed"),
        staged_artifact(output) as temporary_path,
    ):
        temporary_path.write_bytes(b"partial")
        raise RuntimeError("body failed")

    assert not output.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_staged_artifact_failure_preserves_existing_output(tmp_path) -> None:
    output = tmp_path / "artifact.bin"
    output.write_bytes(b"keep-me")

    with (
        pytest.raises(RuntimeError, match="body failed"),
        staged_artifact(output, overwrite=True) as temporary_path,
    ):
        temporary_path.write_bytes(b"partial")
        raise RuntimeError("body failed")

    assert output.read_bytes() == b"keep-me"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_artifact_package_preflight_rejects_non_directory_output(tmp_path) -> None:
    output = tmp_path / "output"
    output.write_text("not a directory")

    with pytest.raises(ValueError, match="not a directory"):
        preflight_artifact_package(output, ("a",))


def test_artifact_package_rejects_symlinked_output_directory(tmp_path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    protected = target / "protected"
    protected.write_text("keep")
    output = tmp_path / "output"
    output.symlink_to(target, target_is_directory=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "current").write_text("new")

    with pytest.raises(ValueError, match="not a directory"):
        preflight_artifact_package(output, ("current",), overwrite=True)
    with pytest.raises(ValueError, match="not a directory"):
        publish_artifact_package(staged, output, ("current",), overwrite=True)

    assert protected.read_text() == "keep"
    assert not (target / "current").exists()


def test_artifact_package_preflight_rejects_non_regular_destination(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "a").mkdir()

    with pytest.raises(ValueError, match="regular files"):
        preflight_artifact_package(output, ("a",))


def test_artifact_package_preflight_refuses_existing_file_without_overwrite(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "a").write_text("existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        preflight_artifact_package(output, ("a",))


def test_artifact_package_overwrite_replaces_every_file(tmp_path) -> None:
    staged = tmp_path / "staged"
    output = tmp_path / "output"
    staged.mkdir()
    output.mkdir()
    for name in ("a", "b", "c"):
        (staged / name).write_text(f"new-{name}")
        (output / name).write_text(f"old-{name}")

    publish_artifact_package(staged, output, ("a", "b", "c"), overwrite=True)

    assert [(output / name).read_text() for name in ("a", "b", "c")] == [
        "new-a",
        "new-b",
        "new-c",
    ]


def test_artifact_package_overwrite_failure_restores_every_file(tmp_path, monkeypatch) -> None:
    staged = tmp_path / "staged"
    output = tmp_path / "output"
    staged.mkdir()
    output.mkdir()
    for name in ("a", "b", "c"):
        (staged / name).write_text(f"new-{name}")
        (output / name).write_text(f"old-{name}")
    real_replace = os.replace

    def fail_on_second(source, destination):
        if Path(destination).name == "b":
            raise OSError("simulated replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_on_second)

    with pytest.raises(OSError, match="simulated replace failure"):
        publish_artifact_package(staged, output, ("a", "b", "c"), overwrite=True)

    assert [(output / name).read_text() for name in ("a", "b", "c")] == [
        "old-a",
        "old-b",
        "old-c",
    ]


def test_staged_artifact_losing_create_race_preserves_winner(tmp_path, monkeypatch) -> None:
    output = tmp_path / "artifact.bin"
    real_link = os.link

    def publish_winner_then_link(source, destination) -> None:
        output.write_bytes(b"winner")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", publish_winner_then_link)

    with pytest.raises(FileExistsError), staged_artifact(output) as temporary_path:
        temporary_path.write_bytes(b"loser")

    assert output.read_bytes() == b"winner"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []
