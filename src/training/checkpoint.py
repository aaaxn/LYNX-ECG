"""Checkpoint payload format, the single owner of write and read.

A checkpoint written by :func:`save_checkpoint` contains:

- ``model_state``: the unwrapped (non-DDP) ``state_dict``;
- ``model_hparams``: the model's constructor arguments (``None`` when the
  model does not expose ``hparams``, e.g. test stubs), making the file
  self-describing for :func:`reconstruct_model_from_ckpt`;
- ``epoch``: the epoch at save time;
- ``metrics``: the val metrics at save time;
- ``class_names``: the training class order, which evaluation datasets must
  match;
- ``dataset_fingerprint``: the training-dataset semantic identity (sampling
  rate, feature columns, amplitude normalization, class/lead space, and
  node-reducer signature) persisted as a plain ``dict`` so evaluation can
  fail closed before scoring a semantically different HDF5.
- ``symmetrize_edges``: whether the training run's ``ExamGraphDataset``
  symmetrized subgraph edges at load time (see
  :class:`training.dataset.ExamGraphDataset`). Consumers that rebuild graphs
  from the checkpoint (``evaluate``, analysis scripts) must construct their
  dataset with the same flag or their numbers are invalid; see
  :func:`training.hdf5_meta.require_compatible_dataset`.
- ``subgraph_representation``: the complete load-time beat-graph topology and
  edge-attribute policy. Consumers must resolve this field through
  :func:`resolve_subgraph_representation`.

Analysis scripts must rebuild models via
:func:`reconstruct_model_from_ckpt` rather than pattern-matching weight-key
shapes. Key-shape inference cannot recover dropout, GATv2 geometry, or the
encoder configuration, and silently breaks on any layer rename.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from pydantic import ValidationError

from config.train_models import SubgraphRepresentationConfig
from ecg_artifact_io import staged_artifact
from ecg_checkpoint_io import (
    load_bounded_weights_checkpoint_with_sha256,
    map_tensor_leaves,
    validate_tensor_payload,
)
from models.hier_model import HierGNNModel

if TYPE_CHECKING:
    from training.hdf5_meta import DatasetFingerprint


def unwrap_model(model: nn.Module) -> nn.Module:
    """Peel runtime wrappers (``torch.compile``, DDP) off ``model``.

    Checkpoints must always store the plain module's ``state_dict``. An
    ``OptimizedModule`` prefixes every key with ``_orig_mod.`` and DDP with
    ``module.``, either of which would make the file unloadable by
    :func:`reconstruct_model_from_ckpt`. Wrappers can nest (the training CLI
    compiles the DDP-wrapped model), so unwrap iteratively until the bare
    module is reached.
    """
    while True:
        orig_mod = getattr(model, "_orig_mod", None)
        if orig_mod is not None:
            model = orig_mod
            continue
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
            continue
        return model


def _canonicalize_tensors_to_cpu(value: Any) -> Any:
    """Return a primitive-container copy whose tensors are detached on CPU."""
    return map_tensor_leaves(value, lambda tensor: tensor.detach().cpu())


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    epoch: int,
    metric_dict: dict[str, float],
    class_names: Sequence[str],
    dataset_fingerprint: DatasetFingerprint,
    symmetrize_edges: bool,
    subgraph_representation: SubgraphRepresentationConfig,
) -> None:
    """Write the full checkpoint payload (see module docstring)."""
    base_model = unwrap_model(model)
    state = _canonicalize_tensors_to_cpu(
        {
            "model_state": base_model.state_dict(),
            "model_hparams": getattr(base_model, "hparams", None),
            "epoch": int(epoch),
            "metrics": metric_dict,
            "class_names": list(class_names),
            "dataset_fingerprint": asdict(dataset_fingerprint),
            "symmetrize_edges": bool(symmetrize_edges),
            "subgraph_representation": subgraph_representation.model_dump(),
        }
    )
    validate_tensor_payload(state)
    with staged_artifact(path, overwrite=True) as staged_path:
        torch.save(state, staged_path)
        staged_state = load_checkpoint(staged_path)
        require_exact_fields(
            staged_state,
            frozenset(state),
            label="serialized checkpoint payload is malformed",
        )


_SUBGRAPH_REPRESENTATION_FIELDS = frozenset(
    {
        "adjacency",
        "edge_attributes",
    }
)


def require_exact_fields(
    payload: dict[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    """Raise ``ValueError`` unless ``payload`` carries exactly the ``expected`` keys."""
    fields = frozenset(payload)
    if fields != expected:
        missing = sorted(expected - fields)
        extra = sorted(fields - expected)
        raise ValueError(f"{label}: missing={missing}, extra={extra}")


def resolve_symmetrize_edges(ckpt_state: dict[str, Any]) -> bool:
    """Return the checkpointed edge-direction policy."""
    if "symmetrize_edges" not in ckpt_state:
        raise ValueError("checkpoint has no 'symmetrize_edges' field")
    value = ckpt_state["symmetrize_edges"]
    if not isinstance(value, bool):
        raise ValueError("checkpoint symmetrize_edges must be a boolean")
    return value


def resolve_subgraph_representation(
    ckpt_state: dict[str, Any],
) -> SubgraphRepresentationConfig:
    """Return the checkpointed beat-graph representation; a missing or malformed payload fails."""
    if "subgraph_representation" not in ckpt_state:
        raise ValueError("checkpoint has no 'subgraph_representation' field")
    payload = ckpt_state["subgraph_representation"]
    if not isinstance(payload, dict):
        raise ValueError(
            "checkpoint 'subgraph_representation' payload is malformed: expected a mapping"
        )

    require_exact_fields(
        payload,
        _SUBGRAPH_REPRESENTATION_FIELDS,
        label="checkpoint 'subgraph_representation' payload is malformed",
    )

    try:
        return SubgraphRepresentationConfig.model_validate(payload)
    except (TypeError, ValidationError) as exc:
        raise ValueError(
            f"checkpoint 'subgraph_representation' payload is malformed: {exc}"
        ) from exc


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A validated checkpoint state and the digest of the exact bytes loaded."""

    state: dict[str, Any]
    sha256: str


def load_checkpoint_with_sha256(ckpt_path: Path) -> LoadedCheckpoint:
    """Stream-hash and bounded-load one stable open-file snapshot.

    Payload safety (primitive containers, dense CPU tensors, resource bounds)
    is already enforced by the bounded loader via ``validate_tensor_payload``;
    the checks below validate the checkpoint *format* on top of that.
    """
    state, digest = load_bounded_weights_checkpoint_with_sha256(ckpt_path)
    if not isinstance(state, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    model_state = state.get("model_state")
    if model_state is not None and (
        not isinstance(model_state, dict)
        or any(
            not isinstance(key, str) or not torch.is_tensor(value)
            for key, value in model_state.items()
        )
    ):
        raise ValueError("checkpoint model_state must be a string-keyed tensor mapping")
    if "dataset_fingerprint" in state and not isinstance(state["dataset_fingerprint"], dict):
        raise ValueError("checkpoint dataset_fingerprint must be a dictionary")
    resolve_symmetrize_edges(state)
    resolve_subgraph_representation(state)
    return LoadedCheckpoint(state=state, sha256=digest)


def load_checkpoint(ckpt_path: Path) -> dict[str, Any]:
    """Load and validate a checkpoint payload onto CPU."""
    return load_checkpoint_with_sha256(ckpt_path).state


def load_model_state(model: nn.Module, state: dict[str, Any]) -> nn.Module:
    """Load ``state["model_state"]`` into ``model``; returns the unwrapped module."""
    if state.get("model_hparams") is not None:
        resolve_model_hparams(state["model_hparams"])
    base_model = unwrap_model(model)
    base_model.load_state_dict(state["model_state"])
    return base_model


def resolve_model_hparams(hparams: object) -> dict[str, Any]:
    """Return the checkpointed model hparams after a type check."""
    if not isinstance(hparams, dict):
        raise ValueError("checkpoint 'model_hparams' payload must be a dictionary")
    if not all(isinstance(key, str) for key in hparams):
        raise ValueError("checkpoint 'model_hparams' keys must be strings")
    return dict(cast("dict[str, Any]", hparams))


def _reconstruct_model(
    ckpt_path: Path,
    loaded: LoadedCheckpoint,
    *,
    device: torch.device | None,
) -> HierGNNModel:
    state = loaded.state
    hparams = state.get("model_hparams")
    if hparams is None:
        raise ValueError(
            f"{ckpt_path}: checkpoint has no 'model_hparams' payload. Build the model "
            "explicitly and use load_model_state()."
        )
    hparams = resolve_model_hparams(hparams)
    state["model_hparams"] = hparams
    model = HierGNNModel(**hparams)
    model.load_state_dict(state["model_state"])
    if device is not None:
        model.to(device)
    return model


def reconstruct_model_from_ckpt_with_sha256(
    ckpt_path: Path,
    *,
    device: torch.device | None = None,
) -> tuple[HierGNNModel, dict[str, Any], str]:
    """Rebuild a model and return the digest of the exact checkpoint bytes loaded."""
    loaded = load_checkpoint_with_sha256(Path(ckpt_path))
    model = _reconstruct_model(Path(ckpt_path), loaded, device=device)
    return model, loaded.state, loaded.sha256


def reconstruct_model_from_ckpt(
    ckpt_path: Path,
    *,
    device: torch.device | None = None,
) -> tuple[HierGNNModel, dict[str, Any]]:
    """Rebuild the exact :class:`HierGNNModel` a checkpoint was saved from.

    Returns ``(model, state)`` so callers also get ``class_names`` and
    ``epoch`` without re-loading the file. Raises ``ValueError`` for
    checkpoints without ``model_hparams`` (for example from a
    non-``HierGNNModel`` training run); those must be loaded manually with an
    explicitly constructed model.
    """
    model, state, _sha256 = reconstruct_model_from_ckpt_with_sha256(Path(ckpt_path), device=device)
    return model, state
