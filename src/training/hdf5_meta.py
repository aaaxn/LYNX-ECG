"""Typed HDF5 dataset metadata reader.

Single-open read of the current dataset contract: exam counts, class names,
label arrays, split folds, and sampling parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from config.models import DatasetConfig
from dataset.schema import require_dataset_contract

if TYPE_CHECKING:
    from config.train_models import ModelConfig


@dataclass(frozen=True)
class DatasetFingerprint:
    """The semantic identity of a dataset, for compatibility checks.

    Captures everything that must match for a model trained on one dataset to
    be validly evaluated on another: label/lead space, sampling rate, feature
    columns, amplitude normalization, and the canonical active node-reducer
    signature. Excludes per-exam arrays so it is
    cheap to persist in a checkpoint via ``dataclasses.asdict``.
    """

    num_classes: int
    num_leads: int
    class_names: list[str]
    sampling_rate: int
    node_feature_names: list[str]
    edge_attr_names: list[str]
    amplitude_normalization: str | None
    node_reduction: dict[str, Any]


@dataclass(frozen=True)
class HDF5Meta:
    num_classes: int
    num_leads: int
    class_names: list[str]
    strat_fold: np.ndarray | None
    exam_labels: np.ndarray
    sampling_rate: int
    total_exams: int
    total_graphs: int
    patient_ids: np.ndarray | None
    node_feature_names: list[str]
    edge_attr_names: list[str]
    node_reduction: dict[str, Any]
    amplitude_normalization: str | None = None

    @property
    def fingerprint(self) -> DatasetFingerprint:
        """Derive the semantic-identity fingerprint of this dataset."""
        return DatasetFingerprint(
            num_classes=self.num_classes,
            num_leads=self.num_leads,
            class_names=list(self.class_names),
            sampling_rate=self.sampling_rate,
            node_feature_names=list(self.node_feature_names),
            edge_attr_names=list(self.edge_attr_names),
            amplitude_normalization=self.amplitude_normalization,
            node_reduction=dict(self.node_reduction),
        )


def assert_fingerprints_compatible(
    reference: DatasetFingerprint,
    candidate: DatasetFingerprint,
    *,
    reference_label: str,
    candidate_label: str,
) -> None:
    """Raise ``ValueError`` if two dataset fingerprints are not identical.

    ``reference`` is the trusted baseline (the training dataset);
    ``candidate`` is the dataset being checked against it. The labels appear
    in the error prose so the same comparator serves both the train-vs-test
    split check and the checkpoint-vs-evaluation-dataset check.
    """
    if candidate.num_classes != reference.num_classes:
        raise ValueError(
            f"{candidate_label} has {candidate.num_classes} classes, "
            f"but {reference_label} has {reference.num_classes}"
        )
    if candidate.class_names != reference.class_names:
        raise ValueError(
            f"{candidate_label} label names {candidate.class_names} "
            f"do not match {reference_label} {reference.class_names}"
        )
    if candidate.num_leads != reference.num_leads:
        raise ValueError(
            f"{candidate_label} has {candidate.num_leads} leads, "
            f"but {reference_label} has {reference.num_leads}"
        )
    if candidate.sampling_rate != reference.sampling_rate:
        raise ValueError(
            f"{candidate_label} sampling_rate {candidate.sampling_rate} Hz "
            f"does not match {reference_label} {reference.sampling_rate} Hz"
        )
    if candidate.node_feature_names != reference.node_feature_names:
        raise ValueError(
            f"{candidate_label} node features {candidate.node_feature_names} "
            f"do not match {reference_label} {reference.node_feature_names}; "
            "the model would read the wrong (or missing) node_x columns."
        )
    if candidate.edge_attr_names != reference.edge_attr_names:
        raise ValueError(
            f"{candidate_label} edge attrs {candidate.edge_attr_names} "
            f"do not match {reference_label} {reference.edge_attr_names}"
        )
    if candidate.amplitude_normalization != reference.amplitude_normalization:
        raise ValueError(
            f"{candidate_label} amplitude_normalization "
            f"{candidate.amplitude_normalization!r} does not match "
            f"{reference_label} {reference.amplitude_normalization!r}; "
            "node-amplitude features live on different scales, so threshold "
            "calibration would not transfer."
        )
    if candidate.node_reduction != reference.node_reduction:
        raise ValueError(
            f"{candidate_label} node_reduction {candidate.node_reduction!r} does not match "
            f"{reference_label} {reference.node_reduction!r}; stored node semantics differ."
        )


def assert_test_meta_compatible(train_meta: HDF5Meta, test_meta: HDF5Meta) -> None:
    """Raise ``ValueError`` if a separate test dataset is incompatible with train.

    Training may load an optional second HDF5 for the test split. It must be structurally compatible with the training
    dataset, or evaluation silently scores against wrong data:

    * ``num_classes`` / ``num_leads`` mismatches are unusable;
    * a differing ``class_names`` order evaluates predictions against the wrong
      ground-truth columns;
    * a differing ``sampling_rate`` changes the time scale of the graphs built
      from the signal (the ``delta_t`` edge attribute scales with
      1/sampling_rate), so the model would see an out-of-distribution test set;
    * a differing ``amplitude_normalization`` changes the node-amplitude feature
      scale. Raw and normalized builds live on different scales, so threshold
      calibration from a normalized training set would not transfer to a raw test
      set.

    The invariant lives here so the two call sites cannot drift apart.
    """
    assert_fingerprints_compatible(
        train_meta.fingerprint,
        test_meta.fingerprint,
        reference_label="train dataset",
        candidate_label="test_dataset",
    )


def assert_model_meta_compatible(model_config: ModelConfig, meta: HDF5Meta) -> None:
    """Raise ``ValueError`` when the model's input claims don't match the dataset.

    ``ModelConfig.in_channels`` / ``edge_dim`` declare what the HDF5 provides.
    Unchecked, a mismatch only surfaces as an opaque shape error inside the
    first GINEConv forward, after loaders and model are already up.
    Both CLI entry points call this right after reading the metadata so the
    failure is immediate and names both sides.
    """
    if model_config.in_channels != len(meta.node_feature_names):
        raise ValueError(
            f"model.in_channels={model_config.in_channels} but the dataset carries "
            f"{len(meta.node_feature_names)} node features {meta.node_feature_names}; "
            "fix the preset's model.in_channels (or rebuild the dataset)."
        )
    if model_config.edge_dim != len(meta.edge_attr_names):
        raise ValueError(
            f"model.edge_dim={model_config.edge_dim} but the dataset derives "
            f"{len(meta.edge_attr_names)} edge attrs {meta.edge_attr_names}; "
            "fix the preset's model.edge_dim (or rebuild the dataset)."
        )


def _read_node_reduction(
    h5: h5py.File,
    h5_path: Path,
    config: DatasetConfig,
) -> dict[str, Any]:
    reducer = config.node_reduction
    if reducer.method == "none":
        node_reduction: dict[str, Any] = {"method": "none"}
    else:
        active = getattr(reducer, reducer.method)
        if active is None:
            raise ValueError(f"{h5_path}: node_reduction has no active {reducer.method!r} config")
        node_reduction = {"method": reducer.method, **active.model_dump()}

    active_target_nodes = (
        getattr(getattr(reducer, reducer.method, None), "target_nodes", None)
        if reducer.method != "none"
        else None
    )
    root_target_nodes = h5.attrs.get("target_nodes")
    if root_target_nodes is not None and (
        active_target_nodes is None or int(root_target_nodes) != int(active_target_nodes)
    ):
        raise ValueError(
            f"{h5_path}: target_nodes={int(root_target_nodes)!r} disagrees with "
            f"config_json active reducer target_nodes={active_target_nodes!r}"
        )
    return node_reduction


def read_hdf5_meta(h5_path: Path) -> HDF5Meta:
    """Read and validate dataset metadata in one h5py open."""
    with h5py.File(h5_path, "r") as f:
        config = require_dataset_contract(f, h5_path)
        graph_leads = list(f.attrs["graph_leads"])
        num_leads = max(int(max(graph_leads)) + 1, 1)
        strat_fold = f["exam_strat_fold"][:] if "exam_strat_fold" in f else None
        patient_ids = f["exam_patient_id"][:] if "exam_patient_id" in f else None
        node_feature_names = [str(n) for n in f.attrs["node_feature_names"]]
        num_node_features = int(f["node_x"].shape[1])
        if len(node_feature_names) != num_node_features:
            raise ValueError(
                f"{h5_path}: node_feature_names attr lists {len(node_feature_names)} "
                f"features {node_feature_names} but node_x has {num_node_features} "
                "columns. The file is corrupt or was built outside the standard pipeline."
            )
        norm_attr = f.attrs.get("amplitude_normalization")
        amplitude_normalization = str(norm_attr) if norm_attr is not None else None
        node_reduction = _read_node_reduction(f, h5_path, config)
        return HDF5Meta(
            num_classes=int(f["exam_labels"].shape[1]),
            num_leads=num_leads,
            class_names=list(f.attrs["exam_label_names"]),
            strat_fold=strat_fold,
            exam_labels=f["exam_labels"][:].astype(np.uint8),
            sampling_rate=int(f.attrs["sampling_rate"]),
            total_exams=int(f["exam_id"].shape[0]),
            total_graphs=int(f["graph_exam_index"].shape[0]),
            patient_ids=patient_ids,
            node_feature_names=node_feature_names,
            edge_attr_names=[str(n) for n in f.attrs["edge_attr_names"]],
            amplitude_normalization=amplitude_normalization,
            node_reduction=node_reduction,
        )


def require_compatible_dataset(
    ckpt_state: dict[str, Any],
    h5_path: Path,
    *,
    ckpt_label: str = "checkpoint",
    allow_symmetrized: bool = False,
) -> HDF5Meta:
    """Validate that ``h5_path`` is semantically compatible with a checkpoint.

    Runs the full guard the ``evaluate`` CLI pioneered: (1) the checkpoint must
    carry a ``dataset_fingerprint`` payload; (2) the payload must parse into
    :class:`DatasetFingerprint`; (3) the fingerprint must be compatible with the
    dataset's (``assert_fingerprints_compatible``); (4) the checkpoint's
    top-level ``class_names`` must match the dataset's class order (the
    model's logit columns follow that order);
    (5) unless ``allow_symmetrized=True``, the checkpoint must not have been
    trained with ``symmetrize_edges=True``. A consumer that builds directed
    (as-stored) graphs would silently score a model on a different receptive
    field than it was trained with. Raises ``ValueError`` with an actionable
    message on any failure; returns the parsed :class:`HDF5Meta` on success so
    callers read the file once.
    """
    meta = read_hdf5_meta(h5_path)

    if ckpt_state["symmetrize_edges"] and not allow_symmetrized:
        raise ValueError(
            f"this {ckpt_label} was trained with symmetrize_edges=True, but this "
            "consumer builds directed (as-stored) graphs; its numbers would be "
            "invalid. Use the evaluate CLI, or update the consumer to construct "
            "its ExamGraphDataset with symmetrize_edges=True."
        )

    ckpt_fingerprint = ckpt_state.get("dataset_fingerprint")
    if ckpt_fingerprint is None:
        raise ValueError(
            f"this {ckpt_label} has no 'dataset_fingerprint' payload, so its "
            "training-data semantics (sampling rate, feature columns, amplitude "
            "normalization) cannot be verified against the evaluation dataset."
        )
    try:
        ckpt_fp = DatasetFingerprint(**ckpt_fingerprint)
    except TypeError as exc:
        raise ValueError(
            f"{ckpt_label} 'dataset_fingerprint' payload is malformed and cannot be parsed: {exc}"
        ) from exc

    assert_fingerprints_compatible(
        ckpt_fp,
        meta.fingerprint,
        reference_label=f"{ckpt_label}'s training dataset",
        candidate_label="evaluation dataset",
    )

    ckpt_class_names = ckpt_state.get("class_names")
    if ckpt_class_names is None or list(ckpt_class_names) != list(meta.class_names):
        raise ValueError(
            f"{ckpt_label} class_names "
            f"{list(ckpt_class_names) if ckpt_class_names is not None else None} "
            f"do not match the dataset class order {list(meta.class_names)}; "
            "the model's logit columns would be scored against the wrong labels."
        )

    return meta
