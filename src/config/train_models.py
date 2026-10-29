"""Pydantic v2 model for the training-stack JSON config (``extra='forbid'``).

Mirrors ``configs/dataset/<preset>.json`` ergonomics: one JSON file under
``configs/train/`` is the source of truth for a run; CLI flags only override
discrete fields. Cross-field rules go in ``model_validator(mode='after')``
so typos and conflicting combinations fail at load time, not deep inside the
training loop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from models.hier_model import HierGNNModel


class BeatEncoderConfig(BaseModel):
    """GINE beat-graph encoder: message-passing depth and MLP width."""

    model_config = ConfigDict(extra="forbid")

    layers: int = Field(default=3, gt=0)
    mlp_hidden: int = Field(default=256, gt=0)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_channels: int = Field(default=128, gt=0)
    in_channels: int = Field(default=2, gt=0)
    edge_dim: int = Field(default=2, gt=0)
    beat_encoder: BeatEncoderConfig = Field(default_factory=BeatEncoderConfig)
    dropout: float = Field(default=0.10, ge=0.0, le=1.0)
    gatv2_heads: int = Field(default=4, gt=0)
    gatv2_layers: int = Field(default=2, gt=0)
    gatv2_max_edge_step: int = Field(default=1, gt=0)
    head_hidden_multiplier: float = Field(default=0.5, gt=0.0)
    head_dropout: float = Field(default=0.04, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _heads_divide_hidden(self) -> ModelConfig:
        if self.hidden_channels % self.gatv2_heads != 0:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must be divisible by "
                f"gatv2_heads ({self.gatv2_heads})"
            )
        return self

    def build(self, *, num_classes: int, num_leads: int) -> HierGNNModel:
        """Instantiate the :class:`HierGNNModel` this config describes.

        Flat fields pass through by name: those field names are the
        constructor parameter names. The nested ``beat_encoder`` block maps to
        the ``beat_encoder_*`` parameters, so the model keeps a flat
        ``hparams`` / checkpoint payload.

        Device placement and DDP wrapping stay with the caller
        (``.to(device)`` / :func:`training.ddp.wrap_model`): the config
        describes an architecture, not a runtime.
        """
        from models.hier_model import HierGNNModel

        return HierGNNModel(
            num_classes=num_classes,
            num_leads=num_leads,
            beat_encoder_layers=self.beat_encoder.layers,
            beat_encoder_mlp_hidden=self.beat_encoder.mlp_hidden,
            **self.model_dump(exclude={"beat_encoder"}),
        )


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=8, gt=0)
    eval_batch_size: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Batch size for the val/test loaders. Evaluation runs without "
            "backward, so it tolerates a much larger batch than training on "
            "the same VRAM. None = same as batch_size."
        ),
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        gt=0,
        description=(
            "Number of memory-limited micro-batches whose gradients are "
            "accumulated before a single optimizer step, so the effective "
            "batch size is ``batch_size * gradient_accumulation_steps``. Lets a "
            "GPU that only fits ``batch_size`` examples reproduce the effective "
            "batch size of a larger reference run. Micro-batch gradients are "
            "averaged by example count (not by micro-batch count), so a full "
            "window is mathematically equivalent to one forward/backward over "
            "the concatenated examples; the trailing partial window (when an "
            "epoch's micro-batch count is not a multiple of this value) still "
            "gets exactly one gradient-clip + optimizer step. Default 1 = no "
            "accumulation (standard one optimizer step per micro-batch)."
        ),
    )
    lr: float = Field(default=1.28e-4, gt=0.0)
    weight_decay: float = Field(default=6e-5, ge=0.0)
    early_stop_patience: int = Field(default=10, gt=0)
    scheduler_patience: int = Field(default=2, ge=0)
    scheduler_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    eval_val_every_epochs: int = Field(default=1, gt=0)
    grad_clip_norm: float | None = Field(default=1.0)

    @model_validator(mode="after")
    def _validate_grad_clip(self) -> OptimConfig:
        if self.grad_clip_norm is not None and (
            not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0
        ):
            raise ValueError(
                f"grad_clip_norm must be > 0 and finite, or None (off), got {self.grad_clip_norm}"
            )
        return self

    compile: bool = Field(
        default=False,
        description=(
            "torch.compile(dynamic=True) the model for training and eval. "
            "dynamic=True because batch node/edge counts vary; the first few "
            "steps pay one-off compilation latency."
        ),
    )
    log_every_steps: int = Field(default=50, gt=0)
    num_workers: int = Field(default=2, ge=0)
    shuffle_block_exams: int | None = Field(
        default=65536,
        gt=0,
        description=(
            "Locality window (in exams) for the train sampler's visit order: "
            "block order and within-block order are shuffled each epoch so HDF5 "
            "reads stay page-cache friendly. None = fully random order."
        ),
    )
    seed: int = 42


class DDPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    launcher: Literal["single", "ddp"] = "single"
    gpus: int = Field(default=1, gt=0)


class ThresholdConfig(BaseModel):
    """Decision threshold applied to every class at validation and test time."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(default=0.5, ge=0.0, le=1.0)


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["patient_hash"] = "patient_hash"
    val_fraction: float = Field(default=0.02, gt=0.0, lt=1.0)
    seed: int = 42


class SamplingConfig(BaseModel):
    """Per-epoch training-pool composition.

    Every epoch combines all positive exams with a fresh without-replacement
    sample of ``n_neg`` exams from the negative ("Undiagnosed") pool. This is
    the protocol from *An Empirical Comparison of ECG Foundation Models Across
    Modalities* (§3.3).
    """

    model_config = ConfigDict(extra="forbid")

    n_neg: int = Field(gt=0)


class SubgraphRepresentationConfig(BaseModel):
    """Load-time beat-graph topology and edge-feature policy.

    The defaults name the canonical loader behavior explicitly: use the
    stored visibility edges and derive ``delta_t``/``delta_y`` from
    their endpoints. Keeping the two factors together prevents a caller from
    changing one split or checkpoint consumer without the other.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    adjacency: Literal["stored_visibility", "temporal_chain"] = "stored_visibility"
    edge_attributes: Literal["derived", "zero"] = "derived"


class TrainPreset(BaseModel):
    """Top-level training-config preset."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    dataset_path: str
    test_dataset_path: str | None = None
    results_dir: str = "results"
    symmetrize_edges: bool = False
    subgraph_representation: SubgraphRepresentationConfig = Field(
        default_factory=SubgraphRepresentationConfig
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    ddp: DDPConfig = Field(default_factory=DDPConfig)
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    sampling: SamplingConfig

    @model_validator(mode="after")
    def _validate_split(self) -> TrainPreset:
        if self.split.strategy == "patient_hash" and self.test_dataset_path is None:
            raise ValueError(
                "split.strategy='patient_hash' requires test_dataset_path "
                "(patient_hash only produces train/val splits; test must come from a separate HDF5)"
            )
        return self


def load_train_preset(path: Path | str) -> TrainPreset:
    """Load and validate a JSON training preset (``extra='forbid'``)."""
    p = Path(path)
    with p.open("r") as f:
        data = json.load(f)
    return TrainPreset.model_validate(data)
