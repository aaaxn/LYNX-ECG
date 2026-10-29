"""Unit tests for ecg_checkpoint_io's payload tree-walk primitives.

``map_tensor_leaves`` is the canonical walker over the checkpoint payload shape
(tensors + primitive ``dict``/``list``/``tuple`` containers). These tests pin
its structural contract on CPU so it is exercised in CI regardless of GPU
availability.
"""

from __future__ import annotations

import pytest
import torch

from ecg_checkpoint_io import map_tensor_leaves

pytestmark = pytest.mark.unit


def _nested_payload() -> dict:
    return {
        "model_state": {"weight": torch.ones(2), "bias": torch.zeros(3)},
        "nested_state": {
            "state": {0: {"exp_avg": torch.full((2,), 2.0), "step": 7}},
            "param_groups": [{"lr": 1e-3, "params": [0]}],
        },
        "history": [torch.tensor([1.0]), (torch.tensor([2.0]), "label")],
        "epoch": 3,
        "note": None,
    }


class TestMapTensorLeaves:
    def test_applies_fn_to_every_tensor(self):
        payload = _nested_payload()
        mapped = map_tensor_leaves(payload, lambda tensor: tensor + 1)
        assert torch.equal(mapped["model_state"]["weight"], torch.full((2,), 2.0))
        assert torch.equal(mapped["model_state"]["bias"], torch.ones(3))
        assert torch.equal(mapped["nested_state"]["state"][0]["exp_avg"], torch.full((2,), 3.0))
        assert torch.equal(mapped["history"][0], torch.tensor([2.0]))
        assert torch.equal(mapped["history"][1][0], torch.tensor([3.0]))

    def test_preserves_container_shapes_and_non_tensor_leaves(self):
        payload = _nested_payload()
        mapped = map_tensor_leaves(payload, lambda tensor: tensor)
        assert isinstance(mapped, dict)
        assert isinstance(mapped["nested_state"]["param_groups"], list)
        assert isinstance(mapped["history"][1], tuple)
        assert mapped["nested_state"]["state"][0]["step"] == 7
        assert mapped["epoch"] == 3
        assert mapped["note"] is None
        assert mapped["history"][1][1] == "label"

    def test_does_not_mutate_the_original(self):
        payload = _nested_payload()
        original_weight = payload["model_state"]["weight"].clone()
        map_tensor_leaves(payload, lambda tensor: tensor * 10)
        assert torch.equal(payload["model_state"]["weight"], original_weight)

    def test_empty_containers_pass_through(self):
        mapped = map_tensor_leaves({"d": {}, "l": [], "t": ()}, lambda tensor: tensor)
        assert mapped == {"d": {}, "l": [], "t": ()}

    def test_bare_tensor_root(self):
        assert torch.equal(
            map_tensor_leaves(torch.zeros(2), lambda tensor: tensor + 5), torch.full((2,), 5.0)
        )
