"""Shared validation of the forward-kwargs dict that explanation methods accept.

Integrated Gradients, input replication, and structural deletion all take the
``inputs`` dict that :meth:`models.hier_model.HierGNNModel.forward` consumes.
Centralizing the contract here keeps the validation identical across them.
"""

from __future__ import annotations

from typing import Any

REQUIRED_INPUT_KEYS = frozenset(
    {
        "x",
        "edge_index",
        "edge_attr",
        "subgraph_node_batch",
        "super_edge_index",
        "super_edge_step",
        "subgraph_lead",
        "subgraph_exam",
        "batch_size",
    }
)
ALLOWED_INPUT_KEYS = REQUIRED_INPUT_KEYS | {"subgraph_beat"}


def validate_input_keys(inputs: dict[str, Any]) -> None:
    """Reject a forward-kwargs dict with missing or unknown keys.

    Shared by the explanation entry points and
    :func:`explain.replicate.replicate_inputs`.
    Raises ``ValueError`` with an actionable message; returns ``None`` on success.
    """
    missing = REQUIRED_INPUT_KEYS - inputs.keys()
    if missing:
        raise ValueError(f"inputs is missing required forward keys: {sorted(missing)}")
    unknown = inputs.keys() - ALLOWED_INPUT_KEYS
    if unknown:
        raise ValueError(f"inputs has unexpected keys (not forward kwargs): {sorted(unknown)}")
