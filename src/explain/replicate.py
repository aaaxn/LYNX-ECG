"""Replicate a forward-kwargs dict as independent batch copies.

Integrated Gradients scores many variants of the SAME exam (its interpolation
points). The attribution docstring (:mod:`explain.attribution`) states the
property that makes batching those variants safe: because the model pools per
exam, gradients (and, a fortiori, values) of one exam's logit are structurally
independent of another exam's nodes. Tiling the SAME graph structure ``n``
times as disjoint batch copies and only varying ``x`` per copy therefore scores
``n`` variants in one forward pass, with copy ``i``'s output identical to what
a standalone forward on copy ``i`` alone would produce.
"""

from __future__ import annotations

from typing import Any

import torch

from explain._inputs import validate_input_keys


def replicate_inputs(inputs: dict[str, Any], n: int) -> dict[str, Any]:
    """Tile a forward-kwargs dict ``n`` times as independent batch copies.

    Equivalent to ``Batch.from_data_list([data] * n)`` for the keys the
    explain API uses: node/subgraph/exam index tensors are offset per copy so
    the copies are structurally disjoint; feature tensors are tiled. The
    caller then overwrites ``x`` rows per copy (rows ``i*N:(i+1)*N`` belong to
    copy ``i``, where ``N`` is the node count of the original input).
    Gradients/probabilities of copy ``i``'s exams are unaffected by other
    copies (per-exam pooling => cross-copy gradients are structurally zero;
    see :mod:`explain.attribution`'s module docstring).

    Offsets: node ids (``edge_index``, ``subgraph_node_batch``) are shifted by
    ``i * N`` (node count); subgraph ids (``super_edge_index``) by ``i * G``
    (subgraph count); exam ids (``subgraph_exam``) by ``i * batch_size``.
    Feature/attribute tensors (``edge_attr``, ``super_edge_step``,
    ``subgraph_lead``, ``subgraph_beat``) are tiled unchanged since they carry
    no cross-copy identity. ``batch_size`` becomes ``n * batch_size``.

    ``inputs`` must contain exactly the forward-kwargs keys
    :func:`explain._inputs.validate_input_keys` accepts (``x``,
    ``edge_index``, ``edge_attr``, ``subgraph_node_batch``,
    ``super_edge_index``, ``super_edge_step``, ``subgraph_lead``,
    ``subgraph_exam``, ``batch_size``, and optionally ``subgraph_beat``).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    validate_input_keys(inputs)

    if n == 1:
        replicated = dict(inputs)
        replicated["batch_size"] = int(inputs["batch_size"])
        return replicated

    x: torch.Tensor = inputs["x"]
    edge_index: torch.Tensor = inputs["edge_index"]
    edge_attr: torch.Tensor = inputs["edge_attr"]
    subgraph_node_batch: torch.Tensor = inputs["subgraph_node_batch"]
    super_edge_index: torch.Tensor = inputs["super_edge_index"]
    super_edge_step: torch.Tensor = inputs["super_edge_step"]
    subgraph_lead: torch.Tensor = inputs["subgraph_lead"]
    subgraph_exam: torch.Tensor = inputs["subgraph_exam"]
    batch_size = int(inputs["batch_size"])

    num_nodes = x.shape[0]
    num_subgraphs = subgraph_lead.shape[0]

    replicated: dict[str, Any] = {
        "x": x.repeat(n, 1),
        "edge_index": torch.cat([edge_index + i * num_nodes for i in range(n)], dim=1),
        "edge_attr": edge_attr.repeat(n, 1),
        "subgraph_node_batch": torch.cat(
            [subgraph_node_batch + i * num_subgraphs for i in range(n)]
        ),
        "super_edge_index": torch.cat(
            [super_edge_index + i * num_subgraphs for i in range(n)], dim=1
        ),
        "super_edge_step": super_edge_step.repeat(n),
        "subgraph_lead": subgraph_lead.repeat(n),
        "subgraph_exam": torch.cat([subgraph_exam + i * batch_size for i in range(n)]),
        "batch_size": n * batch_size,
    }
    if "subgraph_beat" in inputs:
        subgraph_beat = inputs["subgraph_beat"]
        replicated["subgraph_beat"] = None if subgraph_beat is None else subgraph_beat.repeat(n)
    return replicated
