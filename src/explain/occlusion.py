"""Structural deletion (occlusion) for *causal* importance tests.

A feature-baseline perturbation "removes" a node by overwriting its features
with the Integrated-Gradients baseline while keeping its edges. That
perturbation scales with the node's feature magnitude, so a near-isoelectric
node (tiny amplitude) is barely perturbed no matter how much the model relies on
it. For a causal region test -- "does the 1dAVb logit depend on the PR-interval
nodes?" -- that bias is disqualifying, because the PR segment is near-isoelectric
*by definition*: zeroing an already-zero amplitude is a no-op, so a
feature-baseline deletion would report "PR does not matter" as an artifact.

This module deletes *structurally* instead, at two granularities:

- :func:`delete_nodes` removes a set of *nodes* (and their incident edges)
  while every subgraph survives with at least one node -- the within-beat
  region-occlusion use case (e.g. deleting just the PR-interval nodes of a
  beat, keeping the beat's subgraph itself). Each interior run of deleted
  nodes is then bridged: the last surviving node before the run is linked to
  the first surviving node after it, so a deleted region leaves the beat graph
  connected instead of split. Only node-level tensors (``x``,
  ``subgraph_node_batch``) and intra-subgraph edges change; the supergraph and
  every subgraph-level index tensor are untouched.
- :func:`delete_subgraphs` removes whole *subgraphs* (e.g. every beat of a
  lead) -- the operation :func:`delete_nodes` deliberately forbids, because
  emptying a subgraph of all its nodes leaves its pooled embedding undefined.
  Deleting a subgraph on purpose instead removes it, its nodes, its intra
  edges, and its supergraph edges, and relabels every subgraph-indexed tensor.
- :func:`delete_nodes_with_cascade` composes those two operations for region
  coalitions: any fully emptied lead-beat subgraph is removed deliberately,
  then partial node deletion is applied to the surviving subgraphs.

A node's (or subgraph's) presence in the graph does not depend on its
amplitude, so both interventions are amplitude-independent -- exactly what a
causal claim needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.utils import subgraph

from explain._inputs import validate_input_keys
from models.hier_model import HierGNNModel


class EmptySubgraphError(ValueError):
    """A structural deletion would leave at least one subgraph with no nodes.

    The subgraph encoder pools nodes into one embedding per subgraph; a subgraph
    with zero surviving nodes has no defined embedding and would desynchronize
    the subgraph-level index tensors. Callers should treat the affected
    (exam, region) as unusable rather than perturb around it.
    """


class EmptyExamError(ValueError):
    """A structural subgraph deletion would leave at least one exam with none.

    Mirrors :class:`EmptySubgraphError` one level up: an exam with zero
    surviving subgraphs has no defined lead or exam embedding (the intra- and
    inter-lead pools have nothing to pool over), so the affected exam is
    unusable rather than perturbed around.
    """


@dataclass(frozen=True)
class TargetScore:
    """Raw target logit and its sigmoid probability from one forward pass."""

    logit: float
    probability: float


@dataclass(frozen=True)
class CascadingNodeDeletionResult:
    """Inputs and audit counts after node deletion with empty-subgraph cascade."""

    inputs: dict[str, Any]
    removed_nodes: int
    removed_subgraphs: int
    removed_subgraph_ids: torch.Tensor


def _validated_node_remove_mask(
    model: HierGNNModel,
    inputs: dict[str, Any],
    remove_mask: torch.Tensor,
) -> torch.Tensor:
    """Validate a node-removal request and move its mask to the graph device."""
    validate_input_keys(inputs)
    x: torch.Tensor = inputs["x"]
    if remove_mask.shape != (x.shape[0],):
        raise ValueError(
            f"remove_mask must have shape ({x.shape[0]},) to align with nodes, "
            f"got {tuple(remove_mask.shape)}"
        )
    if remove_mask.dtype != torch.bool:
        raise ValueError(f"remove_mask must be bool, got {remove_mask.dtype}")
    return remove_mask.to(x.device)


def _emptied_subgraph_ids(
    subgraph_node_batch: torch.Tensor,
    keep_nodes: torch.Tensor,
    num_subgraphs: int,
) -> torch.Tensor:
    """Return subgraph ids with no nodes selected by ``keep_nodes``."""
    kept_per_subgraph = torch.bincount(subgraph_node_batch[keep_nodes], minlength=num_subgraphs)
    return (kept_per_subgraph == 0).nonzero(as_tuple=False).flatten()


def _gap_bridges(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    subgraph_node_batch: torch.Tensor,
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the edges that reconnect each deleted run, in pre-deletion node ids.

    Nodes of a beat graph are stored in time order, and the graph always holds
    the consecutive edges ``i -> i+1``. For each maximal run of deleted nodes with
    a surviving node ``p`` before it and ``q`` after it in the same subgraph, the
    bridge ``p -> q`` gets the sum of the consecutive-edge attributes from ``p``
    to ``q``. Edge attributes are endpoint differences (or zeros), so the sum
    equals the attribute the edge ``p -> q`` would have had. A graph that also
    stores reverse edges gets the reverse bridge ``q -> p`` the same way. A
    bridge that already exists as an edge is not added again. Runs at the start
    or end of a subgraph have nothing to bridge.
    """
    empty_index = edge_index.new_empty((2, 0))
    empty_attr = edge_attr.new_empty((0, edge_attr.shape[1]))
    kept = keep.nonzero(as_tuple=False).flatten()
    before, after = kept[:-1], kept[1:]
    gap = (after - before > 1) & (subgraph_node_batch[before] == subgraph_node_batch[after])
    before, after = before[gap], after[gap]
    if before.numel() == 0:
        return empty_index, empty_attr

    n = keep.numel()
    src, dst = edge_index[0], edge_index[1]
    existing = src * n + dst
    indices, attrs = [], []
    for direction in (1, -1):
        step = dst - src == direction
        if direction == -1 and not bool(step.any()):
            continue
        lower = src[step] if direction == 1 else dst[step]
        has_step = torch.zeros(n + 1, dtype=torch.long, device=keep.device)
        has_step[lower + 1] = 1
        step_attr = torch.zeros(
            (n + 1, edge_attr.shape[1]), dtype=torch.float64, device=edge_attr.device
        )
        step_attr[lower + 1] = edge_attr[step].double()
        covered = torch.cumsum(has_step, 0)
        summed = torch.cumsum(step_attr, 0)
        if bool((covered[after] - covered[before] != after - before).any()):
            raise ValueError(
                "cannot bridge a deleted run: a consecutive edge i -> i+1 is missing "
                "inside the run's subgraph"
            )
        attr = (summed[after] - summed[before]).to(edge_attr.dtype)
        pair = torch.stack([before, after]) if direction == 1 else torch.stack([after, before])
        new = ~torch.isin(pair[0] * n + pair[1], existing)
        indices.append(pair[:, new])
        attrs.append(attr[new])
    return torch.cat(indices, dim=1), torch.cat(attrs, dim=0)


def delete_nodes(
    model: HierGNNModel, inputs: dict[str, Any], remove_mask: torch.Tensor
) -> dict[str, Any]:
    """Return a forward-inputs dict with ``remove_mask`` nodes structurally removed.

    ``remove_mask`` is a boolean ``[N]`` tensor aligned with ``inputs['x']`` rows
    (True marks a node for deletion). Incident edges are dropped and the surviving
    nodes are relabelled to ``0..N'-1`` in their original order, so ``x`` and
    ``subgraph_node_batch`` stay row-aligned with the relabelled ``edge_index``.
    Every interior run of deleted nodes within a subgraph is bridged as described
    in :func:`_gap_bridges`. The returned dict shares the unchanged
    subgraph-level tensors with ``inputs`` (not copied); only the node-level
    entries are replaced.

    Raises :class:`EmptySubgraphError` if the deletion would empty any subgraph.
    """
    remove = _validated_node_remove_mask(model, inputs, remove_mask)
    x: torch.Tensor = inputs["x"]
    keep = ~remove
    sgnb: torch.Tensor = inputs["subgraph_node_batch"]
    n_sub = inputs["subgraph_lead"].shape[0]

    emptied = _emptied_subgraph_ids(sgnb, keep, n_sub)
    if emptied.numel():
        raise EmptySubgraphError(
            f"deletion would empty {emptied.numel()} of {n_sub} subgraph(s) "
            f"(ids {emptied.tolist()[:8]}{'...' if emptied.numel() > 8 else ''})"
        )

    new_edge_index, new_edge_attr = subgraph(
        keep,
        inputs["edge_index"],
        edge_attr=inputs["edge_attr"],
        relabel_nodes=True,
        num_nodes=x.shape[0],
    )
    bridge_index, bridge_attr = _gap_bridges(inputs["edge_index"], inputs["edge_attr"], sgnb, keep)
    new_id = torch.cumsum(keep.long(), 0) - 1
    new_edge_index = torch.cat([new_edge_index, new_id[bridge_index]], dim=1)
    new_edge_attr = torch.cat([new_edge_attr, bridge_attr], dim=0)

    new_inputs = dict(inputs)
    new_inputs["x"] = x[keep]
    new_inputs["subgraph_node_batch"] = sgnb[keep]
    new_inputs["edge_index"] = new_edge_index
    new_inputs["edge_attr"] = new_edge_attr
    return new_inputs


def delete_subgraphs(
    model: HierGNNModel, inputs: dict[str, Any], remove_mask: torch.Tensor
) -> dict[str, Any]:
    """Return a forward-inputs dict with ``remove_mask`` subgraphs structurally removed.

    ``remove_mask`` is a boolean ``[S]`` tensor aligned with the subgraph-level
    tensors (``subgraph_lead``, ``subgraph_exam``, ``subgraph_beat``). Every
    node belonging to a removed subgraph -- and its incident intra-subgraph
    edges -- is dropped; every supergraph edge with either endpoint in a
    removed subgraph is dropped; and the surviving subgraphs are relabelled to
    ``0..S'-1`` in their original order so every subgraph-indexed tensor
    (``subgraph_node_batch`` values, ``subgraph_lead``, ``subgraph_exam``,
    ``subgraph_beat``, ``super_edge_index``) stays consistent.

    This is the subgraph-granularity counterpart to :func:`delete_nodes`, which
    forbids leaving any subgraph with zero nodes -- exactly the operation this
    function performs on purpose. Use :func:`delete_nodes` to remove some nodes
    from within one or more subgraphs while every subgraph survives; use this
    function to remove whole subgraphs (e.g. every beat of a lead).

    Raises :class:`EmptyExamError` if the deletion would leave any exam with
    zero surviving subgraphs.
    """
    validate_input_keys(inputs)
    subgraph_lead: torch.Tensor = inputs["subgraph_lead"]
    subgraph_exam: torch.Tensor = inputs["subgraph_exam"]
    n_sub = subgraph_lead.shape[0]
    if remove_mask.shape != (n_sub,):
        raise ValueError(
            f"remove_mask must have shape ({n_sub},) to align with subgraphs, "
            f"got {tuple(remove_mask.shape)}"
        )
    if remove_mask.dtype != torch.bool:
        raise ValueError(f"remove_mask must be bool, got {remove_mask.dtype}")

    keep_sub = ~remove_mask.to(subgraph_lead.device)
    batch_size = int(inputs["batch_size"])

    kept_per_exam = torch.bincount(subgraph_exam[keep_sub], minlength=batch_size)
    emptied = (kept_per_exam == 0).nonzero(as_tuple=False).flatten()
    if emptied.numel():
        raise EmptyExamError(
            f"deletion would empty {emptied.numel()} of {batch_size} exam(s) of "
            f"every subgraph (exam ids {emptied.tolist()[:8]}"
            f"{'...' if emptied.numel() > 8 else ''})"
        )

    old_ids = keep_sub.nonzero(as_tuple=False).flatten()
    new_id_of_old = torch.full((n_sub,), -1, dtype=torch.long, device=subgraph_lead.device)
    new_id_of_old[old_ids] = torch.arange(old_ids.numel(), device=subgraph_lead.device)

    x: torch.Tensor = inputs["x"]
    sgnb: torch.Tensor = inputs["subgraph_node_batch"]
    node_keep = keep_sub[sgnb]

    new_edge_index, new_edge_attr = subgraph(
        node_keep,
        inputs["edge_index"],
        edge_attr=inputs["edge_attr"],
        relabel_nodes=True,
        num_nodes=x.shape[0],
    )

    super_edge_index: torch.Tensor = inputs["super_edge_index"]
    super_edge_step: torch.Tensor = inputs["super_edge_step"]
    if super_edge_index.numel() > 0:
        src, dst = super_edge_index[0], super_edge_index[1]
        edge_keep = keep_sub[src] & keep_sub[dst]
        new_super_edge_index = new_id_of_old[super_edge_index[:, edge_keep]]
        new_super_edge_step = super_edge_step[edge_keep]
    else:
        new_super_edge_index = super_edge_index
        new_super_edge_step = super_edge_step

    new_inputs = dict(inputs)
    new_inputs["x"] = x[node_keep]
    new_inputs["subgraph_node_batch"] = new_id_of_old[sgnb[node_keep]]
    new_inputs["edge_index"] = new_edge_index
    new_inputs["edge_attr"] = new_edge_attr
    new_inputs["subgraph_lead"] = subgraph_lead[keep_sub]
    new_inputs["subgraph_exam"] = subgraph_exam[keep_sub]
    new_inputs["super_edge_index"] = new_super_edge_index
    new_inputs["super_edge_step"] = new_super_edge_step
    subgraph_beat = inputs.get("subgraph_beat")
    if subgraph_beat is not None:
        new_inputs["subgraph_beat"] = subgraph_beat[keep_sub]
    return new_inputs


def delete_nodes_with_cascade(
    model: HierGNNModel, inputs: dict[str, Any], remove_mask: torch.Tensor
) -> CascadingNodeDeletionResult:
    """Delete nodes and deliberately remove any subgraphs left empty.

    The requested node mask remains the intervention source of truth. A
    subgraph whose every node is requested for deletion is first removed via
    :func:`delete_subgraphs`; requested nodes in surviving subgraphs are then
    removed via :func:`delete_nodes`. The operation raises
    :class:`EmptyExamError` if the cascade would remove every subgraph of any
    exam.

    ``removed_subgraph_ids`` contains the original pre-deletion subgraph ids,
    preserving an audit trail across relabelling.
    """
    remove = _validated_node_remove_mask(model, inputs, remove_mask)
    sgnb: torch.Tensor = inputs["subgraph_node_batch"]
    n_sub = inputs["subgraph_lead"].shape[0]
    keep = ~remove
    emptied_ids = _emptied_subgraph_ids(sgnb, keep, n_sub)

    current_inputs = inputs
    remaining_remove = remove
    if emptied_ids.numel():
        remove_subgraphs = torch.zeros(n_sub, dtype=torch.bool, device=sgnb.device)
        remove_subgraphs[emptied_ids] = True
        surviving_nodes = ~remove_subgraphs[sgnb]
        current_inputs = delete_subgraphs(model, inputs, remove_subgraphs)
        remaining_remove = remove[surviving_nodes]

    deleted_inputs = delete_nodes(model, current_inputs, remaining_remove)
    return CascadingNodeDeletionResult(
        inputs=deleted_inputs,
        removed_nodes=int(remove.sum().item()),
        removed_subgraphs=int(emptied_ids.numel()),
        removed_subgraph_ids=emptied_ids,
    )


def target_scores(
    model: HierGNNModel,
    inputs: dict[str, Any],
    *,
    target_classes: Sequence[int],
    target_exam: int = 0,
) -> dict[int, TargetScore]:
    """Return several class scores from one deterministic forward pass.

    Runs in eval mode with gradients disabled and restores the model's previous
    train/eval state before returning.
    """
    validate_input_keys(inputs)
    batch_size = int(inputs["batch_size"])
    classes = tuple(target_classes)
    if not 0 <= target_exam < batch_size:
        raise ValueError(f"target_exam {target_exam} out of range [0, {batch_size})")
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("target_classes must be non-empty and unique")
    if any(not 0 <= target_class < model.num_classes for target_class in classes):
        raise ValueError(f"target_classes must lie in [0, {model.num_classes})")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(**inputs).logits[target_exam]
        return {
            target_class: TargetScore(
                logit=float(logits[target_class]),
                probability=float(torch.sigmoid(logits[target_class])),
            )
            for target_class in classes
        }
    finally:
        model.train(was_training)
