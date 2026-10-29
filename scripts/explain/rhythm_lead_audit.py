"""Rhythm, rate, lead, and regional evidence for all six CODE-6 classes.

The regional PR-QRS-T audit covers only the three morphology classes. This
workflow asks the complementary questions for every class, with emphasis on
the rhythm classes (SB, AF, ST):

- ``descriptors``: measured heart rate and RR irregularity per exam, next to
  the intact six-class logits. No model intervention.
- ``interventions``: beat-sequence counterfactuals. Each variant rebuilds the
  exam graph from its own lead-beat subgraphs:
  ``isolated`` removes every beat-to-beat supergraph edge; ``shuffled``
  permutes beat order within each lead; ``template`` replaces every beat of a
  lead with copies of its median-duration beat; ``template_rr`` does the same
  but restores each beat's own duration; ``rr_regularized`` rescales
  each interior beat's time axis to the lead's median beat duration;
  ``rr_irregularized`` imposes beat-duration ratios drawn from the dataset's
  AF-positive exams; ``rate_sweep`` rescales every beat's time axis so the
  exam's heart rate becomes each target value.
- ``leads``: exact 12-lead Shapley allocation of each class logit. The model
  has no cross-lead edges, so each lead embedding is independent of the other
  leads. On the first exams the script checks every lead subset against
  structural lead deletion before it evaluates all subsets from cached lead
  embeddings.
- ``regions``: the eight-state PR-QRS-T deletion game with Prominence
  delineation, read for all six classes on the exams the delineation QC
  covers (all exams on CODE-Test, label-positive exams on PTB-XL).
- ``diastolic``: interval-preserving rate change; only the time before P onset
  and after T offset is resized. It needs the all-exam delineation QC.

Time rescaling changes only the edge ``delta_t`` attribute. The node feature
``t_norm_beat`` is relative to the beat duration and does not change when a
whole beat is stretched. Uniform stretching also stretches PR, QRS, and QT, so
the rate sweep is a model-sensitivity test, not a physiological simulation.

Usage:
    uv run python scripts/explain/rhythm_lead_audit.py \
        --run 55=results/code100_tp_s55/best_f1_val.pt \
        --dataset graph_datasets/codetest/DATASET.h5 \
        --delineation-qc reports/region_delineation_qc/codetest_region_delineation_qc.json \
        --device cuda --output reports/rhythm_lead_audit/codetest_s55.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

from cli.run_arguments import seed_path_type
from dataset.graph_ops import build_visibility_graph
from dataset.schema import decode_vlen_str
from ecg_artifact_io import sha256_file
from explain.coalition import (
    EIGHT_STATE_ORDER,
    evaluate_eight_state_coalitions_multiclass,
    exact_regional_shapley,
)
from explain.common_support import AvailabilityRule, prominence_intervals_for_exam
from explain.delineation import DelineationMethod
from explain.delineation_qc import FiducialSamples
from explain.delineation_workflow import load_delineation_qc_report
from explain.ecg_timing import exam_timing
from explain.occlusion import delete_subgraphs
from explain.region_intervention import graph_local_region_masks
from models.hier_model import HierGNNModel
from training.dataset import HierData, _build_super_edges
from training.inference import (
    batch_to_forward_kwargs,
    load_checkpoint_and_dataset,
    resolve_device,
    single_exam_inputs,
)

BLOCKS = ("descriptors", "interventions", "leads", "regions", "diastolic")
RATE_TARGETS_BPM = tuple(range(30, 185, 5))
DIASTOLIC_TARGETS_BPM = tuple(range(40, 155, 5))
MIN_DIASTOLE_S = 0.02
MAX_UNCHANGED_FRACTION = 0.10
MAX_GAP_SAMPLES = 24
N_SHUFFLES = 5
N_IRREGULAR_DRAWS = 5
LEAD_CHECK_EXAMS = 5


@dataclass
class BeatPiece:
    """One lead-beat subgraph with local node indices."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    lead: int
    beat_index: int
    duration_s: float
    interior: bool


def decompose_exam(data: HierData, beat_meta: np.ndarray) -> list[BeatPiece]:
    """Split one exam item into its subgraphs, in HDF5 (lead, beat) order.

    ``beat_meta`` rows are ``(duration_s, boundary_type)`` per subgraph.
    """
    sgnb = data.subgraph_node_batch
    n_sub = int(data.subgraph_lead.numel())
    counts = torch.bincount(sgnb, minlength=n_sub)
    node_ptr = torch.zeros(n_sub + 1, dtype=torch.long)
    node_ptr[1:] = torch.cumsum(counts, 0)
    edge_beat = sgnb[data.edge_index[0]]
    order = torch.argsort(edge_beat, stable=True)
    edge_index = data.edge_index[:, order]
    edge_attr = data.edge_attr[order]
    edge_counts = torch.bincount(edge_beat, minlength=n_sub)
    edge_ptr = torch.zeros(n_sub + 1, dtype=torch.long)
    edge_ptr[1:] = torch.cumsum(edge_counts, 0)
    pieces = []
    for s in range(n_sub):
        n0, n1 = int(node_ptr[s]), int(node_ptr[s + 1])
        e0, e1 = int(edge_ptr[s]), int(edge_ptr[s + 1])
        pieces.append(
            BeatPiece(
                x=data.x[n0:n1],
                edge_index=edge_index[:, e0:e1] - n0,
                edge_attr=edge_attr[e0:e1],
                lead=int(data.subgraph_lead[s]),
                beat_index=int(data.subgraph_beat[s]),
                duration_s=float(beat_meta[s, 0]),
                interior=int(beat_meta[s, 1]) == 0,
            )
        )
    return pieces


def assemble_exam(
    pieces: list[BeatPiece],
    *,
    time_scale: list[float] | None = None,
    chain: bool = True,
) -> HierData:
    """Rebuild one exam item from pieces; list order defines beat adjacency.

    Pieces of one lead must be contiguous. ``time_scale`` multiplies each
    piece's edge ``delta_t`` (column 0). ``chain=False`` drops every
    beat-to-beat supergraph edge.
    """
    xs, eis, eas = [], [], []
    offset = 0
    for i, piece in enumerate(pieces):
        xs.append(piece.x)
        eis.append(piece.edge_index + offset)
        attr = piece.edge_attr
        if time_scale is not None and time_scale[i] != 1.0:
            attr = attr.clone()
            attr[:, 0] *= time_scale[i]
        eas.append(attr)
        offset += piece.x.shape[0]
    counts = torch.tensor([p.x.shape[0] for p in pieces], dtype=torch.long)
    lead = np.array([p.lead for p in pieces], dtype=np.int64)
    if np.any(np.diff(lead) < 0):
        raise ValueError("pieces of one lead must be contiguous and lead-ordered")
    position = np.arange(len(pieces), dtype=np.int64)
    if chain:
        super_ei, super_step = _build_super_edges(lead, position)
    else:
        super_ei = np.empty((2, 0), dtype=np.int64)
        super_step = np.empty((0,), dtype=np.int64)
    return HierData(
        x=torch.cat(xs),
        edge_index=torch.cat(eis, dim=1),
        edge_attr=torch.cat(eas),
        subgraph_node_batch=torch.repeat_interleave(torch.arange(len(pieces)), counts),
        super_edge_index=torch.from_numpy(super_ei).long(),
        super_edge_step=torch.from_numpy(super_step).long(),
        subgraph_lead=torch.from_numpy(lead).long(),
        subgraph_exam=torch.zeros(len(pieces), dtype=torch.long),
        subgraph_beat=torch.from_numpy(position).long(),
    )


def lead_groups(pieces: list[BeatPiece]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i, piece in enumerate(pieces):
        groups.setdefault(piece.lead, []).append(i)
    return groups


def lead_median_duration(pieces: list[BeatPiece], idx: list[int]) -> float:
    interior = [pieces[i].duration_s for i in idx if pieces[i].interior]
    return float(np.median(interior if interior else [pieces[i].duration_s for i in idx]))


def template_index(pieces: list[BeatPiece], idx: list[int]) -> int:
    """The lead's interior beat closest to its median beat duration."""
    med = lead_median_duration(pieces, idx)
    candidates = [i for i in idx if pieces[i].interior] or idx
    return min(candidates, key=lambda i: (abs(pieces[i].duration_s - med), i))


def irregular_scale(
    pieces: list[BeatPiece], af_ratios: np.ndarray, rng: np.random.Generator
) -> list[float]:
    """Per-piece ``delta_t`` scale that imposes AF beat-duration ratios.

    One ratio per beat position, shared by all leads, so simultaneous beats of
    different leads receive the same imposed duration.
    """
    max_beat = max(p.beat_index for p in pieces) + 1
    ratio = rng.choice(af_ratios, size=max_beat, replace=True)
    scale = [1.0] * len(pieces)
    for idx in lead_groups(pieces).values():
        med = lead_median_duration(pieces, idx)
        for i in idx:
            if pieces[i].interior:
                scale[i] = ratio[pieces[i].beat_index] * med / pieces[i].duration_s
    return scale


def rhythm_descriptors(r_peak: np.ndarray, lead: np.ndarray, beat: np.ndarray, fs: float) -> dict:
    """Heart rate and RR irregularity from the stored per-lead R anchors.

    RR intervals use only consecutive beat indices, so QC-rejected beats do not
    merge two intervals. Each lead gives one value; the exam value is the
    median across leads.
    """
    hr, cv, rmssd = [], [], []
    for lead_id in np.unique(lead):
        sel = lead == lead_id
        rp, bi = r_peak[sel], beat[sel]
        consecutive = np.diff(bi) == 1
        rr = (np.diff(rp) / fs)[consecutive]
        if rr.size < 2:
            continue
        hr.append(60.0 / float(np.median(rr)))
        cv.append(float(np.std(rr) / np.mean(rr)))
        succ = np.diff(rr)[consecutive[1:] & consecutive[:-1]] if rr.size > 2 else np.array([])
        if succ.size:
            rmssd.append(float(np.sqrt(np.mean(succ**2)) / np.mean(rr)))
    return {
        "hr_bpm": float(np.median(hr)) if hr else None,
        "rr_cv": float(np.median(cv)) if cv else None,
        "rr_nrmssd": float(np.median(rmssd)) if rmssd else None,
        "n_rhythm_leads": len(hr),
    }


@torch.no_grad()
def batched_logits(
    model: HierGNNModel, items: list[HierData], device: torch.device, max_nodes: int = 200_000
) -> np.ndarray:
    """fp32 logits for a list of exam items, chunked by node count."""
    out, chunk, nodes = [], [], 0
    for item in [*items, None]:
        if item is not None and (not chunk or nodes + item.x.shape[0] <= max_nodes):
            chunk.append(item)
            nodes += item.x.shape[0]
            continue
        if chunk:
            batch = Batch.from_data_list(chunk).to(device)
            out.append(model(**batch_to_forward_kwargs(batch)).logits.float().cpu().numpy())
        if item is not None:
            chunk, nodes = [item], item.x.shape[0]
    return np.concatenate(out, axis=0)


def intervention_variants(
    pieces: list[BeatPiece],
    hr_bpm: float | None,
    af_ratios: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[str], list[HierData]]:
    names: list[str] = ["intact", "isolated"]
    items: list[HierData] = [assemble_exam(pieces), assemble_exam(pieces, chain=False)]
    groups = lead_groups(pieces)

    for k in range(N_SHUFFLES):
        order = [i for idx in groups.values() for i in rng.permutation(idx).tolist()]
        names.append(f"shuffled_{k}")
        items.append(assemble_exam([pieces[i] for i in order]))

    template: list[BeatPiece] = []
    template_rr_scale: list[float] = []
    for idx in groups.values():
        pick = template_index(pieces, idx)
        template.extend(pieces[pick] for _ in idx)
        template_rr_scale.extend(
            pieces[i].duration_s / pieces[pick].duration_s if pieces[i].interior else 1.0
            for i in idx
        )
    names.append("template")
    items.append(assemble_exam(template))
    names.append("template_rr")
    items.append(assemble_exam(template, time_scale=template_rr_scale))

    reg_scale = [1.0] * len(pieces)
    for idx in groups.values():
        med = lead_median_duration(pieces, idx)
        for i in idx:
            if pieces[i].interior:
                reg_scale[i] = med / pieces[i].duration_s
    names.append("rr_regularized")
    items.append(assemble_exam(pieces, time_scale=reg_scale))

    for k in range(N_IRREGULAR_DRAWS):
        names.append(f"rr_irregularized_{k}")
        items.append(assemble_exam(pieces, time_scale=irregular_scale(pieces, af_ratios, rng)))

    if hr_bpm is not None:
        for target in RATE_TARGETS_BPM:
            s = hr_bpm / target
            names.append(f"rate_{target}")
            items.append(assemble_exam(pieces, time_scale=[s] * len(pieces)))
    return names, items


def beat_core(
    fiducials: tuple[int | None, int | None] | None,
    start_sample: int,
    r_local: float,
    duration_s: float,
    fs: float,
) -> tuple[float, float, bool]:
    """Beat-local ``(P onset, T offset)`` in seconds and whether it is delineated.

    The fixed fallback core covers PR, QRS, and QT at ordinary rates; it is
    used only where Prominence did not return an ordered P onset and T offset.
    """
    if fiducials is not None and None not in fiducials:
        c0 = (fiducials[0] - start_sample) / fs
        c1 = (fiducials[1] - start_sample) / fs
        if 0.0 <= c0 < r_local < c1 <= duration_s:
            return c0, c1, True
    return max(0.0, r_local - 0.25), min(duration_s, r_local + 0.45), False


def diastolic_piece(
    piece: BeatPiece, c0: float, c1: float, scale: float, fs: float
) -> BeatPiece | None:
    """Resize only the pre-P and post-T parts of one beat by ``scale`` of its duration.

    Node times inside ``[c0, c1]`` keep their spacing, so PR, QRS, and QT do not
    change. Support nodes are refilled with the build rule (no gap above
    ``MAX_GAP_SAMPLES``, linear amplitude), and natural visibility is rebuilt on
    ordinal positions as in the dataset build. Returns ``None`` when the
    beat has no diastole to resize, or when compression would leave less than
    ``MIN_DIASTOLE_S`` (or less than the beat already has).
    """
    dur = piece.duration_s
    diastole = c0 + (dur - c1)
    new_diastole = diastole + dur * (scale - 1.0)
    if scale == 1.0:
        k = 1.0
    elif diastole <= 0 or new_diastole < min(MIN_DIASTOLE_S, diastole):
        return None
    else:
        k = new_diastole / diastole
    t = piece.x[:, 1].double().numpy() * dur
    amp = piece.x[:, 0].double().numpy()
    pre = c0 * k
    w = np.where(t < c0, t * k, np.where(t <= c1, pre + (t - c0), pre + (c1 - c0) + (t - c1) * k))
    new_dur = pre + (c1 - c0) + (dur - c1) * k
    return rebuild_piece(piece, w, amp, new_dur, fs)


def rebuild_piece(
    piece: BeatPiece, t: np.ndarray, amp: np.ndarray, duration_s: float, fs: float
) -> BeatPiece:
    """Beat subgraph from node times and amplitudes, with the dataset build rules.

    Support nodes are refilled so no gap exceeds ``MAX_GAP_SAMPLES`` (linear
    amplitude), and natural visibility is rebuilt on ordinal positions.
    """
    ts, amps = [t[0]], [amp[0]]
    for i in range(1, t.size):
        gap = round((t[i] - t[i - 1]) * fs)
        if gap > MAX_GAP_SAMPLES:
            n_ins = (gap - 1) // MAX_GAP_SAMPLES
            for j in range(1, n_ins + 1):
                f = j / (n_ins + 1)
                ts.append(t[i - 1] + f * (t[i] - t[i - 1]))
                amps.append(amp[i - 1] + f * (amp[i] - amp[i - 1]))
        ts.append(t[i])
        amps.append(amp[i])
    ts_a, amps_a = np.asarray(ts), np.asarray(amps)
    ei = build_visibility_graph(amps_a)
    attr = np.stack([ts_a[ei[1]] - ts_a[ei[0]], amps_a[ei[1]] - amps_a[ei[0]]], axis=1)
    x = np.stack([amps_a, ts_a / duration_s], axis=1)
    return BeatPiece(
        x=torch.from_numpy(x).float(),
        edge_index=torch.from_numpy(np.ascontiguousarray(ei)).long(),
        edge_attr=torch.from_numpy(attr).float(),
        lead=piece.lead,
        beat_index=piece.beat_index,
        duration_s=float(duration_s),
        interior=piece.interior,
    )


def diastolic_variants(
    pieces: list[BeatPiece],
    cores: list[tuple[float, float]],
    hr_bpm: float | None,
    fs: float,
) -> tuple[list[str], list[HierData | None], list[float]]:
    """Identity check plus one exam per target rate; ``None`` marks an infeasible target.

    Also returns, per variant, the fraction of interior beats left unchanged.
    """
    names: list[str] = ["identity"]
    scales: list[float] = [1.0]
    if hr_bpm is not None:
        for target in DIASTOLIC_TARGETS_BPM:
            names.append(f"rate_{target}")
            scales.append(hr_bpm / target)
    items: list[HierData | None] = []
    unchanged: list[float] = []
    n_interior = max(1, sum(p.interior for p in pieces))
    for scale in scales:
        new_pieces: list[BeatPiece] = []
        kept = 0
        for piece, (c0, c1) in zip(pieces, cores, strict=True):
            warped = diastolic_piece(piece, c0, c1, scale, fs)
            if warped is None:
                kept += piece.interior
                warped = piece
            new_pieces.append(warped)
        frac = kept / n_interior
        unchanged.append(frac)
        items.append(assemble_exam(new_pieces) if frac <= MAX_UNCHANGED_FRACTION else None)
    return names, items, unchanged


@torch.no_grad()
def lead_embeddings(model: HierGNNModel, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-lead embeddings ``[L, H]`` and presence mask ``[L]`` for one exam."""
    sub = model.subgraph_encoder(
        inputs["x"],
        inputs["edge_index"],
        inputs["edge_attr"],
        inputs["subgraph_node_batch"],
        num_subgraphs=inputs["subgraph_lead"].size(0),
    )
    super_attr = (
        model.supergraph_encoder.embed_step(inputs["super_edge_step"])
        if inputs["super_edge_index"].numel() > 0
        else None
    )
    h, _ = model.supergraph_encoder(
        sub, inputs["super_edge_index"], super_attr, inputs["subgraph_lead"]
    )
    lead_emb, present, _ = model._intra_lead_pool(
        h, inputs["subgraph_exam"], inputs["subgraph_lead"], 1
    )
    return lead_emb[0], present[0]


@torch.no_grad()
def coalition_logits(
    model: HierGNNModel, lead_emb: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Logits ``[M, C]`` for lead coalitions ``masks`` ``[M, L]`` (bool)."""
    emb = lead_emb.unsqueeze(0).expand(masks.shape[0], -1, -1)
    pooled, _ = model._inter_lead_pool(emb, masks)
    return model.head(pooled)


def exact_lead_shapley(model: HierGNNModel, lead_emb: torch.Tensor, present: torch.Tensor) -> dict:
    players = present.nonzero().flatten().tolist()
    n = len(players)
    n_leads = present.numel()
    codes = torch.arange(2**n, device=lead_emb.device)
    bits = ((codes.unsqueeze(1) >> torch.arange(n, device=lead_emb.device)) & 1).bool()
    masks = torch.zeros((2**n, n_leads), dtype=torch.bool, device=lead_emb.device)
    masks[:, players] = bits
    values = coalition_logits(model, lead_emb, masks).double()
    size = bits.sum(1)
    weight = torch.tensor(
        [math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)],
        dtype=torch.float64,
        device=lead_emb.device,
    )
    shapley = torch.zeros((n_leads, values.shape[1]), dtype=torch.float64, device=lead_emb.device)
    for j, lead in enumerate(players):
        without = ~bits[:, j]
        base = codes[without]
        gain = values[base | (1 << j)] - values[base]
        shapley[lead] = (weight[size[without]].unsqueeze(1) * gain).sum(0)
    full = values[-1]
    empty = values[0]
    alone = torch.full((n_leads, values.shape[1]), float("nan"), dtype=torch.float64)
    loo = torch.full_like(alone, float("nan"))
    for j, lead in enumerate(players):
        alone[lead] = values[1 << j].cpu()
        loo[lead] = values[(2**n - 1) ^ (1 << j)].cpu()
    efficiency = float((shapley.sum(0) - (full - empty)).abs().max())
    return {
        "present_leads": players,
        "full_logits": full.cpu().tolist(),
        "empty_logits": empty.cpu().tolist(),
        "shapley_logit": shapley.cpu().tolist(),
        "lead_alone_logits": alone.tolist(),
        "leave_one_out_logits": loo.tolist(),
        "efficiency_error": efficiency,
    }


def check_lead_factorization(
    model: HierGNNModel, inputs: dict, lead_emb: torch.Tensor, present: torch.Tensor
) -> float:
    """Max |logit| gap between cached-embedding coalitions and structural lead deletion.

    Every non-empty subset of the present leads is checked. The empty subset has
    no graph to delete, and its cached value is the zero exam embedding.
    """
    players = present.nonzero().flatten().tolist()
    n = len(players)
    codes = torch.arange(1, 2**n, device=lead_emb.device)
    bits = ((codes.unsqueeze(1) >> torch.arange(n, device=lead_emb.device)) & 1).bool()
    masks = torch.zeros((codes.numel(), present.numel()), dtype=torch.bool, device=lead_emb.device)
    masks[:, players] = bits
    cached = coalition_logits(model, lead_emb, masks)
    worst = 0.0
    for mask, cached_logits in zip(masks, cached, strict=True):
        remove = ~mask[inputs["subgraph_lead"]]
        with torch.no_grad():
            direct = model(**delete_subgraphs(model, inputs, remove)).logits[0]
        worst = max(worst, float((cached_logits - direct).abs().max()))
    return worst


def regional_block(model, inputs, h5, exam_index, graph_exam, records, candidate, n_classes, fs):
    g0 = int(np.searchsorted(graph_exam, exam_index, side="left"))
    g1 = int(np.searchsorted(graph_exam, exam_index, side="right"))
    timing = exam_timing(h5, exam_index, graph_exam_index=graph_exam)
    out = {}
    for rule in (AvailabilityRule.BEAT_COMPLETE, AvailabilityRule.PER_REGION):
        intervals = prominence_intervals_for_exam(
            records,
            graph_indices=tuple(range(g0, g1)),
            beat_r_peak_t=timing.beat_r_peak_t,
            sampling_rate=fs,
            selected_prominence_candidate=candidate,
            availability_rule=rule,
        )
        masks = graph_local_region_masks(timing.node_t, timing.node_beat, intervals)
        evaluations = evaluate_eight_state_coalitions_multiclass(
            model, inputs, masks, target_classes=tuple(range(n_classes))
        )
        states = {
            name: [
                None
                if evaluations[c].states[k].score is None
                else evaluations[c].states[k].score.logit
                for c in range(n_classes)
            ]
            for k, name in enumerate(EIGHT_STATE_ORDER)
        }
        shapley = []
        for c in range(n_classes):
            ev = evaluations[c]
            if any(state.score is None for state in ev.states):
                shapley.append(None)
                continue
            alloc = exact_regional_shapley(ev.score_values("logit"))
            shapley.append(dict(alloc.contributions))
        available = intervals.available
        out[str(rule)] = {
            "region_available_units": available.sum(0).tolist(),
            "beat_complete_units": int(available.all(1).sum()),
            "n_units": int(available.shape[0]),
            "state_logits": states,
            "shapley_logit": shapley,
        }
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--run", required=True, type=seed_path_type("SEED=CHECKPOINT"), metavar="SEED=CHECKPOINT"
    )
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--delineation-qc", type=Path, default=None, help="required for the regions block"
    )
    p.add_argument("--blocks", nargs="+", choices=BLOCKS, default=list(BLOCKS))
    p.add_argument(
        "--intervention-exams",
        choices=("all", "rhythm_positive_plus_sample"),
        default="all",
        help="PTB-XL: limit interventions to SB/AF/ST positives plus a random sample",
    )
    p.add_argument("--negative-sample", type=int, default=2000)
    p.add_argument(
        "--lead-check-exams",
        type=int,
        default=LEAD_CHECK_EXAMS,
        help="exams whose lead subsets are all checked against structural lead deletion",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N exams only")
    return p


def main() -> None:
    args = _parser().parse_args()
    if {"regions", "diastolic"} & set(args.blocks) and args.delineation_qc is None:
        raise SystemExit("the regions and diastolic blocks need --delineation-qc")
    device = resolve_device(args.device)
    dataset_path = args.dataset.resolve()
    ckpt = args.run.path.resolve()
    bundle = load_checkpoint_and_dataset(ckpt, dataset_path, device)
    model: HierGNNModel = bundle.model
    model.eval()
    classes = list(bundle.meta.class_names)
    if list(bundle.dataset.edge_attr_names) != ["delta_t", "delta_y"]:
        raise SystemExit(f"unexpected edge attributes {bundle.dataset.edge_attr_names}")

    records_by_exam: dict[int, list] = {}
    candidate = None
    fiducials_by_graph: dict[int, FiducialSamples] = {}
    region_exams: set[int] = set()
    if {"regions", "diastolic"} & set(args.blocks):
        report = load_delineation_qc_report(args.delineation_qc.resolve())
        if Path(report.dataset.path) != dataset_path:
            raise SystemExit("delineation QC was built on a different HDF5")
        candidate = report.config.selected_prominence_candidate
        region_exams = set(report.dataset.analyzed_exam_indices)
        for record in report.records:
            if record.method is DelineationMethod.PROMINENCE and record.candidate == candidate:
                records_by_exam.setdefault(record.exam_idx, []).append(record)
                fiducials_by_graph[record.graph_idx] = record.fiducials
        del report

    with bundle, h5py.File(dataset_path, "r") as h5:
        fs = float(h5.attrs["sampling_rate"])
        labels = np.asarray(h5["exam_labels"][:], dtype=np.int64)
        exam_ids = h5["exam_id"][:]
        graph_exam = np.asarray(h5["graph_exam_index"][:], dtype=np.int64)
        g_lead = np.asarray(h5["graph_lead"][:], dtype=np.int64)
        g_beat = np.asarray(h5["graph_beat_index"][:], dtype=np.int64)
        g_r = np.asarray(h5["graph_r_peak_sample"][:], dtype=np.int64)
        g_start = np.asarray(h5["graph_start_sample"][:], dtype=np.int64)
        g_end = np.asarray(h5["graph_end_sample"][:], dtype=np.int64)
        g_bound = np.asarray(h5["graph_boundary_type"][:], dtype=np.int64)
        starts = np.searchsorted(graph_exam, np.arange(labels.shape[0]), side="left")
        ends = np.searchsorted(graph_exam, np.arange(labels.shape[0]), side="right")
        n_exams = labels.shape[0] if args.limit <= 0 else min(args.limit, labels.shape[0])

        af = classes.index("AF")
        af_ratios = []
        for e in np.flatnonzero(labels[:, af] == 1):
            for lead_id in np.unique(g_lead[starts[e] : ends[e]]):
                sel = np.arange(starts[e], ends[e])[g_lead[starts[e] : ends[e]] == lead_id]
                sel = sel[g_bound[sel] == 0]
                if sel.size >= 3:
                    d = (g_end[sel] - g_start[sel]) / fs
                    af_ratios.extend((d / np.median(d)).tolist())
        af_ratios = np.asarray(af_ratios)

        def exam_meta(e: int) -> np.ndarray:
            g0, g1 = int(starts[e]), int(ends[e])
            return np.stack([(g_end[g0:g1] - g_start[g0:g1]) / fs, g_bound[g0:g1]], axis=1)

        rng = np.random.default_rng(args.run.seed)
        intervention_set = set(range(n_exams))
        if args.intervention_exams == "rhythm_positive_plus_sample":
            rhythm = [classes.index(c) for c in ("SB", "AF", "ST")]
            pos = set(np.flatnonzero(labels[:n_exams, rhythm].any(1)).tolist())
            rest = sorted(set(range(n_exams)) - pos)
            sample = np.random.default_rng(0).choice(
                rest, size=min(args.negative_sample, len(rest)), replace=False
            )
            intervention_set = pos | set(sample.tolist())

        exams = []
        lead_check = []
        intact_check = []
        identity_check = []
        variant_names: list[str] | None = None
        from tqdm import tqdm

        for e in tqdm(range(n_exams), unit="exam", mininterval=10):
            g0, g1 = int(starts[e]), int(ends[e])
            desc = rhythm_descriptors(g_r[g0:g1], g_lead[g0:g1], g_beat[g0:g1], fs)
            record: dict = {
                "exam_index": e,
                "exam_id": decode_vlen_str(exam_ids[e]),
                "labels": labels[e].tolist(),
                **desc,
            }
            data = bundle.dataset[e]
            if True:
                pieces = decompose_exam(data, exam_meta(e))
                if "interventions" in args.blocks and e in intervention_set:
                    names, items = intervention_variants(pieces, desc["hr_bpm"], af_ratios, rng)
                    logits = batched_logits(model, items, device)
                    record["interventions"] = {
                        n: row.tolist() for n, row in zip(names, logits, strict=True)
                    }
                    record["intact_logits"] = logits[0].tolist()
                    if variant_names is None:
                        variant_names = names
                else:
                    record["intact_logits"] = batched_logits(
                        model, [assemble_exam(pieces)], device
                    )[0].tolist()
            inputs = single_exam_inputs(bundle.dataset, e, device)
            if len(intact_check) < LEAD_CHECK_EXAMS:
                with torch.no_grad():
                    direct = model(**inputs).logits[0].float().cpu().numpy()
                intact_check.append(
                    float(np.abs(direct - np.asarray(record["intact_logits"])).max())
                )
                if intact_check[-1] > 1e-3:
                    raise SystemExit(
                        f"reassembled exam differs from dataset item: {intact_check[-1]}"
                    )
            if "leads" in args.blocks:
                lead_emb, present = lead_embeddings(model, inputs)
                if len(lead_check) < args.lead_check_exams and int(present.sum()) >= 2:
                    lead_check.append(check_lead_factorization(model, inputs, lead_emb, present))
                    if lead_check[-1] > 1e-3:
                        raise SystemExit(f"lead factorization check failed: {lead_check[-1]}")
                record["leads"] = exact_lead_shapley(model, lead_emb, present)
            if "diastolic" in args.blocks:
                cores, delineated = [], 0
                for k, g in enumerate(range(g0, g1)):
                    fid = fiducials_by_graph.get(g)
                    c0, c1, ok = beat_core(
                        None if fid is None else (fid.p_onset_sample, fid.t_offset_sample),
                        int(g_start[g]),
                        (g_r[g] - g_start[g]) / fs,
                        pieces[k].duration_s,
                        fs,
                    )
                    cores.append((c0, c1))
                    delineated += ok
                names, items, unchanged = diastolic_variants(pieces, cores, desc["hr_bpm"], fs)
                ok_items = [it for it in items if it is not None]
                ok_logits = iter(batched_logits(model, ok_items, device)) if ok_items else iter(())
                record["diastolic"] = {
                    n: (None if it is None else next(ok_logits).tolist())
                    for n, it in zip(names, items, strict=True)
                }
                record["diastolic_unchanged_fraction"] = dict(zip(names, unchanged, strict=True))
                record["diastolic_delineated_units"] = delineated
                record["n_units"] = g1 - g0
                if len(identity_check) < LEAD_CHECK_EXAMS:
                    identity_check.append(
                        float(
                            np.abs(
                                np.asarray(record["diastolic"]["identity"])
                                - np.asarray(record["intact_logits"])
                            ).max()
                        )
                    )
            if "regions" in args.blocks and e in region_exams:
                record["regions"] = regional_block(
                    model,
                    inputs,
                    h5,
                    e,
                    graph_exam,
                    records_by_exam.get(e, []),
                    candidate,
                    len(classes),
                    int(fs),
                )
            exams.append(record)

    report = {
        "producer": "rhythm_lead_audit",
        "seed": args.run.seed,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "dataset_path": str(dataset_path),
        "dataset_hdf5_sha256": sha256_file(dataset_path),
        "class_order": classes,
        "blocks": list(args.blocks),
        "rate_targets_bpm": list(RATE_TARGETS_BPM),
        "n_af_donor_ratios": int(af_ratios.size),
        "reassembly_max_abs_error": max(intact_check) if intact_check else None,
        "diastolic_identity_max_abs_error": max(identity_check) if identity_check else None,
        "diastolic_targets_bpm": list(DIASTOLIC_TARGETS_BPM),
        "lead_factorization_max_abs_error": max(lead_check) if lead_check else None,
        "lead_factorization_checked_exams": len(lead_check),
        "region_order": ["pr", "qrs", "t"],
        "exams": exams,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report))
    print(f"wrote {args.output} ({len(exams)} exams)")


if __name__ == "__main__":
    main()
