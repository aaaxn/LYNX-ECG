"""Aggregate three-seed ``rhythm_lead_audit`` reports into tables.

Reads one report per seed for one dataset and writes a Markdown summary with
the rate/irregularity descriptors, beat-sequence interventions, rate sweep,
exact lead Shapley, and (when present) the six-class PR-QRS-T game. Verdicts
are exam-class-seed triples; a positive prediction uses the fixed 0.5
threshold on the logit (logit >= 0).

Usage:
    uv run python scripts/explain/rhythm_lead_summary.py \
        --reports reports/rhythm_lead_audit/codetest_r*_s*.json \
        --output reports/rhythm_lead_audit/summary_codetest.md
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
TERRITORIES = {
    "inferior (II, III, aVF)": (1, 2, 5),
    "lateral (I, aVL, V5, V6)": (0, 4, 10, 11),
    "septal (V1, V2)": (6, 7),
    "anterior (V3, V4)": (8, 9),
    "aVR": (3,),
}
VARIANTS = ("isolated", "shuffled", "template", "template_rr", "rr_regularized", "rr_irregularized")
HR_BINS = (0, 45, 50, 55, 60, 90, 100, 110, 1e9)
CV_BINS = (0, 0.05, 0.10, 0.15, 0.20, 1e9)


def load(paths: list[Path]) -> tuple[list[dict], list[str]]:
    reports = [json.loads(p.read_text()) for p in paths]
    classes = reports[0]["class_order"]
    ids = [e["exam_id"] for e in reports[0]["exams"]]
    for r in reports[1:]:
        if r["class_order"] != classes or [e["exam_id"] for e in r["exams"]] != ids:
            raise SystemExit("reports cover different exams or class orders")
    if len({r["seed"] for r in reports}) != len(reports):
        raise SystemExit("duplicate seeds")
    return reports, classes


def fmt_med(values: np.ndarray, digits: int = 2) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def bin_label(lo: float, hi: float) -> str:
    if lo == 0:
        return f"< {hi:g}"
    if hi >= 1e8:
        return f">= {lo:g}"
    return f"{lo:g}-{hi:g}"


def descriptor_section(reports: list[dict], classes: list[str]) -> str:
    ex = reports[0]["exams"]
    labels = np.array([e["labels"] for e in ex])
    hr = np.array([np.nan if e["hr_bpm"] is None else e["hr_bpm"] for e in ex])
    cv = np.array([np.nan if e["rr_cv"] is None else e["rr_cv"] for e in ex])
    logits = np.stack([[e["intact_logits"] for e in r["exams"]] for r in reports])
    pred = logits >= 0
    neg6 = labels.sum(1) == 0
    lines = ["## 1. Measured rate and RR irregularity\n"]
    rows = []
    for c, name in enumerate(classes):
        pos = labels[:, c] == 1
        rows.append([name, int(pos.sum()), fmt_med(hr[pos], 1), fmt_med(cv[pos], 3)])
    rows.append(["six-class-negative", int(neg6.sum()), fmt_med(hr[neg6], 1), fmt_med(cv[neg6], 3)])
    lines.append(
        "Heart rate (bpm) and RR coefficient of variation, median [IQR] over exams. "
        "Values come from the stored R anchors (median across leads).\n"
    )
    lines.append(table(["Cohort", "n", "HR (bpm)", "RR CV"], rows))

    sb, st, af = classes.index("SB"), classes.index("ST"), classes.index("AF")
    rows = []
    for lo, hi in itertools.pairwise(HR_BINS):
        m = (hr >= lo) & (hr < hi)
        rows.append(
            [
                bin_label(lo, hi),
                int(m.sum()),
                int(labels[m, sb].sum()),
                f"{pred[:, m, sb].sum(1).mean():.1f}",
                int(labels[m, st].sum()),
                f"{pred[:, m, st].sum(1).mean():.1f}",
                int(labels[m, af].sum()),
            ]
        )
    lines.append("\nExams by measured heart rate. Predicted counts are means over seeds.\n")
    lines.append(
        table(
            [
                "HR bin (bpm)",
                "Exams",
                "SB labels",
                "SB predicted",
                "ST labels",
                "ST predicted",
                "AF labels",
            ],
            rows,
        )
    )

    rows = []
    for lo, hi in itertools.pairwise(CV_BINS):
        m = (cv >= lo) & (cv < hi)
        rows.append(
            [
                bin_label(lo, hi),
                int(m.sum()),
                int(labels[m, af].sum()),
                f"{pred[:, m, af].sum(1).mean():.1f}",
            ]
        )
    lines.append("\nExams by RR coefficient of variation.\n")
    lines.append(table(["RR CV bin", "Exams", "AF labels", "AF predicted"], rows))

    rows = []
    for name, c, desc, dname in (
        ("SB", sb, hr, "HR"),
        ("ST", st, hr, "HR"),
        ("AF", af, cv, "RR CV"),
    ):
        pos = labels[:, c] == 1
        fp = (~pos)[None, :] & pred[:, :, c]
        fn = pos[None, :] & ~pred[:, :, c]
        tp = pos[None, :] & pred[:, :, c]
        rows.append(
            [
                name,
                dname,
                fmt_med(np.concatenate([desc[m] for m in tp]), 3 if c == af else 1),
                f"{fp.sum() / len(reports):.1f}",
                fmt_med(np.concatenate([desc[m] for m in fp]), 3 if c == af else 1),
                f"{fn.sum() / len(reports):.1f}",
                fmt_med(np.concatenate([desc[m] for m in fn]), 3 if c == af else 1),
            ]
        )
    lines.append(
        "\nDescriptor values of true positives, false positives, and false negatives "
        "(pooled over seeds; counts are per-seed means).\n"
    )
    lines.append(
        table(
            [
                "Class",
                "Descriptor",
                "TP median [IQR]",
                "FP per seed",
                "FP median [IQR]",
                "FN per seed",
                "FN median [IQR]",
            ],
            rows,
        )
    )

    rows = []
    for name, c in (("SB", sb), ("AF", af), ("ST", st)):
        fp_exams = np.flatnonzero(((labels[:, c] == 0)[None, :] & pred[:, :, c]).any(0))
        for e in fp_exams:
            other = [classes[k] for k in range(len(classes)) if labels[e, k]]
            rows.append(
                [
                    name,
                    ex[e]["exam_id"],
                    int(pred[:, e, c].sum()),
                    f"{hr[e]:.1f}",
                    f"{cv[e]:.3f}",
                    ", ".join(other) or "none",
                ]
            )
    if len(rows) <= 80:
        lines.append("\nRhythm false positives (exams predicted positive in at least one seed).\n")
        lines.append(table(["Class", "Exam", "Seeds FP", "HR", "RR CV", "Labels"], rows))
    return "\n".join(lines)


def variant_keys(interventions: dict, name: str) -> list[str]:
    """Keys of one intervention family: the name itself or its numbered random draws.

    A plain prefix match would also take ``template_rr`` into ``template``.
    """
    prefix = name + "_"
    return sorted(
        k
        for k in interventions
        if k == name or (k.startswith(prefix) and k[len(prefix) :].isdigit())
    )


def intervention_section(reports: list[dict], classes: list[str]) -> str:
    verdicts = [e for r in reports for e in r["exams"] if "interventions" in e]
    if not verdicts:
        return ""
    labels = np.array([e["labels"] for e in verdicts])
    n_exams = len({e["exam_index"] for e in verdicts})

    def variant_logits(name: str) -> np.ndarray:
        """[draws, V, C] logits for a variant family."""
        keys = variant_keys(verdicts[0]["interventions"], name)
        return np.stack([np.array([e["interventions"][k] for e in verdicts]) for k in keys])

    intact = variant_logits("intact")[0]
    lines = [
        f"## 2. Beat-sequence interventions ({n_exams} exams, {len(verdicts)} exam-seed pairs)\n",
        "Cells give the share of positive verdicts (exam x seed) that stay positive, "
        "and the median logit change from intact. Shuffled and irregularized "
        "average five random draws.\n",
    ]
    header = ["Class", "Positive verdicts", "Intact TP", *list(VARIANTS)]
    rows = []
    fam = {v: variant_logits(v) for v in VARIANTS}
    for c, name in enumerate(classes):
        pos = labels[:, c] == 1
        row = [name, int(pos.sum()), f"{(intact[pos, c] >= 0).mean():.2f}"]
        for v in VARIANTS:
            lv = fam[v][:, pos, c]
            dlog = np.median(lv - intact[None, pos, c])
            row.append(f"{(lv >= 0).mean():.2f} ({dlog:+.1f})")
        rows.append(row)
    lines.append(table(header, rows))

    lines.append("\nFalse-positive induction in class-negative exams: share predicted positive.\n")
    rows = []
    for c, name in enumerate(classes):
        neg = labels[:, c] == 0
        row = [name, int(neg.sum()), f"{(intact[neg, c] >= 0).mean():.3f}"]
        row += [f"{(fam[v][:, neg, c] >= 0).mean():.3f}" for v in VARIANTS]
        rows.append(row)
    lines.append(table(["Class", "Negative verdicts", "Intact FP", *list(VARIANTS)], rows))

    targets = reports[0]["rate_targets_bpm"]
    swept = [e for e in verdicts if f"rate_{targets[0]}" in e["interventions"]]
    sweep = np.stack([np.array([e["interventions"][f"rate_{t}"] for e in swept]) for t in targets])
    lab = np.array([e["labels"] for e in swept])
    sb, st, af = classes.index("SB"), classes.index("ST"), classes.index("AF")
    cohorts = {
        "six-class-negative": lab.sum(1) == 0,
        "SB positive": lab[:, sb] == 1,
        "ST positive": lab[:, st] == 1,
        "AF positive": lab[:, af] == 1,
    }
    lines.append("\n## 3. Rate sweep\n")
    lines.append(
        "Every beat's time axis is scaled by measured HR / target HR. Cells give the "
        "share of verdicts predicted SB / ST / AF at each target rate; n counts verdicts.\n"
    )
    shown = [
        t
        for t in targets
        if t in (30, 40, 45, 50, 55, 60, 70, 80, 90, 95, 100, 105, 110, 120, 140, 160, 180)
    ]
    rows = []
    for t in shown:
        k = targets.index(t)
        row = [t]
        for m in cohorts.values():
            p = sweep[k][m] >= 0
            row.append(f"{p[:, sb].mean():.2f} / {p[:, st].mean():.2f} / {p[:, af].mean():.2f}")
        rows.append(row)
    lines.append(
        table(["Target HR"] + [f"{c} (n={int(m.sum())})" for c, m in cohorts.items()], rows)
    )

    rows = []
    tarr = np.array(targets, dtype=float)
    for cname, m in cohorts.items():
        psb = sweep[:, m, sb] >= 0
        pst = sweep[:, m, st] >= 0
        sb_edge = np.where(psb.any(0), tarr[::-1][np.argmax(psb[::-1], axis=0)], np.nan)
        st_edge = np.where(pst.any(0), tarr[np.argmax(pst, axis=0)], np.nan)
        rows.append(
            [
                cname,
                fmt_med(sb_edge, 0),
                fmt_med(st_edge, 0),
                f"{np.isfinite(sb_edge).mean():.2f}",
                f"{np.isfinite(st_edge).mean():.2f}",
            ]
        )
    lines.append(
        "\nImplied decision rates: highest target HR still called SB and lowest target "
        "HR called ST, median [IQR] over verdicts (CODE labels: SB <= 50, ST >= 100 bpm).\n"
    )
    lines.append(
        table(["Cohort", "SB edge (bpm)", "ST edge (bpm)", "Any SB call", "Any ST call"], rows)
    )

    rows = []
    for c, name in enumerate(classes):
        if name in ("SB", "ST"):
            continue
        m = cohorts["six-class-negative"]
        row = [name]
        for t in (40, 60, 100, 150):
            row.append(f"{(sweep[targets.index(t), m, c] >= 0).mean():.3f}")
        rows.append(row)
    lines.append(
        "\nOther-class calls on six-class-negative exams under time scaling "
        "(time stretch also widens PR and QRS).\n"
    )
    lines.append(table(["Class", "40 bpm", "60 bpm", "100 bpm", "150 bpm"], rows))
    return "\n".join(lines)


def lead_section(reports: list[dict], classes: list[str]) -> str:
    if "leads" not in reports[0]["exams"][0]:
        return ""
    ex = reports[0]["exams"]
    labels = np.array([e["labels"] for e in ex])
    phi = np.stack([[e["leads"]["shapley_logit"] for e in r["exams"]] for r in reports])
    alone = np.stack([[e["leads"]["lead_alone_logits"] for e in r["exams"]] for r in reports])
    loo = np.stack([[e["leads"]["leave_one_out_logits"] for e in r["exams"]] for r in reports])
    full = np.stack([[e["leads"]["full_logits"] for e in r["exams"]] for r in reports])
    lines = [
        "## 4. Exact 12-lead Shapley\n",
        "Positive exams only. Share = lead Shapley / sum of positive lead Shapley values "
        "in the verdict, averaged over verdicts. Top-1 = share of verdicts where the "
        "lead has the largest Shapley value. The six limb leads are linear combinations "
        "of two independent signals (Einthoven and Goldberger relations). The exam "
        "embedding is an attention-weighted mean of lead embeddings, so a negative "
        "value can mean dilution of stronger leads, not evidence against the class.\n",
    ]
    rows_share, rows_top, rows_terr, rows_suff, rows_alone = [], [], [], [], []
    for c, name in enumerate(classes):
        pos = labels[:, c] == 1
        p = phi[..., c][:, pos].reshape(-1, len(LEADS))
        p = np.nan_to_num(p)
        mass = np.clip(p, 0, None).sum(1, keepdims=True)
        share = np.where(mass > 0, p / np.where(mass > 0, mass, 1), 0).mean(0)
        top = np.bincount(p.argmax(1), minlength=len(LEADS)) / p.shape[0]
        rows_share.append([name, pos.sum()] + [f"{s:.2f}" for s in share])
        rows_top.append([name] + [f"{t:.2f}" for t in top])
        terr = [f"{share[list(ix)].sum():.2f}" for ix in TERRITORIES.values()]
        pos_part = np.clip(p, 0, None)
        eff = (pos_part.sum(1) ** 2) / np.maximum((pos_part**2).sum(1), 1e-12)
        rows_terr.append([name, *terr, fmt_med(eff, 1)])
        a = alone[..., c][:, pos].reshape(-1, len(LEADS))
        lo = loo[..., c][:, pos].reshape(-1, len(LEADS))
        f = full[:, pos, c].reshape(-1)
        tpm = f >= 0
        any_alone = (a[tpm] >= 0).any(1).mean() if tpm.any() else float("nan")
        n_alone = np.nanmean((a[tpm] >= 0).sum(1)) if tpm.any() else float("nan")
        loo_flip = (lo[tpm] < 0).any(1).mean() if tpm.any() else float("nan")
        best_alone = [LEADS[j] for j in np.argsort(-(a[tpm] >= 0).mean(0))[:3]] if tpm.any() else []
        rows_suff.append(
            [
                name,
                int(tpm.sum()),
                f"{any_alone:.2f}",
                f"{n_alone:.1f}",
                ", ".join(best_alone),
                f"{loo_flip:.2f}",
            ]
        )
        rows_alone.append([name] + [f"{v:.2f}" for v in (a[tpm] >= 0).mean(0)])
    lines.append(table(["Class", "Positives", *list(LEADS)], rows_share))
    lines.append("\nTop-1 lead frequency.\n")
    lines.append(table(["Class", *list(LEADS)], rows_top))
    lines.append(
        "\nShare by territory, and effective number of contributing leads "
        "(participation ratio of positive Shapley values).\n"
    )
    lines.append(table(["Class", *list(TERRITORIES), "Effective leads"], rows_terr))
    lines.append(
        "\nSingle-lead sufficiency among true-positive verdicts: a lead alone keeps "
        "the logit >= 0; leave-one-out flips the call.\n"
    )
    lines.append(
        table(
            [
                "Class",
                "TP verdicts",
                "Some lead alone suffices",
                "Leads that suffice alone (mean)",
                "Most often sufficient",
                "Some single-lead removal flips",
            ],
            rows_suff,
        )
    )
    lines.append(
        "\nPer-lead sufficiency: share of true-positive verdicts in which the lead "
        "alone keeps the call positive.\n"
    )
    lines.append(table(["Class", *list(LEADS)], rows_alone))
    return "\n".join(lines)


def region_section(reports: list[dict], classes: list[str]) -> str:
    ex = reports[0]["exams"]
    has_regions = np.array(["regions" in e for e in ex])
    if not has_regions.any():
        return ""
    labels = np.array([e["labels"] for e in ex])
    regions = ("pr", "qrs", "t")
    lines = [
        "## 5. PR-QRS-T structural deletion for all six classes\n",
        "Same eight-state game as the paper (Prominence delineation). Leading region "
        "counts among correct verdicts; median Shapley (logit) over all positive verdicts.\n",
    ]
    for rule in ("beat_complete", "per_region"):
        rows = []
        for c, name in enumerate(classes):
            pos = np.flatnonzero((labels[:, c] == 1) & has_regions)
            lead_counts = {g: 0 for g in regions}
            correct = total = 0
            vals = {g: [] for g in regions}
            for r in reports:
                for e in pos:
                    block = r["exams"][e]["regions"][rule]
                    sh = block["shapley_logit"][c]
                    if sh is None:
                        continue
                    total += 1
                    for g in regions:
                        vals[g].append(sh[g])
                    if block["state_logits"]["intact"][c] >= 0:
                        correct += 1
                        lead_counts[max(regions, key=lambda g: sh[g])] += 1
            rows.append(
                [name, total, correct]
                + [lead_counts[g] for g in regions]
                + [fmt_med(np.array(vals[g]), 2) for g in regions]
            )
        lines.append(f"\nAvailability rule: `{rule}`.\n")
        lines.append(
            table(
                [
                    "Class",
                    "Verdicts",
                    "Correct",
                    "PR leads",
                    "QRS leads",
                    "T leads",
                    "PR Shapley",
                    "QRS Shapley",
                    "T Shapley",
                ],
                rows,
            )
        )

    rows = []
    cohorts = {name: labels[:, c] == 1 for c, name in enumerate(classes)}
    cohorts["six-class-negative"] = labels.sum(1) == 0
    for cname, m in cohorts.items():
        m = m & has_regions
        if not m.any():
            continue
        units = np.array([ex[e]["regions"]["per_region"]["n_units"] for e in np.flatnonzero(m)])
        avail = np.array(
            [ex[e]["regions"]["per_region"]["region_available_units"] for e in np.flatnonzero(m)]
        )
        comp = np.array(
            [ex[e]["regions"]["beat_complete"]["beat_complete_units"] for e in np.flatnonzero(m)]
        )
        tot = units.sum()
        rows.append(
            [cname, int(m.sum())]
            + [f"{avail[:, k].sum() / tot:.3f}" for k in range(3)]
            + [f"{comp.sum() / tot:.3f}"]
        )
    lines.append("\nDelineation availability: share of lead-beat units with each region marked.\n")
    lines.append(table(["Cohort", "Exams", "PR", "QRS", "T", "Complete triplet"], rows))
    return "\n".join(lines)


def boot_ratio(num: np.ndarray, den: np.ndarray, n_boot: int, rng: np.random.Generator) -> str:
    """Pooled ratio sum(num) / sum(den) with a percentile interval.

    ``num`` and ``den`` hold one value per exam, already summed over seeds, so
    each resample draws whole exams with all their seeds.
    """
    keep = den > 0
    num, den = num[keep].astype(float), den[keep].astype(float)
    if den.size == 0:
        return "n/a"
    point = num.sum() / den.sum()
    draws = []
    for start in range(0, n_boot, 200):
        idx = rng.integers(0, den.size, size=(min(200, n_boot - start), den.size))
        draws.append(num[idx].sum(1) / den[idx].sum(1))
    lo, hi = np.percentile(np.concatenate(draws), [2.5, 97.5])
    return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


def _crossing(targets: np.ndarray, share: np.ndarray, falling: bool) -> float:
    """Interpolated target rate where ``share`` crosses 0.5.

    ``falling`` finds the highest crossing from above 0.5 to below it in the
    bradycardia range; otherwise the lowest crossing from below to above.
    """
    above = share >= 0.5
    steps = (
        np.flatnonzero(above[:-1] & ~above[1:])
        if falling
        else np.flatnonzero(~above[:-1] & above[1:])
    )
    if steps.size == 0:
        return float("nan")
    k = steps[-1] if falling else steps[0]
    t0, t1, s0, s1 = targets[k], targets[k + 1], share[k], share[k + 1]
    return float(t0 + (0.5 - s0) * (t1 - t0) / (s1 - s0))


def boot_crossing(
    calls: np.ndarray,
    den: np.ndarray,
    targets: np.ndarray,
    window: tuple[float, float],
    falling: bool,
    n_boot: int,
    rng: np.random.Generator,
) -> str:
    """50% crossing of the cohort call-share curve, with an exam bootstrap."""
    sel = (targets >= window[0]) & (targets <= window[1])
    t = targets[sel]
    calls = calls[:, sel].astype(float)
    den = den.astype(float)
    point = _crossing(t, calls.sum(0) / den.sum(), falling)
    idx = rng.integers(0, den.size, size=(n_boot, den.size))
    draws = np.array(
        [_crossing(t, calls[i].sum(0) / den[i].sum(), falling) for i in idx], dtype=float
    )
    if not np.isfinite(point):
        return "none"
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return f"{point:.1f} [{lo:.1f}, {hi:.1f}]"


def ci_section(reports: list[dict], classes: list[str], n_boot: int) -> str:
    rng = np.random.default_rng(0)
    sb, af, st = classes.index("SB"), classes.index("AF"), classes.index("ST")
    ex = reports[0]["exams"]
    labels = np.array([e["labels"] for e in ex])
    hr = np.array([np.nan if e["hr_bpm"] is None else e["hr_bpm"] for e in ex])
    cv = np.array([np.nan if e["rr_cv"] is None else e["rr_cv"] for e in ex])
    calls = np.stack([[np.array(e["intact_logits"]) >= 0 for e in r["exams"]] for r in reports])
    n_calls = calls.sum(0)
    n_seeds = len(reports)
    lines = [
        "## 6. Paper tables with 95% intervals\n",
        f"Exam-cluster bootstrap: {n_boot} resamples of exams, each with all its seeds; "
        "percentile intervals. Shares pool exam x seed verdicts.\n",
    ]

    rows = []
    specs = (
        ("SB", sb, hr, (0, 50, 55, 60, 1e9), "HR"),
        ("ST", st, hr, (0, 100, 105, 110, 1e9), "HR"),
        ("AF", af, cv, (0, 0.10, 0.20, 1e9), "RR CV"),
    )
    for name, c, desc, edges, dname in specs:
        for lo, hi in itertools.pairwise(edges):
            inbin = (desc >= lo) & (desc < hi)
            pos, neg = inbin & (labels[:, c] == 1), inbin & (labels[:, c] == 0)
            rows.append(
                [
                    name,
                    f"{dname} {bin_label(lo, hi)}",
                    int(pos.sum()),
                    boot_ratio(n_calls[pos, c], np.full(pos.sum(), n_seeds), n_boot, rng),
                    int(neg.sum()),
                    boot_ratio(n_calls[neg, c], np.full(neg.sum(), n_seeds), n_boot, rng),
                ]
            )
    lines.append(
        "\nRecall among label positives and false-call rate among label negatives, "
        "by measured descriptor.\n"
    )
    lines.append(
        table(
            ["Class", "Bin", "Positives", "Recall", "Negatives", "False-call rate"],
            rows,
        )
    )

    verdicts = [e for r in reports for e in r["exams"] if "interventions" in e]
    if verdicts:
        exam_of = np.array([e["exam_index"] for e in verdicts])
        uniq, inv = np.unique(exam_of, return_inverse=True)
        vlab = np.array([e["labels"] for e in verdicts])

        def per_exam(values: np.ndarray) -> np.ndarray:
            return np.bincount(inv, weights=values.astype(float), minlength=uniq.size)

        def share_ci(keys: list[str], c: int, mask: np.ndarray) -> str:
            hit = np.mean(
                [np.array([e["interventions"][k][c] >= 0 for e in verdicts]) for k in keys],
                axis=0,
            )
            return boot_ratio(per_exam(hit * mask), per_exam(mask), n_boot, rng)

        keys_of = {v: variant_keys(verdicts[0]["interventions"], v) for v in ("intact", *VARIANTS)}
        rows = []
        for c, name in enumerate(classes):
            pos = vlab[:, c] == 1
            rows.append(
                [name, int(pos.sum())]
                + [share_ci(keys_of[v], c, pos) for v in ("intact", *VARIANTS)]
            )
        lines.append(
            "\nShare of label-positive verdicts called positive under each intervention.\n"
        )
        lines.append(table(["Class", "Verdicts", "intact", *VARIANTS], rows))
        rows = []
        for c, name in enumerate(classes):
            neg = vlab[:, c] == 0
            rows.append(
                [name, int(neg.sum())]
                + [share_ci(keys_of[v], c, neg) for v in ("intact", *VARIANTS)]
            )
        lines.append("\nShare of label-negative verdicts called positive (false-call induction).\n")
        lines.append(table(["Class", "Verdicts", "intact", *VARIANTS], rows))

        targets = np.array(reports[0]["rate_targets_bpm"], dtype=float)
        swept = np.array([f"rate_{int(targets[0])}" in e["interventions"] for e in verdicts])
        sweep = np.array(
            [
                [e["interventions"][f"rate_{int(t)}"] for t in targets]
                if s
                else np.full((targets.size, len(classes)), np.nan)
                for e, s in zip(verdicts, swept, strict=True)
            ]
        )
        cohorts = {
            "six-class-negative": vlab.sum(1) == 0,
            "SB positive": vlab[:, sb] == 1,
            "ST positive": vlab[:, st] == 1,
            "AF positive": vlab[:, af] == 1,
        }
        rows = []
        for cname, m in cohorts.items():
            m = m & swept
            ex_ids, ex_inv = np.unique(exam_of[m], return_inverse=True)
            den = np.bincount(ex_inv, minlength=ex_ids.size)
            out = [cname, int(ex_ids.size)]
            for c, window, falling in ((sb, (30, 80), True), (st, (80, 140), False)):
                hits = (sweep[m][:, :, c] >= 0).astype(float)
                per = np.stack(
                    [
                        np.bincount(ex_inv, weights=hits[:, k], minlength=ex_ids.size)
                        for k in range(targets.size)
                    ],
                    axis=1,
                )
                out.append(boot_crossing(per, den, targets, window, falling, n_boot, rng))
            rows.append(out)
        lines.append(
            "\nImposed heart rate at which half of the cohort's verdicts are called SB "
            "(falling edge, 30-80 bpm) or ST (rising edge, 80-140 bpm); 'none' means "
            "the cohort curve never reaches 0.5.\n"
        )
        lines.append(table(["Cohort", "Exams", "SB 50% rate (bpm)", "ST 50% rate (bpm)"], rows))

        m = cohorts["AF positive"] & swept
        rows = []
        for t in (40, 60, 80, 100, 120, 150, 180):
            k = int(np.flatnonzero(targets == t)[0])
            hit = (sweep[:, k, af] >= 0) & m
            rows.append([t, boot_ratio(per_exam(hit), per_exam(m), n_boot, rng)])
        lines.append("\nAF-positive verdicts still called AF at each imposed heart rate.\n")
        lines.append(table(["Imposed HR (bpm)", "Called AF"], rows))

    if "leads" in ex[0]:
        alone = np.stack([[e["leads"]["lead_alone_logits"] for e in r["exams"]] for r in reports])
        full = np.stack([[e["leads"]["full_logits"] for e in r["exams"]] for r in reports])
        rows = []
        for c, name in enumerate(classes):
            tp = (labels[None, :, c] == 1) & (full[:, :, c] >= 0)
            den = tp.sum(0)
            row = [name, int(den.sum())]
            for j in range(len(LEADS)):
                num = ((alone[:, :, j, c] >= 0) & tp).sum(0)
                row.append(boot_ratio(num, den, n_boot, rng))
            rows.append(row)
        lines.append(
            "\nShare of true-positive verdicts in which one lead alone keeps the call positive.\n"
        )
        lines.append(table(["Class", "TP verdicts", *LEADS], rows))
    return "\n".join(lines)


def diastolic_section(reports: list[dict], classes: list[str], n_boot: int) -> str:
    """Interval-preserving rate change: only the pre-P and post-T time is resized."""
    verdicts = [e for r in reports for e in r["exams"] if "diastolic" in e]
    if not verdicts:
        return ""
    rng = np.random.default_rng(0)
    targets = np.array(reports[0]["diastolic_targets_bpm"], dtype=float)
    labels = np.array([e["labels"] for e in verdicts])
    exam_of = np.array([e["exam_index"] for e in verdicts])
    n_cls = len(classes)
    logits = np.full((len(verdicts), targets.size, n_cls), np.nan)
    for v, e in enumerate(verdicts):
        for k, t in enumerate(targets):
            row = e["diastolic"].get(f"rate_{int(t)}")
            if row is not None:
                logits[v, k] = row
    feasible = np.isfinite(logits[:, :, 0])
    delineated = sum(e["diastolic_delineated_units"] for e in verdicts) / sum(
        e["n_units"] for e in verdicts
    )
    ident = max(r.get("diastolic_identity_max_abs_error") or 0 for r in reports)
    idx = {c: classes.index(c) for c in classes}
    lines = [
        "## 7. Interval-preserving rate change\n",
        "Only the time before P onset and after T offset changes; PR, QRS, and QT "
        "keep their durations. Support nodes are refilled with the build rule and "
        "natural visibility is rebuilt. A target is infeasible when an interior beat "
        f"would keep less than 20 ms of diastole in more than 10% of interior beats (those beats keep their shape). Delineated cores cover {delineated:.1%} "
        f"of units; the rest use R - 0.25 s to R + 0.45 s. Identity check max |logit| "
        f"error: {ident:.1e}.\n",
    ]
    cohorts = {
        "six-class-negative": labels.sum(1) == 0,
        **{f"{c} positive": labels[:, idx[c]] == 1 for c in classes},
    }
    shown = [t for t in (40, 50, 60, 70, 80, 90, 100, 110, 120, 130) if t in targets]
    lines.append(
        "\nShare of feasible verdicts called positive for the cohort's class "
        "(six-class-negative: SB / ST); feasible share in parentheses.\n"
    )
    rows = []
    for cname, m in cohorts.items():
        row = [cname, int(m.sum())]
        for t in shown:
            k = int(np.flatnonzero(targets == t)[0])
            f = feasible[m, k]
            if not f.any():
                row.append(f"— ({f.mean():.2f})")
                continue
            if cname == "six-class-negative":
                sb = (logits[m, k, idx["SB"]][f] >= 0).mean()
                st = (logits[m, k, idx["ST"]][f] >= 0).mean()
                row.append(f"{sb:.2f} / {st:.2f} ({f.mean():.2f})")
            else:
                c = idx[cname.split()[0]]
                row.append(f"{(logits[m, k, c][f] >= 0).mean():.2f} ({f.mean():.2f})")
        rows.append(row)
    lines.append(table(["Cohort", "Verdicts", *[f"{t} bpm" for t in shown]], rows))

    specs = [
        ("six-class-negative", "SB", (40, 70), True),
        ("SB positive", "SB", (45, 80), True),
        ("six-class-negative", "ST", (70, 130), False),
        ("ST positive", "ST", (60, 110), False),
    ]
    rows = []
    for cname, cls, window, falling in specs:
        sel = (targets >= window[0]) & (targets <= window[1])
        m = cohorts[cname] & feasible[:, sel].all(1)
        ex_ids, ex_inv = np.unique(exam_of[m], return_inverse=True)
        if ex_ids.size == 0:
            rows.append([cname, cls, f"{window[0]}-{window[1]}", "0", "none"])
            continue
        den = np.bincount(ex_inv, minlength=ex_ids.size)
        hits = (logits[m][:, :, idx[cls]] >= 0).astype(float)
        per = np.stack(
            [
                np.bincount(ex_inv, weights=hits[:, k], minlength=ex_ids.size)
                for k in range(targets.size)
            ],
            axis=1,
        )
        rows.append(
            [
                cname,
                cls,
                f"{window[0]}-{window[1]}",
                f"{int(m.sum())}/{int(cohorts[cname].sum())}",
                boot_crossing(per, den, targets, window, falling, n_boot, rng),
            ]
        )
    lines.append(
        "\n50% crossing of the call-share curve (verdicts feasible over the whole "
        "window), 95% exam-bootstrap interval.\n"
    )
    lines.append(
        table(["Cohort", "Class", "Window (bpm)", "Verdicts used", "Crossing (bpm)"], rows)
    )

    rows = []
    for cls in ("1dAVb", "RBBB", "LBBB"):
        m = cohorts[f"{cls} positive"]
        out = [cls]
        for t in (50, 60, 80, 100):
            k = int(np.flatnonzero(targets == t)[0])
            mm = m & feasible[:, k]
            hit = (logits[:, k, idx[cls]] >= 0) & mm
            num = np.bincount(np.unique(exam_of, return_inverse=True)[1], weights=hit.astype(float))
            den = np.bincount(np.unique(exam_of, return_inverse=True)[1], weights=mm.astype(float))
            out.append(boot_ratio(num, den, n_boot, rng))
        rows.append(out)
    lines.append("\nConduction-positive verdicts still called positive, with 95% intervals.\n")
    lines.append(table(["Class", "50 bpm", "60 bpm", "80 bpm", "100 bpm"], rows))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reports", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    reports, classes = load(args.reports)
    head = [
        f"# Rhythm, rate, and lead audit: {Path(reports[0]['dataset_path']).name}\n",
        f"Seeds: {', '.join(str(r['seed']) for r in reports)}. "
        f"Exams: {len(reports[0]['exams'])}. "
        f"Reassembly check max |logit| error: "
        f"{max(r['reassembly_max_abs_error'] or 0 for r in reports):.1e}; "
        f"lead factorization check: "
        f"{max(r.get('lead_factorization_max_abs_error') or 0 for r in reports):.1e}.\n",
    ]
    sections = [
        descriptor_section(reports, classes),
        intervention_section(reports, classes),
        lead_section(reports, classes),
        region_section(reports, classes),
        ci_section(reports, classes, args.n_boot),
        diastolic_section(reports, classes, args.n_boot),
    ]
    args.output.write_text("\n".join(head + [s for s in sections if s]) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
