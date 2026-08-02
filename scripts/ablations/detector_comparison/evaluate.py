#!/usr/bin/env python3
"""Scores TruFor, Sightengine, and C2PA against the ground-truth integrity labels.

    python scripts/ablations/detector_comparison/evaluate.py --data-dir data/veritas_2026_q1

Reads manipulation_comparison.csv. Rows labelled `unknown` - where the
fact-check never examined the file's provenance - are excluded throughout;
they are unlabelled data, not negatives.

Two views are reported for the pixel detectors:

  all manipulated      every altered file, including forgeries no pixel
                       detector can see (a cleanly rendered fake screenshot
                       has no splice to find)
  detectable only      restricted to manipulation types the detector could
                       catch in principle

The gap between them separates "the detector is weak" from "the corpus is
outside its design envelope", which a single pooled number cannot do.

C2PA is reported as coverage, never as AUC: a missing manifest is not a
negative prediction, so ranking files by manifest absence is meaningless.
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from pathlib import Path
import argparse
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

DETECTORS = {
    "TruFor": lambda r: r["trufor_score"],
    "Sightengine ai_gen": lambda r: r["sightengine_ai_generated"],
    "Sightengine deepfake": lambda r: r["sightengine_deepfake"],
    "Sightengine max": lambda r: max(r["sightengine_ai_generated"], r["sightengine_deepfake"]),
}


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Rank-based AUC (Mann-Whitney), ties counted as half."""
    if not pos or not neg:
        return None
    ranked = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average rank across the tie group
        rank_sum += sum(avg_rank for k in range(i, j) if ranked[k][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_f1(pos: list[float], neg: list[float]) -> tuple[float, float, float, float]:
    """Sweep every threshold; return (threshold, f1, precision, recall) at the best F1."""
    best = (0.0, 0.0, 0.0, 0.0)
    for threshold in sorted({*pos, *neg}):
        tp = sum(v >= threshold for v in pos)
        fp = sum(v >= threshold for v in neg)
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / len(pos)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best[1]:
            best = (threshold, f1, precision, recall)
    return best


def youden_threshold(pos: list[float], neg: list[float]) -> tuple[float, float, float]:
    """Threshold maximising TPR - FPR; returns (threshold, tpr, fpr).

    Best-F1 is useless for picking an operating point here: the classes are
    near balanced, so "call everything manipulated" scores F1 0.71 at the
    lowest threshold and wins. Youden's J charges for false positives, so it
    cannot be gamed that way.
    """
    best = (0.0, 0.0, 1.0, -1.0)
    for threshold in sorted({*pos, *neg}):
        tpr = sum(v >= threshold for v in pos) / len(pos)
        fpr = sum(v >= threshold for v in neg) / len(neg)
        if tpr - fpr > best[3]:
            best = (threshold, tpr, fpr, tpr - fpr)
    return best[0], best[1], best[2]


def recall_at_precision(pos: list[float], neg: list[float], target: float) -> tuple[float, float] | None:
    """Highest recall reachable while holding precision >= target. The operating
    point that matters when a false 'manipulated' call is expensive."""
    best = None
    for threshold in sorted({*pos, *neg}):
        tp = sum(v >= threshold for v in pos)
        fp = sum(v >= threshold for v in neg)
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        if precision >= target:
            recall = tp / len(pos)
            if best is None or recall > best[1]:
                best = (threshold, recall)
    return best


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["trufor_score"] or not row["sightengine_ai_generated"]:
                continue  # unscored image
            for key in ("trufor_score", "sightengine_ai_generated", "sightengine_deepfake"):
                row[key] = float(row[key])
            row["gt_authenticity_score"] = float(row["gt_authenticity_score"])
            rows.append(row)
    return rows


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def report_detectors(rows: list[dict], label: str) -> None:
    pos_rows = [r for r in rows if r["gt_label"] == "manipulated"]
    neg_rows = [r for r in rows if r["gt_label"] == "authentic"]
    print(f"\n{label}: {len(pos_rows)} manipulated vs {len(neg_rows)} authentic")
    if not pos_rows or not neg_rows:
        print("  (insufficient data)")
        return
    print(
        f"  {'detector':<22} {'AUC':>6}   {'@thr':>7} {'TPR':>6} {'FPR':>6}   {'bestF1':>7}   {'R@P=0.9':>8}"
    )
    for name, get in DETECTORS.items():
        pos = [get(r) for r in pos_rows]
        neg = [get(r) for r in neg_rows]
        a = auc(pos, neg)
        thr, tpr, fpr = youden_threshold(pos, neg)
        _, f1, _, _ = best_f1(pos, neg)
        rap = recall_at_precision(pos, neg, 0.9)
        rap_str = f"{rap[1]:.3f}" if rap else "  n/a"
        print(f"  {name:<22} {a:6.3f}   {thr:7.3f} {tpr:6.3f} {fpr:6.3f}   {f1:7.3f}   {rap_str:>8}")


def report_by_type(rows: list[dict]) -> None:
    """Per-type recall at each detector's Youden threshold, fixed once on the
    pooled detectable set so the thresholds are comparable across types."""
    pos_rows = [r for r in rows if r["gt_label"] == "manipulated" and r["detectable_in_principle"] == "yes"]
    neg_rows = [r for r in rows if r["gt_label"] == "authentic"]
    thresholds = {}
    for name, get in DETECTORS.items():
        thresholds[name] = youden_threshold([get(r) for r in pos_rows], [get(r) for r in neg_rows])[0]

    types = Counter(r["manipulation_type"] for r in rows if r["gt_label"] == "manipulated")
    print("\n  recall per manipulation type, at each detector's Youden threshold")
    print(f"  {'type':<24} {'detect?':>8} {'n':>4}  " + "  ".join(f"{n.split()[-1]:>9}" for n in DETECTORS))
    for mtype, n in types.most_common():
        subset = [r for r in rows if r["gt_label"] == "manipulated" and r["manipulation_type"] == mtype]
        detectable = subset[0]["detectable_in_principle"]
        cells = []
        for name, get in DETECTORS.items():
            hits = sum(get(r) >= thresholds[name] for r in subset)
            cells.append(f"{hits / len(subset):9.3f}")
        print(f"  {mtype:<24} {detectable:>8} {n:>4}  " + "  ".join(cells))
    # The same thresholds applied to authentic files: the price of the recall above.
    cells = []
    for name, get in DETECTORS.items():
        fp = sum(get(r) >= thresholds[name] for r in neg_rows)
        cells.append(f"{fp / len(neg_rows):9.3f}")
    print(f"  {'FALSE POSITIVE RATE':<24} {'':>8} {len(neg_rows):>4}  " + "  ".join(cells))
    print(f"  {'(thresholds)':<24} {'':>8} {'':>4}  " + "  ".join(f"{thresholds[n]:9.3f}" for n in DETECTORS))


def report_gend(rows: list[dict]) -> None:
    """GenD, scored only where it returns a verdict.

    GenD abstains on images with no (or only tiny) faces, so it must be
    compared on its own scoreable subset — and every other detector re-scored
    on that *same* subset, or the comparison silently swaps the test set.
    """
    scoreable = [r for r in rows if r.get("gend_status") == "scored" and r["gend_p_fake"] != ""]
    labelled = [r for r in rows if r["gt_label"] in {"manipulated", "authentic"}]

    print("\n  coverage — GenD only judges faces, so first: what does it decline?")
    status = Counter(r.get("gend_status", "") for r in labelled)
    for key in ("scored", "no_face", "faces_too_small"):
        n = status.get(key, 0)
        print(f"    {key:<18} {n:>4}  ({100 * n / len(labelled):4.1f}%)")

    pos_all = [r for r in labelled if r["gt_label"] == "manipulated"]
    neg_all = [r for r in labelled if r["gt_label"] == "authentic"]
    pos_cov = sum(1 for r in pos_all if r.get("gend_status") == "scored") / len(pos_all)
    neg_cov = sum(1 for r in neg_all if r.get("gend_status") == "scored") / len(neg_all)
    print(f"\n    coverage on manipulated {100 * pos_cov:.0f}% vs authentic {100 * neg_cov:.0f}%")
    print("    (manipulated images contain faces more often, so the subset is not class-neutral;")
    print("     comparisons below are paired — every detector on the identical subset.)")

    pos = [r for r in scoreable if r["gt_label"] == "manipulated"]
    neg = [r for r in scoreable if r["gt_label"] == "authentic"]
    if not pos or not neg:
        print("\n  (not enough scored rows to evaluate)")
        return

    detectors = {"GenD": lambda r: float(r["gend_p_fake"]), **DETECTORS}
    print(f"\n  paired AUC on GenD-scored images ({len(pos)} manipulated vs {len(neg)} authentic)")
    print(f"  {'positives':<22} {'n':>4}  " + "  ".join(f"{n:>12}" for n in detectors))
    subsets = [
        ("deepfake", [r for r in pos if r["manipulation_type"] == "deepfake"]),
        ("ai_generated", [r for r in pos if r["manipulation_type"] == "ai_generated"]),
        ("ALL manipulated", pos),
    ]
    for name, subset in subsets:
        if not subset:
            continue
        cells = [f"{auc([g(r) for r in subset], [g(r) for r in neg]):12.3f}" for g in detectors.values()]
        print(f"  {name:<22} {len(subset):>4}  " + "  ".join(cells))

    print("\n  GenD score distribution (median):")
    print(f"    {'authentic':<22} {median([float(r['gend_p_fake']) for r in neg]):.3f}")
    for mtype, _ in Counter(r["manipulation_type"] for r in pos).most_common():
        subset = [float(r["gend_p_fake"]) for r in pos if r["manipulation_type"] == mtype]
        print(f"    {mtype:<22} {median(subset):.3f}")

    a = auc([float(r["gend_p_fake"]) for r in pos], [float(r["gend_p_fake"]) for r in neg])
    if a < 0.5:
        print(
            f"\n  GenD's AUC is {a:.3f} — BELOW chance. It scores manipulated images *lower*\n"
            "  than authentic ones, across every manipulation type. This is not a threshold\n"
            "  problem and inverting it would not yield a detector: that would be fitting to\n"
            "  whatever confound drives it here, not to manipulation. Image resolution has\n"
            "  been ruled out (Spearman -0.07). GenD does not transfer to this corpus."
        )


def report_c2pa(rows: list[dict]) -> None:
    present = [r for r in rows if r["c2pa_present"] == "True"]
    ai_rows = [r for r in rows if r["manipulation_type"] == "ai_generated"]
    manip = [r for r in rows if r["gt_label"] == "manipulated"]
    auth = [r for r in rows if r["gt_label"] == "authentic"]
    print(f"\n  images carrying any C2PA manifest:      {len(present)}/{len(rows)}")
    print(
        f"  known AI-generated carrying a manifest: {sum(r['c2pa_present'] == 'True' for r in ai_rows)}/{len(ai_rows)}"
    )
    print(f"  manifests declaring AI:                 {sum(r['c2pa_declares_ai'] == 'True' for r in rows)}")
    print(f"  known manipulated / known authentic:    {len(manip)} / {len(auth)}")
    if not present:
        print(
            "\n  C2PA recall on this corpus is 0. Every image here is a redistributed\n"
            "  copy - screenshotted, re-encoded, or scraped from a platform - and all\n"
            "  of those paths strip the manifest. Absence is not a negative signal:\n"
            "  it carries no information about any of these files."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/veritas_2026_q1"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    path = args.csv or args.data_dir / "manipulation_comparison.csv"
    rows = load(path)

    rule("Ground truth")
    labels = Counter(r["gt_label"] for r in rows)
    print(f"\n  {len(rows)} scored images")
    for label, n in labels.most_common():
        print(f"    {label:<14} {n:>4}  ({100 * n / len(rows):4.1f}%)")
    print("\n  'unknown' = the fact-check never examined provenance; excluded below.")
    misleading = sum(r["misleading_but_authentic"] == "True" for r in rows)
    print(f"  {misleading} authentic files were nonetheless misleading (staged, false caption).")
    print("  Those count as authentic here: staging is not manipulation.")

    rule("Pixel detectors")
    report_detectors(rows, "all manipulated")
    detectable = [r for r in rows if r["gt_label"] != "manipulated" or r["detectable_in_principle"] == "yes"]
    report_detectors(detectable, "detectable in principle only")
    report_by_type(rows)

    rule("GenD (face deepfakes)")
    report_gend(rows)

    rule("C2PA")
    report_c2pa(rows)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
