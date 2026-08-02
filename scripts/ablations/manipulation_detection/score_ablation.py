#!/usr/bin/env python3
"""Scores the manipulation-detection ablation: baseline vs Sightengine vs oracle.

    python scripts/ablations/manipulation_detection/score_ablation.py
    python scripts/ablations/manipulation_detection/score_ablation.py --judge opus5
    python scripts/ablations/manipulation_detection/score_ablation.py --judge both --by-integrity

Three arms differ only in which detector the media agent is given. Everything is
compared PAIRED on the claims scored in all three arms — claim difficulty varies
far more than the arms do, so the unpaired difference is mostly between-claim
noise.

Two judges can be scored:

  gemini  the verdict recorded by the run itself (results.jsonl)
  opus5   a re-judge of the same stored evidence by a stronger model, produced by
          scripts/rejudge_traces.py (out/rejudge_opus5/<arm>.jsonl)

The second answers "is the null just a judge too weak to use the evidence?".

NOTE ON REJUDGE FILES: a claim can appear more than once — an early failed attempt
(HTTP 400) followed by a successful retry. Only rows carrying a parsed label are
kept, last one wins, so a partial run that was resumed scores correctly.
"""

from __future__ import annotations

from math import comb
from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

# The 7-point scale mapped to evenly spaced values in [-1, 1].
#
# Written as explicit fractions rather than np.linspace: linspace lands 1 ULP off
# symmetric (+1/3 and -1/3 differ in magnitude), so two arms equidistant from the
# truth on opposite sides — an exact tie — get scored as a win for one of them.
# Negation in IEEE 754 is exact, so this construction is a perfect mirror.
LABELS = [
    "compromised (certain)",
    "compromised (rather certain)",
    "compromised (rather uncertain)",
    "unknown",
    "intact (rather uncertain)",
    "intact (rather certain)",
    "intact (certain)",
]
VALUE = {label: v for label, v in zip(LABELS, [-1.0, -2 / 3, -1 / 3, 0.0, 1 / 3, 2 / 3, 1.0])}

# Squared errors that differ only by floating-point noise are ties, not wins.
_TIE_TOLERANCE = 1e-12

# Run directories, one per arm. Renaming a run directory is safe for this script
# (it reads results.jsonl by path) but NOT for a rejudge, which resolves media
# through the run's own ezmm registry — see the reproduction notes.
DEFAULT_RUNS = {
    "baseline": "out/veritas-2026_q1-7class-20260728-eom_v4_hybrid",
    "sightengine": "out/veritas-2026_q1-7class-20260801-eom_v4_media-only_sightengine",
    "oracle": "out/veritas-2026_q1-7class-20260801-eom_v4_media-only_oracle",
}
DEFAULT_REJUDGE_DIR = "out/rejudge_opus5"
COMPARISONS = [("sightengine", "baseline"), ("oracle", "baseline"), ("sightengine", "oracle")]
N_BOOTSTRAP = 20_000


def coarsen(label: str) -> str:
    """Collapse the 7-point scale to direction only."""
    return "intact" if "intact" in label else ("compromised" if "compromised" in label else "unknown")


def load_run_predictions(run_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (predictions, ground_truth) keyed by claim id, from a run's results."""
    predictions, truth = {}, {}
    with open(Path(run_dir) / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            claim_id = str(row["claim_id"])
            if row.get("predicted") in VALUE:
                predictions[claim_id] = row["predicted"]
            if row.get("ground_truth") in VALUE:
                truth[claim_id] = row["ground_truth"]
    return predictions, truth


def load_rejudge_predictions(path: Path) -> dict[str, str]:
    """Last successfully parsed label per claim (see the module docstring)."""
    predictions = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("predicted") in VALUE:
                predictions[str(row["claim_id"])] = row["predicted"]
    return predictions


def squared_error(predicted: str, truth: str) -> float:
    return float((VALUE[predicted] - VALUE[truth]) ** 2)


def sign_test(differences: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact sign test over the claims whose error actually changed."""
    better = int((differences < -_TIE_TOLERANCE).sum())
    worse = int((differences > _TIE_TOLERANCE).sum())
    n = better + worse
    if n == 0:
        return better, worse, 1.0
    k = min(better, worse)
    return better, worse, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on the discordant pairs. Returns (a_only, b_only, p)."""
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    return a_only, b_only, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2**n))


def bootstrap_ci(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = np.array(
        [rng.choice(differences, len(differences), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report_arms(preds: dict[str, dict[str, str]], truth: dict[str, str], ids: list[str], title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}  (n={len(ids)})\n{'=' * 72}")
    print(f"  {'arm':<14}{'MSE':>9}{'MAE':>9}{'Acc.7':>9}{'Acc.3':>9}")
    for arm in preds:
        p = [preds[arm][i] for i in ids]
        t = [truth[i] for i in ids]
        mse = np.mean([squared_error(a, b) for a, b in zip(p, t)])
        mae = np.mean([abs(VALUE[a] - VALUE[b]) for a, b in zip(p, t)])
        acc7 = np.mean([a == b for a, b in zip(p, t)]) * 100
        acc3 = np.mean([coarsen(a) == coarsen(b) for a, b in zip(p, t)]) * 100
        print(f"  {arm:<14}{mse:9.4f}{mae:9.4f}{acc7:8.2f}%{acc3:8.2f}%")


def report_pairs(
    preds: dict[str, dict[str, str]], truth: dict[str, str], ids: list[str], rng: np.random.Generator
) -> None:
    print("\n  paired differences in squared error (negative favours the first arm):")
    for first, second in COMPARISONS:
        d = np.array(
            [
                squared_error(preds[first][i], truth[i]) - squared_error(preds[second][i], truth[i])
                for i in ids
            ]
        )
        low, high = bootstrap_ci(d, rng)
        better, worse, p = sign_test(d)
        label = f"{first} vs {second}"
        print(
            f"    {label:<28} dMSE {d.mean():+.4f}  CI[{low:+.4f},{high:+.4f}]"
            f"  better {better:3} worse {worse:3} same {len(d) - better - worse:3}  p={p:.3g}"
        )

    print("\n  exact-label accuracy, McNemar vs baseline:")
    base = np.array([preds["baseline"][i] == truth[i] for i in ids])
    for arm in preds:
        if arm == "baseline":
            continue
        arr = np.array([preds[arm][i] == truth[i] for i in ids])
        arm_only, base_only, p = mcnemar(arr, base)
        print(
            f"    {arm:<28} {base.mean() * 100:.2f}% -> {arr.mean() * 100:.2f}% "
            f"({(arr.mean() - base.mean()) * 100:+.2f}pp)  +{arm_only}/-{base_only}  p={p:.4g}"
        )


def report_by_integrity(
    preds: dict[str, dict[str, str]],
    truth: dict[str, str],
    ids: list[str],
    data_dir: Path,
    rng: np.random.Generator,
) -> None:
    """Split by the image's true integrity label. Post-hoc and underpowered —
    read the point estimates as observations, not findings."""
    comparison_csv = data_dir / "manipulation_comparison.csv"
    if not comparison_csv.exists():
        print(f"\n  (no {comparison_csv}; skipping the integrity split)")
        return
    integrity = {
        row["claim_id"]: row["gt_label"]
        for row in csv.DictReader(open(comparison_csv, encoding="utf-8"))
        if row["gt_label"] in ("manipulated", "authentic")
    }
    print("\n  by the image's true integrity (post-hoc, underpowered):")
    for want in ("manipulated", "authentic"):
        subset = [i for i in ids if integrity.get(i) == want]
        if not subset:
            continue
        base_mse = np.mean([squared_error(preds["baseline"][i], truth[i]) for i in subset])
        line = f"    {want:<12} n={len(subset):3}  baseline {base_mse:.4f}"
        for arm in preds:
            if arm == "baseline":
                continue
            d = np.array(
                [
                    squared_error(preds[arm][i], truth[i]) - squared_error(preds["baseline"][i], truth[i])
                    for i in subset
                ]
            )
            low, high = bootstrap_ci(d, rng)
            arm_mse = np.mean([squared_error(preds[arm][i], truth[i]) for i in subset])
            line += f" | {arm} {arm_mse:.4f} ({d.mean():+.4f} CI[{low:+.3f},{high:+.3f}])"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--judge", choices=("gemini", "opus5", "both"), default="both")
    parser.add_argument("--rejudge-dir", default=DEFAULT_REJUDGE_DIR)
    parser.add_argument("--data-dir", default="data/veritas_2026_q1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    gemini, truth = {}, {}
    for arm, run_dir in DEFAULT_RUNS.items():
        gemini[arm], arm_truth = load_run_predictions(run_dir)
        for claim_id, label in arm_truth.items():
            truth.setdefault(claim_id, label)

    judges: dict[str, dict[str, dict[str, str]]] = {}
    if args.judge in ("gemini", "both"):
        judges["Gemini (the runs' own judge)"] = gemini
    if args.judge in ("opus5", "both"):
        rejudge = {}
        for arm in DEFAULT_RUNS:
            path = Path(args.rejudge_dir) / f"{arm}.jsonl"
            if not path.exists():
                print(f"missing {path} — run scripts/rejudge_traces.py for this arm first")
                sys.exit(1)
            rejudge[arm] = load_rejudge_predictions(path)
        judges["Opus 5 re-judge"] = rejudge

    # One claim set across every judge, so the panels are directly comparable.
    ids = set(truth)
    for preds in judges.values():
        for arm_preds in preds.values():
            ids &= set(arm_preds)
    ids = sorted(ids)
    print(f"paired claim set: n={len(ids)} (scored in every arm under every judge shown)")

    for title, preds in judges.items():
        report_arms(preds, truth, ids, title)
        report_pairs(preds, truth, ids, rng)
        report_by_integrity(preds, truth, ids, Path(args.data_dir), rng)

    if len(judges) == 2:
        a, b = judges.values()
        agreement = np.mean([a["baseline"][i] == b["baseline"][i] for i in ids]) * 100
        print(f"\n  inter-judge agreement on the baseline arm (exact label): {agreement:.1f}%")


if __name__ == "__main__":
    main()
