"""Grouped permutation importance for the magnitude regressor.

``HistGradientBoostingRegressor`` exposes no native feature importances, so we use
permutation importance: for each CV fold we fit on the train split, measure the
out-of-fold magnitude MAE, then shuffle a feature (or a whole feature *family*)
in the validation split and measure how much MAE degrades. Importance = mean MAE
increase across folds × repeats. Families (e.g. the PCA'd justification-embedding
block) are permuted jointly so correlated columns are credited together.

Usage
-----
    python scripts/training/feature_importance.py \
        --features out/training/pilot500_features_embfull.csv \
        --out out/training/pilot500_run_embpca/feature_importance.json \
        --folds 5 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from mafc.training.features import TARGET_COL
from mafc.training.train import (
    TrainConfig,
    _fold_indices_kfold,
    feature_columns,
    fit,
    predict,
)

# Family assignment by exact name or prefix. First match wins; anything unmatched
# falls into "other".
_FAMILIES: list[tuple[str, list[str], list[str]]] = [
    # (family, exact_names, prefixes)
    ("direction_conditioning", ["judge_direction", "judge_label_known"], []),
    (
        "evidence_sufficiency",
        ["evidence_count", "n_useful_evidence", "useful_ratio", "n_distinct_domains"],
        [],
    ),
    (
        "search_struggle",
        [
            "n_iterations", "hit_max_iterations", "n_delegated_tasks", "n_errors",
            "retrieval_failures", "retrieval_failure_rate", "evidence_growth_total",
            "evidence_growth_steps", "runtime_seconds", "total_calls",
        ],
        [],
    ),
    (
        "judge_hedging",
        [
            "justification_char_len", "justification_word_len", "hedge_count",
            "hedge_density", "judge_output_tokens", "judge_repair_fired",
            "judge_errors_present",
        ],
        [],
    ),
    ("evidence_agreement_emb", [], ["emb_disp_", "emb_ev_claim_"]),
    ("justification_semantics_emb", ["just_emb_norm"], ["just_emb_"]),
    (
        "difficulty_priors",
        [
            "has_media", "n_media", "language", "claim_char_len", "claim_word_len",
            "blueprint_name", "claim_vs_evidence_months",
        ],
        ["cf_"],
    ),
]


def _family_of(col: str) -> str:
    for fam, exact, prefixes in _FAMILIES:
        if col in exact or any(col.startswith(p) for p in prefixes):
            return fam
    return "other"


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def grouped_permutation_importance(
    df: pd.DataFrame, k: int, repeats: int, seed: int
) -> dict:
    feature_cols, _ = feature_columns(df)
    # Map each feature column to its family, and build per-family + per-(structured)
    # column groups. The two embedding blocks are permuted as whole families only.
    fam_to_cols: dict[str, list[str]] = {}
    for c in feature_cols:
        fam_to_cols.setdefault(_family_of(c), []).append(c)

    # Individual columns to score: all non-embedding columns (embedding raw dims are
    # too numerous/individually meaningless — only their family is scored).
    emb_block = re.compile(r"^just_emb_\d+$|^emb_disp_|^emb_ev_claim_")
    indiv_cols = [c for c in feature_cols if not emb_block.match(c)]

    df = df.reset_index(drop=True)
    n = len(df)
    folds = _fold_indices_kfold(n, k, seed)
    cfg = TrainConfig()
    rng = np.random.default_rng(seed)

    fam_deltas: dict[str, list[float]] = {f: [] for f in fam_to_cols}
    col_deltas: dict[str, list[float]] = {c: [] for c in indiv_cols}
    base_maes: list[float] = []

    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = fit(df.iloc[train_idx], cfg)
        val = df.iloc[test_idx].reset_index(drop=True)
        y = val[TARGET_COL].to_numpy(dtype=float)
        base = _mae(y, predict(model, val))
        base_maes.append(base)

        def _perm_delta(cols: list[str]) -> float:
            deltas = []
            for _ in range(repeats):
                shuffled = val.copy()
                perm = rng.permutation(len(shuffled))
                for col in cols:
                    shuffled[col] = shuffled[col].to_numpy()[perm]
                deltas.append(_mae(y, predict(model, shuffled)) - base)
            return float(np.mean(deltas))

        for fam, cols in fam_to_cols.items():
            fam_deltas[fam].append(_perm_delta(cols))
        for col in indiv_cols:
            col_deltas[col].append(_perm_delta([col]))

    def _summ(d: dict[str, list[float]]) -> list[dict]:
        out = [
            {
                "name": k_,
                "importance_mae_increase": float(np.mean(v)) if v else 0.0,
                "std": float(np.std(v)) if v else 0.0,
            }
            for k_, v in d.items()
        ]
        return sorted(out, key=lambda r: r["importance_mae_increase"], reverse=True)

    return {
        "baseline_mae": float(np.mean(base_maes)) if base_maes else None,
        "by_family": _summ(fam_deltas),
        "by_feature_structured": _summ(col_deltas),
        "n_folds": k,
        "repeats": repeats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    result = grouped_permutation_importance(df, args.folds, args.repeats, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"baseline magnitude MAE: {result['baseline_mae']:.4f}\n")
    print("=== Importance by feature family (MAE increase when permuted) ===")
    for r in result["by_family"]:
        print(f"  {r['name']:<28s} {r['importance_mae_increase']:+.4f}  (±{r['std']:.4f})")
    print("\n=== Top 15 individual structured features ===")
    for r in result["by_feature_structured"][:15]:
        print(f"  {r['name']:<32s} {r['importance_mae_increase']:+.4f}  (±{r['std']:.4f})")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
