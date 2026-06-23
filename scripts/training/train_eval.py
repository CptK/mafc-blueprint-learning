#!/usr/bin/env python3
"""Train the magnitude regressor, tune per-side thresholds, and evaluate.

Consumes the feature table + meta table produced by ``build_features.py``. Runs:

  1. CV (k-fold or temporal) for an honest out-of-fold magnitude estimate.
  2. PRIMARY: MSE/MAE of the continuous deliverable (sign(judge) × magnitude).
  3. Diagnostics: 7-class / 3-class / calibration (ECE), with the continuous
     prediction binned by the canonical VeriTaS rule (``label_from_signed_score``)
     — the same function that defines the GT labels, so no tuned thresholds.
  4. A learning curve (error vs. training-set size).
  5. A final model fit on all data, saved to disk.

Usage
-----
    python scripts/training/train_eval.py \\
        --features out/training/features.csv \\
        --meta out/training/features_meta.csv \\
        --out-dir out/training/run \\
        --cv kfold --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from mafc.common.logger import logger
from mafc.training.dataset import META_PREFIX
from mafc.training.evaluate import (
    baseline_continuous_metrics,
    baseline_magnitudes,
    compare_reports,
    continuous_signed_metrics,
    evaluate,
)
from mafc.training.labels import (
    direction_of_label,
    label_from_signed_score,
    sign_of_direction,
)
from mafc.training.train import (
    TrainConfig,
    cross_validate,
    fit,
    learning_curve,
    save_model,
)


def _load(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cv", choices=["kfold", "temporal"], default="kfold")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = _load(args.features)
    meta = _load(args.meta)
    df = features.merge(meta, on="claim_id", how="inner")
    logger.info(f"Merged table: {len(df)} rows.")

    gt_col = f"{META_PREFIX}gt_score"
    judge_col = f"{META_PREFIX}judge_label"
    df = df[df[judge_col].notna() & df[gt_col].notna()].reset_index(drop=True)
    if len(df) < 4:
        logger.error(f"Too few usable rows ({len(df)}) for train/eval.")
        sys.exit(1)

    feat_only = df[[c for c in df.columns if not c.startswith(META_PREFIX)]]
    cfg = TrainConfig(random_state=args.seed)

    # 1. CV for honest OOF magnitude
    date_col = f"{META_PREFIX}date"
    date_series = df[date_col] if args.cv == "temporal" and date_col in df else None
    if args.cv == "temporal" and (date_series is None or date_series.isna().all()):
        logger.warning("No usable date column for temporal CV; falling back to kfold.")
        args.cv = "kfold"
        date_series = None
    cv = cross_validate(feat_only, cfg, k=args.folds, mode=args.cv,
                        date_series=date_series, seed=args.seed)
    oof = cv["oof"]
    valid = ~np.isnan(oof)
    logger.info(f"CV ({args.cv}) overall: {cv['overall']}")

    judge_labels = df[judge_col].astype(str).tolist()
    judge_dirs = [direction_of_label(lbl) for lbl in judge_labels]
    gt_scores = df[gt_col].astype(float).tolist()
    # Ground-truth and prediction are binned by the *same* canonical VeriTaS rule
    # (mafc.eval.veritas THRESHOLDS_7, via label_from_signed_score) — no tuned
    # thresholds. The deliverable itself is the continuous sign×magnitude.
    true_labels_7 = [label_from_signed_score(s) for s in gt_scores]
    mags = np.where(valid, oof, 0.0)

    # 3a. PRIMARY metric: MSE of the continuous deliverable (sign(judge) × magnitude).
    cont = continuous_signed_metrics(judge_dirs, mags, gt_scores)
    base_cont = baseline_continuous_metrics(judge_labels, gt_scores)
    cont_report = (
        "=== PRIMARY: continuous deliverable MSE (sign × magnitude, no binning) ===\n"
        f"  MSE  baseline(judge 7-value)={base_cont['mse']:.4f}   "
        f"regressor(continuous)={cont['mse']:.4f}\n"
        f"  MAE  baseline={base_cont['mae']:.4f}   regressor={cont['mae']:.4f}   (n={cont['n']})\n"
    )

    # 3b. Secondary diagnostics: 7-class / calibration. The continuous prediction is
    # binned by the canonical VeriTaS rule — the same function that defines the GT label.
    signed_pred = [sign_of_direction(d) * m for d, m in zip(judge_dirs, mags)]
    pred_labels_7 = [label_from_signed_score(v) for v in signed_pred]
    model_metrics = evaluate(true_labels_7, pred_labels_7, gt_scores, mags)
    base_metrics = evaluate(
        true_labels_7, judge_labels, gt_scores, baseline_magnitudes(judge_labels)
    )
    report = cont_report + "\n" + compare_reports(base_metrics, model_metrics)
    model_metrics["continuous"] = cont
    base_metrics["continuous"] = base_cont
    logger.info("\n" + report)

    # 4. Learning curve
    curve = learning_curve(feat_only, cfg, seed=args.seed)

    # 5. Final fit + save
    final = fit(feat_only, cfg)
    model_path = out_dir / "magnitude_regressor.joblib"
    save_model(final, model_path)

    (out_dir / "cv.json").write_text(json.dumps(
        {"mode": cv["mode"], "k": cv["k"], "folds": cv["folds"], "overall": cv["overall"]},
        indent=2))
    (out_dir / "metrics_regressor.json").write_text(json.dumps(model_metrics, indent=2, default=float))
    (out_dir / "metrics_baseline.json").write_text(json.dumps(base_metrics, indent=2, default=float))
    (out_dir / "learning_curve.json").write_text(json.dumps(curve, indent=2))
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    logger.info(f"Saved model + reports to {out_dir}")


if __name__ == "__main__":
    main()
