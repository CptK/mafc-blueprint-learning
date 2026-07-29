#!/usr/bin/env python3
"""Re-evaluate an existing benchmark run with the trained calibration model.

Takes a finished run directory (with `traces/`), extracts the lean feature table the
calibration regressor expects, predicts the certainty **magnitude**, keeps the judge's
**direction** as the sign, and produces calibrated predictions
(`final = sign(judge_dir) × magnitude`). It then writes a `calibrated_results/`
sub-directory with the same artifacts the original run has — 7-class and coarsened
3-class confusion matrices, a metrics report, and a summary — plus a per-claim
prediction file and a baseline-vs-calibrated comparison.

The judge's direction is unchanged; only the certainty magnitude is recalibrated. The
7-class label is obtained by binning the continuous prediction with the **canonical
VeriTaS rule** (`label_from_signed_score`), the same rule that defines the GT labels.

Usage
-----
    set -a; source config/.env; set +a          # OpenAI key for embeddings
    python scripts/training/apply_calibration.py \\
        --run out/veritas-2026_q1-7class-200-... \\
        --model out/training/run/magnitude_regressor.joblib \\
        --claims data/veritas_2026_q1
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
from mafc.eval.metrics import save_confusion_matrix_png
from mafc.eval.veritas.metrics import compute_veritas_metrics, format_veritas_metrics_report
from mafc.training.claims_io import load_many, resolve_claims_paths
from mafc.training.dataset import build_dataframe
from mafc.training.evaluate import baseline_continuous_metrics, continuous_signed_metrics
from mafc.training.features import FeatureExtractorConfig
from mafc.training.labels import (
    direction_of_label,
    label_from_signed_score,
    sign_of_direction,
)
from mafc.training.train import load_model, predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run dir containing traces/")
    ap.add_argument("--model", type=Path, required=True, help="magnitude_regressor.joblib")
    ap.add_argument("--claims", nargs="+", required=True, help="claims.json file(s) or dir(s)")
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    args = ap.parse_args()

    traces_dir = args.run / "traces"
    if not traces_dir.is_dir():
        ap.error(f"no traces/ dir in {args.run}")

    # 1. features (the calibration model requires the embedding features)
    claims_by_id = load_many(resolve_claims_paths([Path(p) for p in args.claims]))
    logger.info(f"Loaded {len(claims_by_id)} claims.")
    cfg = FeatureExtractorConfig(include_embeddings=True, embedding_model=args.embedding_model)
    feats = build_dataframe([traces_dir], claims_by_id, cfg)

    # Authoritative eval inputs (GT label, GT score, judge's predicted label) come from
    # the run's own results.jsonl so the baseline reproduces the original report exactly.
    res_rows = []
    for line in (args.run / "results.jsonl").open():
        d = json.loads(line)
        pred, gs = d.get("predicted"), d.get("gt_integrity_score")
        if pred in (None, "None", "") or gs in (None, "None", ""):
            continue
        res_rows.append(
            {
                "claim_id": str(d["claim_id"]),
                "ground_truth": d["ground_truth"],
                "judge_label": str(pred),
                "gt_score": float(gs),
            }
        )
    df = feats.merge(pd.DataFrame(res_rows), on="claim_id", how="inner").reset_index(drop=True)
    if df.empty:
        logger.error("No usable rows (need judge label + gt score). Nothing written.")
        sys.exit(1)
    logger.info(f"Feature table: {len(df)} claims.")

    # 2. predict magnitude, keep judge's sign
    model = load_model(args.model)
    feat_only = df[[c for c in df.columns if c not in ("ground_truth", "judge_label", "gt_score")]]
    mag = predict(model, feat_only)

    judge_labels = df["judge_label"].astype(str).tolist()
    judge_dirs = [direction_of_label(lbl) for lbl in judge_labels]
    gt_scores = df["gt_score"].astype(float).tolist()
    ground_truth_labels = df["ground_truth"].astype(str).tolist()
    cal_signed = [sign_of_direction(d) * float(m) for d, m in zip(judge_dirs, mag)]
    cal_labels = [label_from_signed_score(v) for v in cal_signed]
    true_labels = ground_truth_labels

    # 3. metrics: calibrated vs the judge's original (baseline)
    def payload(pred):
        return [
            {"ground_truth": t, "predicted": p, "gt_integrity_score": s}
            for t, p, s in zip(true_labels, pred, gt_scores)
        ]

    cal_metrics = compute_veritas_metrics(payload(cal_labels), label_scheme=7)
    base_metrics = compute_veritas_metrics(payload(judge_labels), label_scheme=7)
    cal_cont = continuous_signed_metrics(judge_dirs, np.asarray(mag), gt_scores)
    base_cont = baseline_continuous_metrics(judge_labels, gt_scores)

    # 4. write calibrated_results/
    out_dir = args.run / "calibrated_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = cal_metrics.get("confusion_matrix") or {}
    if cm:
        save_confusion_matrix_png(
            cm,
            out_dir / "confusion_matrix_7class.pdf",
            title="Confusion Matrix (7-class) — calibrated",
            subtitle=f"accuracy={cal_metrics.get('accuracy', 0):.1%}",
        )
    cm3 = (cal_metrics.get("coarsened_3class") or {}).get("confusion_matrix") or {}
    if cm3:
        save_confusion_matrix_png(
            cm3,
            out_dir / "confusion_matrix_3class_coarsened.pdf",
            title="Confusion Matrix (3-class coarsened) — calibrated",
            subtitle=f"accuracy={cal_metrics['coarsened_3class'].get('accuracy', 0):.1%}",
        )

    (out_dir / "metrics_report.txt").write_text(
        format_veritas_metrics_report(cal_metrics, label_scheme=7), encoding="utf-8"
    )

    with (out_dir / "calibrated_predictions.jsonl").open("w") as fh:
        for cid, t, jl, jd, m, cs, cl in zip(
            df["claim_id"], true_labels, judge_labels, judge_dirs, mag, cal_signed, cal_labels
        ):
            fh.write(
                json.dumps(
                    {
                        "claim_id": cid,
                        "ground_truth": t,
                        "judge_label": jl,
                        "judge_direction": jd,
                        "magnitude": float(m),
                        "calibrated_score": float(cs),
                        "calibrated_label": cl,
                        "correct": bool(cl == t),
                    }
                )
                + "\n"
            )

    summary = {
        "n": len(df),
        "calibrated": {
            "accuracy_7class": cal_metrics.get("accuracy"),
            "macro_f1_7class": cal_metrics.get("macro", {}).get("f1"),
            "accuracy_3class": cal_metrics.get("coarsened_3class", {}).get("accuracy"),
            "continuous_mse": cal_cont["mse"],
            "continuous_mae": cal_cont["mae"],
        },
        "baseline_judge": {
            "accuracy_7class": base_metrics.get("accuracy"),
            "macro_f1_7class": base_metrics.get("macro", {}).get("f1"),
            "accuracy_3class": base_metrics.get("coarsened_3class", {}).get("accuracy"),
            "continuous_mse": base_cont["mse"],
            "continuous_mae": base_cont["mae"],
        },
        "model": str(args.model),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    c, b = summary["calibrated"], summary["baseline_judge"]
    print("\n=== Calibrated vs. baseline (judge) ===")
    print(f"  continuous MSE   baseline={b['continuous_mse']:.4f}   calibrated={c['continuous_mse']:.4f}")
    print(f"  continuous MAE   baseline={b['continuous_mae']:.4f}   calibrated={c['continuous_mae']:.4f}")
    print(f"  7-class accuracy baseline={b['accuracy_7class']:.2%}   calibrated={c['accuracy_7class']:.2%}")
    print(f"  3-class accuracy baseline={b['accuracy_3class']:.2%}   calibrated={c['accuracy_3class']:.2%}")
    print(
        f"\nWrote: {out_dir}/  (confusion matrices, metrics_report.txt, summary.json, calibrated_predictions.jsonl)"
    )


if __name__ == "__main__":
    main()
