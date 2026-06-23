# Magnitude-regressor calibration subsystem (`mafc/training`)

A standalone discriminative regressor that attempts to recalibrate the judge's
**certainty magnitude** while leaving the **direction** (intact / unknown /
compromised) to the LLM judge.

> ## ⚠️ OUTCOME: NEGATIVE RESULT — the approach does not work
> The trace-feature regressor has **no generalizable signal** for certainty: it
> collapses to a near-constant output and ties/loses to an "always-certain" baseline.
> Confirmed model-agnostically (logistic / GBM / MLP all ~0.58 CV AUC on the
> certain-vs-rather-certain split). The certainty miscalibration is **not learnable
> from execution-trace features.** See [`REPORT.md`](REPORT.md) for the full evidence.
> The code below is retained as the record of the investigation.

## Design

Ground truth is a continuous `integrity.score ∈ [-1, 1]`; the 7-class labels are a
quantization of it (`mafc/eval/veritas/labels.py::THRESHOLDS_7`).

- The **judge owns the sign**: its predicted direction is taken as ground for the
  side of the decision.
- The **regressor owns the magnitude**: it predicts `|integrity.score| ∈ [0, 1]`
  from execution-trace features. The judge's *predicted direction* is itself an
  input feature.
- **Deliverable is continuous**: `final_score = sign(judge_direction) × magnitude`
  ∈ [-1, 1], scored by MSE against the continuous GT. A 7-class label can be produced
  for diagnostics by binning that value with the **canonical VeriTaS rule**
  (`mafc.eval.veritas` `THRESHOLDS_7` via `label_from_signed_score`) — the same
  function that defines the GT labels. No custom/tuned thresholds.

One regressor, target `|integrity.score|`. No ground-truth-derived feature is ever
fed to the model (no `true_label`, no signed score) — only the regression target
uses ground truth.

> **Current canonical feature set, model, and results live in
> [`REPORT.md`](REPORT.md).** The feature set has
> since been pruned to a lean, blueprint-independent set and the justification
> embedding is PCA-reduced in-fold; some sections below describe the original
> broader extractor.

## Pipeline & commands

All three steps have thin CLIs under `scripts/training/`. The library code is pure
and unit-tested (`tests/training/`).

### 1. Sample claims (boundary-weighted, stratified)

```bash
python scripts/training/sample_claims.py \
    --claims data/veritas_2025_q2_with_fact_checks \
             data/veritas_2025_q3_with_fact_checks \
             data/veritas_2025_q4_with_fact_checks \
    --target-n 500 --hard-weight 3.0 --seed 0 \
    --out out/training/sample_q2q4
```

Outputs:
- `*.sample_ids.yaml` — paste the `benchmark.sample_ids` block into an experiment
  config (verified loadable by `mafc.eval.run_config.BenchmarkConfig`).
- `*.manifest.{csv,json}` — `id, score, direction, stratum, weight`.

Stratifies by direction (`|score| < 1/3` ⇒ unknown band), oversamples the
`certain / rather-certain` confusion region (`|score| ∈ [0.5, 1.0]`), and is
deterministic for a fixed seed.

### 2. Build the feature table

```bash
# fast structured-only table
python scripts/training/build_features.py \
    --traces out/<run>/traces \
    --claims data/veritas_2025_q4_with_fact_checks \
    --out out/training/features

# add embedding features (justification embedding + evidence dispersion);
# requires the OpenAI client / network
python scripts/training/build_features.py \
    --traces out/<run>/traces --claims data/veritas_2025_q4_with_fact_checks \
    --embeddings --out out/training/features
```

Writes `<out>.csv` (always) and `<out>.parquet` (only if a parquet engine is
installed — see notes), plus `<out>_meta.csv` carrying evaluation-only ground truth
(`meta__gt_score`, `meta__judge_label`, `meta__date`) kept strictly out of the
feature set.

### 3. Train / evaluate

```bash
python scripts/training/train_eval.py \
    --features out/training/features.csv \
    --meta out/training/features_meta.csv \
    --out-dir out/training/run \
    --cv kfold --folds 5
```

Runs CV (`--cv kfold|temporal`); reports the **primary** continuous-deliverable MSE/MAE
(`sign × magnitude`) plus diagnostics — 7/3-class accuracy, ECE + reliability, and the
focused `certain ↔ rather-certain` confusion **vs. the judge baseline** — with the
prediction binned by the canonical VeriTaS rule (no tuned thresholds). Also computes a
learning curve and saves the final model. Outputs in `--out-dir`:
`magnitude_regressor.joblib`, `cv.json`, `metrics_{regressor,baseline}.json`,
`learning_curve.json`, `report.txt`. (Feature importance: separate
`scripts/training/feature_importance.py`.)

### 4. Apply a trained model to an existing benchmark run

```bash
set -a; source config/.env; set +a   # OpenAI key for embeddings
python scripts/training/apply_calibration.py \
    --run out/veritas-2026_q1-... \
    --model out/training/run/magnitude_regressor.joblib \
    --claims data/veritas_2026_q1
```

Extracts features from the run's `traces/`, predicts the magnitude, keeps the judge's
direction as the sign, and writes a `calibrated_results/` sub-directory in the run:
7-class and coarsened 3-class confusion-matrix PDFs, `metrics_report.txt`,
`summary.json` (calibrated vs. judge baseline), and `calibrated_predictions.jsonl`.
The baseline is read from the run's own `results.jsonl` so it reproduces the original
report exactly. (This is how the negative result was confirmed on held-out 2026 Q1.)

## ML library

Uses **scikit-learn** `HistGradientBoostingRegressor` (already in
`requirements.txt`) — handles NaNs and native categoricals, no new gradient-boosting
dependency. Model persistence via `joblib` (ships with scikit-learn). **No additions
to `requirements.txt` were needed.**

`parquet` output is best-effort: neither `pyarrow` nor `fastparquet` is installed in
this environment, so only CSV is written (the code degrades gracefully and logs a
note). Install `pyarrow` to enable parquet.

## Trace fields: used vs. planned-but-unavailable

There is **no separate `*.judge_trace.json`** on disk. The judge trace is embedded
as `judge_run` inside each `*.fact_check_trace.json` (blueprint runs) and
`*.strategy_trace.json` (strategy runs); both shapes are read via
`trace_io.normalise_trace`. `discover_traces` prefers the richest trace per claim.

**Lean feature set actually emitted** (13 structured; blueprint-coupled and dead
features were pruned — see `REPORT.md`):

| Feature | Trace / claim source |
|---|---|
| `judge_direction` (conditioning) | `judge_run.decision.label` → coarsened |
| `evidence_count`, `useful_ratio`, `n_distinct_domains` | `judge_run.summary.result.evidences[]` (+ domain parse) |
| `n_iterations`, `evidence_growth_total` | `iterations[]` / `rounds[]` |
| `runtime_seconds` | `summary.runtime_seconds` |
| `justification_char_len` | `judge_run.decision.justification` |
| `judge_output_tokens` | `judge_run.summary.total_output_tokens` |
| `has_media`, `language`, `claim_char_len` | `claims.json` |
| `claim_vs_evidence_months` | claim `date` vs. dates parsed from evidence takeaways |

Embedding features (with `--embeddings`): `emb_disp_*`, `emb_ev_claim_*` (evidence
agreement) and the justification embedding.

**Notes (handled gracefully):**

- `summary.runtime_seconds` is **null on strategy traces** ⇒ emitted as NaN there.
- Embedding features are **optional** (`--embeddings`, network call); the
  structured-only table is the fast default. The justification embedding is emitted
  in full (3072 dims) and **PCA-reduced to 32 components inside the model pipeline**
  (`train.py`, refit per CV fold → no leakage).
- There is **no separate `*.judge_trace.json`**: the judge trace is embedded as
  `judge_run` inside `*.fact_check_trace.json` / `*.strategy_trace.json`.

## Notes on the label space

- Both the **GT label** and the **prediction** are binned from the signed score by the
  canonical VeriTaS rule (`mafc.eval.veritas` `THRESHOLDS_7`, via
  `label_from_signed_score`, cuts ±1/6, ±1/2, ±5/6). No tuned thresholds.
- `direction_of_label` coarsens the judge's 7-class label to a direction; `intact
  (rather uncertain)` / `compromised (rather uncertain)` coarsen to **unknown** per the
  existing `COARSEN_7_TO_3`. The judge direction is taken from this coarsening, so a
  judge "rather uncertain" verdict is treated as the unknown side.

## Results — negative

See **[`REPORT.md`](REPORT.md)** for the full evidence. Summary: the model collapses to
a near-constant magnitude (predicted std ~0.02 vs. true ~0.165; Pearson 0.20 on train →
0.02 on held-out 2026 Q1) and **ties/loses to an "always-certain" baseline** (2025:
69.57% vs. 69.98%; 2026: 62.76% vs. 63.78%). Apparent gains in accuracy/ECE were
artifacts of the 76%-`certain` class imbalance (ECE measures aggregate calibration, not
discrimination). The certain-vs-rather-certain distinction is **not learnable from these
trace features by any model class** (logistic / GBM / MLP all ~0.58 CV AUC).
