# Certainty-Calibration Report

**Date:** 2026-06-23
**Goal:** The VeriTaS LLM judge predicts *direction* (intact/unknown/compromised) well — 3-bin accuracy ≈ 90% — but its *certainty* is miscalibrated (heavy `certain ↔ rather-certain` confusion). We keep the judge for the sign and add a separate model that predicts the certainty **magnitude**; the deliverable is the resulting continuous score in [-1, 1].

## ⚠️ OUTCOME: NEGATIVE RESULT — the approach does not work

The trace-feature magnitude regressor **has essentially no discriminative power for certainty.** It collapses to a near-constant ≈0.91 output and **ties-or-loses to a trivial "always predict certain" baseline**. The certainty miscalibration is **not learnable from these execution-trace features**; we are stopping this approach.

Decisive diagnostics (the ones that matter — *not* ECE/accuracy, which are misleading under class imbalance):

| | 2025 (train, OOF) | 2026 Q1 (held-out) |
|---|---|---|
| Predicted magnitude std (true ≈ 0.165) | 0.027 | **0.021** |
| Pearson(pred, true \|score\|) | 0.196 | **0.024** |
| Spearman | 0.18 | 0.05 |
| Mean pred: true-certain vs true-rather-certain | — | 0.915 vs 0.914 (identical) |
| **"always-certain" accuracy** | **69.98%** | **63.78%** |
| **model accuracy** | 69.57% | 62.76% |

The model is **slightly worse than a constant predictor**. The weak 2025 signal (r≈0.2) is overfit and vanishes out-of-sample (r≈0).

**Why:** (1) severe class imbalance (76% `certain`, mean |score| 0.904) + squared-error loss ⇒ predicting ~0.9 everywhere is near-optimal, so the model barely splits; (2) the features genuinely don't carry the `certain` vs `rather-certain` signal (permutation importances were ~0.002 — that was the warning).

**It is a data/signal problem, not a model problem** (confirmed model-agnostically). Binary classification `certain` vs `rather-certain` (n=494, the exact failing distinction), 5-fold CV AUC: Logistic **0.579**, HistGB **0.571**, MLP(64,32) **0.589**, Dummy 0.514 — all statistically the same, barely above chance. **Train (in-sample) AUC = 1.000** for GBM/MLP: they *memorize* the training set but none generalize (the whisper of 2025 signal is r≈0 on held-out 2026). When linear, tree, and neural models all converge to ~0.58 CV AUC, the ceiling is the information in the features, not model capacity — a bigger model just overfits harder.

**Why the earlier "wins" were artifacts:** accuracy rewards predicting the majority class; **ECE is low because a constant at the base rate is "calibrated in aggregate" — it measures aggregate calibration, not discrimination**; the confusion-rate "drop" was just "predict certain ⇒ get all true-certain right." All metrics below in §4 are retained for the record but are **invalidated by the collapse above** — read them through this lens.

**If revisited:** the lever is *different signal, not a different model* — the judge's own token-level logprobs / an explicit numeric confidence from the judge, calibrated directly. And always report **rank-correlation + accuracy-above-the-always-certain baseline**, never bare ECE/accuracy.

---

## 1. Data

| Item | Value |
|---|---|
| Samples | 500, selected by `mafc/training/sampler.py` (stratified, boundary-weighted) over **2025 Q2–Q4** (`_with_fact_checks`) |
| Pipeline | full process with **`gemini_3_flash`** (fact_check / web_search / media / judge), traces in `out/veritas-2025_with_fact_checks-7class-500/` |
| Target | `y = abs(integrity.score)` ∈ [0, 1]; GT integrity score is continuous (e.g. −0.83), mean \|score\| = 0.904 |
| Leakage controls | retrieval date-filtered (`finding-date < claim-date`); Q2–Q4 is past Gemini's Jan-2025 cutoff (no parametric leakage) |

**GT 7-class distribution** (highly imbalanced toward `certain`):

| Class | n |
|---|---|
| Intact (certain) | 177 |
| Intact (rather certain) | 60 |
| Intact (rather uncertain) | 1 |
| Unknown | 3 |
| Compromised (rather uncertain) | 4 |
| Compromised (rather certain) | 51 |
| Compromised (certain) | 204 |

> The four middle classes hold ~8 samples total → effectively a 4-class problem; macro-F1 and the lower thresholds are correspondingly noisy.

---

## 2. Features

**Extraction:** `mafc/training/features.py` + `dataset.py`, joining each `*.fact_check_trace.json` (judge trace embedded as `judge_run`) with `claims.json` by id. `true_label` excluded (leakage). Missing values → NaN (handled natively). The feature set is **lean and blueprint-independent** — nothing tied to the blueprint set (`blueprint_name`, the selector's `cf_*`, `hit_max_iterations`) or empirically dead (hedge lexicon, redundant counts) is included.

**13 structured features** (categoricals: `judge_direction`, `language`):
- *Conditioning:* `judge_direction` (predicted side; sign comes from the judge).
- *Evidence sufficiency:* `evidence_count`, `useful_ratio`, `n_distinct_domains`.
- *Search struggle:* `n_iterations`, `evidence_growth_total`, `runtime_seconds`.
- *Judge output shape:* `justification_char_len`, `judge_output_tokens`.
- *Difficulty priors:* `has_media`, `language`, `claim_char_len`, `claim_vs_evidence_months`.

**Embedding features** — OpenAI `text-embedding-3-large` (cost ≈ $0.11/500 samples):
- *Evidence agreement:* `emb_disp_{mean,var,max}` = pairwise cosine distance over evidence-takeaway embeddings (high = conflicting sources); `emb_ev_claim_{mean,max}` = evidence-vs-claim cosine.
- *Justification semantics:* `just_emb_norm` + the **full 3072-dim** justification vector, **PCA-reduced to 32 components inside the model pipeline** (refit per fold → no leakage).

---

## 3. Model

`sklearn.ensemble.HistGradientBoostingRegressor` (target = `|integrity.score|`), wrapped in a pipeline that ordinal-encodes categoricals and PCA-reduces the justification embedding in-fold.

| Hyperparameter | Value |
|---|---|
| loss | squared_error |
| learning_rate | 0.06 |
| max_iter | 300 |
| min_samples_leaf | 10 |
| early_stopping | True |
| max_leaf_nodes / max_depth / l2 | 31 / None / 0.0 (defaults) |
| random_state | 0 |
| justification-embedding PCA (in-fold) | 32 components |

**Output / deliverable:** the **continuous** value `sign(judge_direction) × m` ∈ [-1, 1], where `m` is the predicted magnitude. `m` is already on the VeriTaS scale (trained on `|integrity.score|`), so no rescaling. A 7-class label can be produced for diagnostics by binning that continuous value with the **canonical VeriTaS rule** (`mafc.eval.veritas` `THRESHOLDS_7`, via `label_from_signed_score`) — the *same* function that defines the GT labels. No custom/tuned thresholds.

---

## 4. Results (5-fold out-of-fold CV)

> ⚠️ **These metrics are invalidated** by the constant-prediction collapse (see top of report).
> They are retained only for the record. The model ties/loses to an "always-certain"
> baseline and has ~zero rank-correlation with the truth.

All metrics in one table. The **deliverable** is the continuous `sign × magnitude`; the
7-bin/3-bin/confusion columns bin *both* baseline and model predictions by the **same
canonical VeriTaS rule** (cuts at ±1/6, ±1/2, ±5/6) — the rule that defines the GT labels —
so the comparison is apples-to-apples, with no tuned thresholds anywhere.

| | MSE ↓ | MAE ↓ | ECE | 7-bin Acc | 3-bin Acc | certain↔rather |
|---|---|---|---|---|---|---|
| Baseline (judge's discrete label) | 0.2538 | 0.2919 | 0.1782 | 44.02% | 89.86% | 51.05% |
| **Our model** | **0.2443** | **0.2039** | **0.0072** | **69.57%** | 89.86% | **23.00%** |

*Per-side confusion: intact 56.0→27.1, compromised 46.6→19.3.* For "Our model": MSE/MAE are
the continuous output; ECE is on the magnitude; 7-bin/3-bin/confusion bin that output by the
VeriTaS rule.

- **Calibration transformed:** ECE 0.178→0.007 (~25×), 7-bin accuracy +25.5 pts, the target
  `certain↔rather` confusion roughly halved (51→23%). **3-bin accuracy unchanged** — direction
  is the judge's, untouched. The judge systematically *under-called* confidence; the model fixes it.
- **MSE gain modest (~4%), MAE gain large (~30%).** MSE is dominated by the **9.7% (48/493) of
  rows where the judge's direction is wrong** — squared errors ≈ (2·score)², which a
  magnitude-only model cannot fix. The continuous output (0.2443) still beats both the baseline
  (0.2538) and binning the prediction (0.2729) — i.e. the dataset's discretization is suboptimal
  as a *prediction target*, so we deliver the continuous value.
- **Honest estimate:** with no tuned thresholds, plain 5-fold OOF is already unbiased (the
  regressor is the only fitted object, fit per fold). Fold-to-fold accuracy spans ~54–70%
  (~±6% at n=500) — solid, but don't over-read the point estimate.

### Learning curve — data is saturated
MAE vs. training size plateaus by **n ≈ 220**; more samples won't improve the model (only tighten the estimate band).

| n_train | 74 | 148 | 222 | 296 | 370 |
|---|---|---|---|---|---|
| MAE | 0.090 | 0.078 | 0.076 | 0.070 | 0.076 |

### Feature importance (grouped permutation, magnitude MAE, 5×5)

| Family | MAE increase when permuted |
|---|---|
| **justification embedding** | **+0.0024** |
| search struggle (`runtime_seconds`) | +0.0013 |
| evidence agreement (dispersion) | +0.0008 |
| difficulty priors | +0.0005 |
| direction conditioning | +0.0002 |
| evidence sufficiency / judge output shape | ≈ 0.0000 |

The judge's **justification text is the workhorse**; behavioural struggle and evidence-agreement follow. (Importances are small and noisy — std ~ mean — so ranks past the top two are soft.)

---

## Open question — signed regressor (for MSE)
The deliverable metric (MSE) is dominated by direction errors, which the magnitude×judge-sign design cannot touch. A **signed regressor** (predict the signed score directly, free to hedge toward 0 when the sign is uncertain) could soften those wrong-sign penalties and may beat 0.2443 — at the cost of the clean "judge owns the sign" separation that makes the certainty story interpretable. **Untested; highest-leverage next experiment if MSE is the headline.**

## Caveats
- Single split (n=500); fold-to-fold spread ~±6%.
- Four middle classes are near-empty (~8 rows total) → 4-class in practice; macro-F1 not meaningful.
- MSE ceiling is set by the 9.7% wrong-direction rows (a magnitude-only model can't fix the sign).

## Artifacts & reproduce
- Sample selection: `out/training/sample_q2q4.*` (sampler manifest + `sample_ids.yaml`).
- Features: `out/training/features.csv` (+ `_meta.csv`). Canonical run: `out/training/run/` (`magnitude_regressor.joblib`, `metrics_{regressor,baseline}.json`, `cv.json`, `learning_curve.json`, `feature_importance.json`, `report.txt`).
- Code: `mafc/training/` + `scripts/training/{build_features,train_eval,feature_importance}.py`. Binning uses `mafc.eval.veritas` `THRESHOLDS_7` (no tuned thresholds). Tests: `tests/training/` (25 pass).

```bash
set -a; source config/.env; set +a   # OpenAI key for embeddings
R=out/veritas-2025_with_fact_checks-7class-500
python scripts/training/build_features.py --traces $R/traces \
  --claims data/veritas_2025_with_fact_checks --embeddings --out out/training/features
python scripts/training/train_eval.py --features out/training/features.csv \
  --meta out/training/features_meta.csv --out-dir out/training/run --cv kfold --folds 5
python scripts/training/feature_importance.py --features out/training/features.csv \
  --out out/training/run/feature_importance.json
```
