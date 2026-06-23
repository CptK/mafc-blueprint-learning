"""Gradient-boosted magnitude regressor (target = ``|integrity.score|`` in [0, 1]).

Uses scikit-learn's ``HistGradientBoostingRegressor`` (already a dependency; no new
gradient-boosting lib needed). The judge's predicted direction is a categorical
input feature. Predictions are clipped to [0, 1].

Supports a single train/predict, k-fold and temporal CV, and a learning curve
(error vs. training-set size). The fitted model + the column schema are saved with
``joblib`` (ships with scikit-learn).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from mafc.training.features import ID_COL, TARGET_COL

# Columns that are identifiers or targets — never model inputs.
NON_FEATURE_COLS = {ID_COL, TARGET_COL}

# Raw justification-embedding dimensions (``just_emb_0``, ``just_emb_1``, …). These
# are PCA-reduced *inside* the pipeline; ``just_emb_norm`` is kept as a plain numeric
# feature and is intentionally not matched here.
_JUST_EMB_RE = re.compile(r"^just_emb_\d+$")


def justification_embedding_cols(cols) -> list[str]:
    return [c for c in cols if _JUST_EMB_RE.match(c)]


def _fill_all_nan_columns(X):
    """Replace columns that are entirely NaN with 0.

    ``HistGradientBoostingRegressor`` handles NaN natively for *partially*-missing
    columns, but its binner raises on a fully-NaN column (common in small CV folds
    where a sparse feature like ``claim_vs_evidence_months`` is absent). Partially
    present columns keep their NaNs so the model can still split on missingness.
    """
    X = np.array(X, dtype=float, copy=True)
    all_nan = np.all(np.isnan(X), axis=0)
    if all_nan.any():
        X[:, all_nan] = 0.0
    return X

# Object/string columns are treated as categoricals.
DEFAULT_CATEGORICAL = ("judge_direction", "language", "blueprint_name")


@dataclass
class TrainConfig:
    learning_rate: float = 0.06
    max_iter: int = 300
    max_depth: int | None = None
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.0
    min_samples_leaf: int = 10
    early_stopping: bool = True
    random_state: int = 0
    # PCA components for the raw justification-embedding block (fit in-fold). 0
    # disables PCA and drops the raw dims entirely (keeps only ``just_emb_norm``).
    pca_components: int = 32


@dataclass
class FittedModel:
    pipeline: Pipeline
    feature_cols: list[str]
    categorical_cols: list[str]
    config: TrainConfig = field(default_factory=TrainConfig)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (all_feature_cols, categorical_cols) for a feature table."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    categorical = [
        c
        for c in feature_cols
        if c in DEFAULT_CATEGORICAL or df[c].dtype == object
    ]
    return feature_cols, categorical


def _build_pipeline(
    feature_cols: list[str], categorical_cols: list[str], cfg: TrainConfig
) -> Pipeline:
    pca_cols = justification_embedding_cols(feature_cols)
    use_pca = bool(pca_cols) and cfg.pca_components > 0
    # When PCA is on, the raw just_emb dims go through their own branch; when off,
    # they are dropped (remainder="drop") so they never reach the tree un-reduced.
    numeric_cols = [
        c for c in feature_cols if c not in categorical_cols and c not in pca_cols
    ]
    cat_idx = list(range(len(numeric_cols), len(numeric_cols) + len(categorical_cols)))
    num_pipe = Pipeline(
        [("nan_guard", FunctionTransformer(_fill_all_nan_columns, validate=False))]
    )
    transformers = [
        ("num", num_pipe, numeric_cols),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_cols,
        ),
    ]
    if use_pca:
        # n_components is capped at fit time by sklearn to min(n_samples, n_features);
        # cap here too so tiny CV/learning-curve folds don't raise.
        k = min(cfg.pca_components, len(pca_cols))
        pca_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("pca", PCA(n_components=k, random_state=cfg.random_state)),
            ]
        )
        transformers.append(("just_pca", pca_pipe, pca_cols))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_depth=cfg.max_depth,
        max_leaf_nodes=cfg.max_leaf_nodes,
        l2_regularization=cfg.l2_regularization,
        min_samples_leaf=cfg.min_samples_leaf,
        early_stopping=cfg.early_stopping,
        categorical_features=cat_idx if cat_idx else None,
        random_state=cfg.random_state,
    )
    return Pipeline([("pre", pre), ("gbr", model)])


# Below this row count, sklearn's internal early-stopping validation split is too
# small to bin reliably (it can raise "window shape cannot be larger than input
# array shape"); disable it and train the full schedule instead.
_MIN_ROWS_FOR_EARLY_STOPPING = 200


def fit(df: pd.DataFrame, cfg: TrainConfig | None = None) -> FittedModel:
    cfg = cfg or TrainConfig()
    if cfg.early_stopping and len(df) < _MIN_ROWS_FOR_EARLY_STOPPING:
        cfg = replace(cfg, early_stopping=False)
    feature_cols, categorical_cols = feature_columns(df)
    pipe = _build_pipeline(feature_cols, categorical_cols, cfg)
    X = df[feature_cols]
    y = df[TARGET_COL].to_numpy(dtype=float)
    pipe.fit(X, y)
    return FittedModel(pipe, feature_cols, categorical_cols, cfg)


def predict(model: FittedModel, df: pd.DataFrame) -> np.ndarray:
    """Predict magnitudes, clipped to [0, 1]."""
    X = df[model.feature_cols]
    preds = model.pipeline.predict(X)
    return np.clip(preds, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _fold_indices_kfold(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def _fold_indices_temporal(order: np.ndarray, k: int) -> list[np.ndarray]:
    """Contiguous temporal folds over a precomputed chronological ordering."""
    return [chunk for chunk in np.array_split(order, k)]


def cross_validate(
    df: pd.DataFrame,
    cfg: TrainConfig | None = None,
    k: int = 5,
    mode: str = "kfold",
    date_series: pd.Series | None = None,
    seed: int = 0,
) -> dict:
    """Run k-fold (``mode='kfold'``) or temporal (``mode='temporal'``) CV.

    Returns out-of-fold predictions + per-fold MSE/MAE. For temporal mode pass a
    ``date_series`` aligned to ``df`` rows (sortable strings/timestamps).
    """
    cfg = cfg or TrainConfig()
    n = len(df)
    df = df.reset_index(drop=True)
    if mode == "temporal":
        if date_series is None:
            raise ValueError("temporal CV requires date_series")
        order = np.argsort(date_series.reset_index(drop=True).fillna("").to_numpy(dtype=str))
        folds = _fold_indices_temporal(order, k)
    else:
        folds = _fold_indices_kfold(n, k, seed)

    oof = np.full(n, np.nan)
    fold_metrics: list[dict] = []
    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        m = fit(df.iloc[train_idx], cfg)
        preds = predict(m, df.iloc[test_idx])
        oof[test_idx] = preds
        y = df.iloc[test_idx][TARGET_COL].to_numpy(dtype=float)
        fold_metrics.append(
            {
                "fold": fi,
                "n_test": int(len(test_idx)),
                "mse": float(np.mean((y - preds) ** 2)),
                "mae": float(np.mean(np.abs(y - preds))),
            }
        )
    valid = ~np.isnan(oof)
    y_all = df[TARGET_COL].to_numpy(dtype=float)
    overall = {
        "mse": float(np.mean((y_all[valid] - oof[valid]) ** 2)) if valid.any() else None,
        "mae": float(np.mean(np.abs(y_all[valid] - oof[valid]))) if valid.any() else None,
        "n": int(valid.sum()),
    }
    return {"mode": mode, "k": k, "folds": fold_metrics, "overall": overall, "oof": oof}


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------


def learning_curve(
    df: pd.DataFrame,
    cfg: TrainConfig | None = None,
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    holdout: float = 0.25,
    seed: int = 0,
) -> list[dict]:
    """Validation error vs. training-set size on a fixed holdout.

    Splits off a holdout once, then trains on growing prefixes of the remaining
    (shuffled) data. Useful to decide whether collecting more 2025 data pays off.
    """
    cfg = cfg or TrainConfig()
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_holdout = max(1, int(len(df) * holdout))
    holdout_idx = idx[:n_holdout]
    train_pool = idx[n_holdout:]
    val = df.iloc[holdout_idx]
    y_val = val[TARGET_COL].to_numpy(dtype=float)

    out: list[dict] = []
    for frac in fractions:
        n_train = max(1, int(len(train_pool) * frac))
        sub = df.iloc[train_pool[:n_train]]
        if sub[TARGET_COL].nunique() < 2:
            continue
        m = fit(sub, cfg)
        preds = predict(m, val)
        out.append(
            {
                "fraction": frac,
                "n_train": int(n_train),
                "mse": float(np.mean((y_val - preds) ** 2)),
                "mae": float(np.mean(np.abs(y_val - preds))),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: FittedModel, path: Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    schema = {
        "feature_cols": model.feature_cols,
        "categorical_cols": model.categorical_cols,
        "config": model.config.__dict__,
    }
    path.with_suffix(".schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def load_model(path: Path) -> FittedModel:
    import joblib

    return joblib.load(Path(path))
