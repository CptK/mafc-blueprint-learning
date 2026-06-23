import numpy as np
import pandas as pd

from mafc.training.train import (
    TrainConfig,
    cross_validate,
    feature_columns,
    fit,
    learning_curve,
    predict,
)


def _synthetic_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    evidence = rng.integers(1, 20, n).astype(float)
    hedge = rng.integers(0, 5, n).astype(float)
    direction = rng.choice(["intact", "compromised", "unknown"], n)
    # target correlates with evidence (more evidence -> higher magnitude) minus hedging
    target = np.clip(0.3 + 0.03 * evidence - 0.05 * hedge + rng.normal(0, 0.05, n), 0, 1)
    # a deliberately all-NaN column to exercise the nan-guard
    sparse = np.full(n, np.nan)
    return pd.DataFrame(
        {
            "claim_id": [f"c{i}" for i in range(n)],
            "target": target,
            "evidence_count": evidence,
            "hedge_count": hedge,
            "judge_direction": direction,
            "claim_vs_evidence_months": sparse,
        }
    )


def test_feature_columns_excludes_id_and_target() -> None:
    df = _synthetic_df(10)
    feats, cats = feature_columns(df)
    assert "claim_id" not in feats and "target" not in feats
    assert "judge_direction" in cats


def test_fit_predict_clipped() -> None:
    df = _synthetic_df()
    model = fit(df, TrainConfig())
    preds = predict(model, df)
    assert preds.min() >= 0.0 and preds.max() <= 1.0
    assert len(preds) == len(df)


def test_fit_handles_all_nan_column() -> None:
    df = _synthetic_df(60)
    # Should not raise despite a fully-NaN feature.
    model = fit(df, TrainConfig())
    assert "claim_vs_evidence_months" in model.feature_cols


def test_cross_validate_kfold_runs() -> None:
    df = _synthetic_df()
    cv = cross_validate(df, TrainConfig(), k=4, mode="kfold")
    assert cv["overall"]["n"] == len(df)
    assert cv["overall"]["mae"] is not None


def test_learning_curve_increasing_train_sizes() -> None:
    df = _synthetic_df()
    curve = learning_curve(df, TrainConfig(), fractions=(0.5, 1.0))
    sizes = [p["n_train"] for p in curve]
    assert sizes == sorted(sizes)
