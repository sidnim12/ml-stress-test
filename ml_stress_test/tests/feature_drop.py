# stress/tests/feature_drop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class FeatureDropConfig:
    top_k: int = 5
    n_repeats: int = 5
    seed: int = 42
    max_rows_for_perm: int = 2000
    drop_strategy: str = "nan"  # "nan" recommended; keeps schema stable


def _is_regression_target(y: pd.Series) -> bool:
    if pd.api.types.is_float_dtype(y):
        return True
    if pd.api.types.is_integer_dtype(y):
        nunique = y.nunique(dropna=True)
        return nunique > 20
    return False


def _primary_metric_name(task: str) -> str:
    return "rmse" if task == "regression" else "accuracy"


def _compute_primary_metric(model, X: pd.DataFrame, y: pd.Series, task: str) -> float:
    y_pred = model.predict(X)

    if task == "regression":
        yt = np.asarray(y, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    yt = np.asarray(y)
    yp = np.asarray(y_pred)
    return float(np.mean(yt == yp))


def _permutation_importance_raw(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cfg: FeatureDropConfig,
) -> List[Tuple[str, float]]:
    if len(X) > cfg.max_rows_for_perm:
        Xp = X.sample(cfg.max_rows_for_perm, random_state=cfg.seed)
        yp = y.loc[Xp.index]
    else:
        Xp, yp = X, y

    primary = _primary_metric_name(task)

    def scoring(estimator, Xs, ys):
        val = _compute_primary_metric(estimator, Xs, ys, task)
        # sklearn expects higher=better; RMSE is lower=better
        return -val if primary == "rmse" else val

    r = permutation_importance(
        model,
        Xp,
        yp,
        scoring=scoring,
        n_repeats=cfg.n_repeats,
        random_state=cfg.seed,
        n_jobs=-1,
    )

    feats = Xp.columns.tolist()
    imps = [float(v) for v in r.importances_mean]
    ranked = sorted(zip(feats, imps), key=lambda kv: kv[1], reverse=True)
    return ranked


def _apply_drop_strategy(X: pd.DataFrame, feature: str, strategy: str) -> pd.DataFrame:
    """
    IMPORTANT: We DO NOT drop the column from the dataframe.
    We keep schema stable for pipeline ColumnTransformer.
    """
    X2 = X.copy()
    if feature not in X2.columns:
        # If feature isn't there, return unchanged (safe)
        return X2

    if strategy == "nan":
        X2[feature] = np.nan
    else:
        # constant fallback (rarely needed)
        X2[feature] = 0
    return X2


def run_feature_drop_test(
    *,
    model: Any,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    top_k: int = 5,
    seed: int = 42,
    n_repeats: int = 5,
    # Preferred (leakage-safe) inputs:
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Phase 3 â€” Feature Drop Sensitivity (Pipeline-safe)

    If X_train/y_train and X_test/y_test are provided, it trains on train and evaluates on test.
    Otherwise, it uses X/y for both (fallback).

    Output schema:
      {
        "status": "ok"|"skip",
        "metric_name": "rmse"|"accuracy",
        "baseline_metric": float,
        "top_k": int,
        "fragility_index": float,
        "ranked": [
          {"feature": str, "after_drop_metric": float, "impact": float}
        ],
        "importance_top": [{"feature": str, "importance": float}, ...]
      }
    """
    # Choose leakage-safe split if available
    if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
        Xtr, ytr, Xte, yte = X_train, y_train, X_test, y_test
    else:
        if X is None or y is None:
            return {"status": "skip", "reason": "No data provided to feature drop."}
        Xtr, ytr, Xte, yte = X, y, X, y

    if not isinstance(Xtr, pd.DataFrame) or not isinstance(Xte, pd.DataFrame):
        raise ValueError("X_train/X_test must be pandas DataFrames")

    if Xtr.shape[1] == 0:
        return {"status": "skip", "reason": "No feature columns found."}

    task = "regression" if _is_regression_target(pd.Series(ytr)) else "classification"
    metric_name = _primary_metric_name(task)

    cfg = FeatureDropConfig(
        top_k=int(top_k),
        n_repeats=int(n_repeats),
        seed=int(seed),
        drop_strategy="nan",
    )

    # Baseline metric: fit fresh clone on full schema train, evaluate on test
    base = clone(model)
    base.fit(Xtr, ytr)
    baseline_val = _compute_primary_metric(base, Xte, yte, task)

    # Rank features on test set (fast + direct impact to evaluation)
    ranked_importance = _permutation_importance_raw(base, Xte, yte, task, cfg)

    k = min(int(cfg.top_k), Xte.shape[1])
    top_feats = [f for f, _ in ranked_importance[:k]]

    ranked_rows: List[Dict[str, Any]] = []

    for feat in top_feats:
        # Keep schema stable: set column to NaN instead of dropping
        Xtr_masked = _apply_drop_strategy(Xtr, feat, cfg.drop_strategy)
        Xte_masked = _apply_drop_strategy(Xte, feat, cfg.drop_strategy)

        m = clone(model)
        m.fit(Xtr_masked, ytr)

        after_val = _compute_primary_metric(m, Xte_masked, yte, task)

        # Impact: positive means worse after feature "drop"
        if task == "regression":
            impact = after_val - baseline_val  # RMSE increase is bad
        else:
            impact = baseline_val - after_val  # Accuracy decrease is bad

        ranked_rows.append(
            {
                "feature": feat,
                "after_drop_metric": float(after_val),
                "impact": float(impact),
            }
        )

    ranked_rows = sorted(ranked_rows, key=lambda r: r["impact"], reverse=True)
    fragility_index = float(np.mean([r["impact"] for r in ranked_rows])) if ranked_rows else float("nan")

    return {
        "status": "ok",
        "metric_name": metric_name,
        "baseline_metric": float(baseline_val),
        "top_k": int(k),
        "fragility_index": float(fragility_index),
        "ranked": ranked_rows,
        "importance_top": [{"feature": f, "importance": float(v)} for f, v in ranked_importance[: max(10, k)]],
    }
