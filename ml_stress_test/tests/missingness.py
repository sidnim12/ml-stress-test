# stress/tests/missingness.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_stress_test.schemas import detect_task_type

# âœ… Your compute_metrics expects: (metrics_dict, *, y_true, y_pred, y_proba)
from ml_stress_test.metrics import compute_metrics, classification_metrics, regression_metrics


@dataclass(frozen=True)
class MissingnessShockConfig:
    """
    Missingness Shock Test:
    - Randomly masks feature values at increasing levels
    - Uses model.predict on the corrupted dataset
    - Assumes the model is a pipeline that can handle missing values via its imputers
    """
    levels: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)
    seed: int = 42

    # Collapse threshold definition:
    # For regression: when RMSE increases by >= collapse_rmse_increase_pct (e.g., 0.80 means +80%)
    # For classification: when primary metric (accuracy by default) drops by >= collapse_drop_pct (e.g., 0.20 means -20 points)
    collapse_rmse_increase_pct: float = 0.80
    collapse_drop_pct: float = 0.20

    # Robustness score (0â€“100):
    # 100 = very stable, 0 = catastrophic degradation
    # We compute area under normalized performance curve (details below)
    score_floor: float = 0.0
    score_ceiling: float = 100.0


def _mask_features_randomly(
    X: pd.DataFrame,
    *,
    level: float,
    seed: int,
) -> pd.DataFrame:
    """
    Randomly set a fraction of feature cells to NaN.
    - Only masks features (not target)
    - Works for numeric + categorical
    """
    if level <= 0:
        return X.copy()

    rng = np.random.default_rng(seed)
    X_masked = X.copy()

    n_rows, n_cols = X_masked.shape
    if n_rows == 0 or n_cols == 0:
        return X_masked

    total_cells = n_rows * n_cols
    n_mask = int(round(level * total_cells))
    n_mask = max(0, min(n_mask, total_cells))

    # Choose random flat indices
    flat_indices = rng.choice(total_cells, size=n_mask, replace=False)
    row_idx = flat_indices // n_cols
    col_idx = flat_indices % n_cols

    for r, c in zip(row_idx, col_idx):
        X_masked.iat[int(r), int(c)] = np.nan

    return X_masked


def _primary_metric_name(task_type: str, metrics: Dict[str, float]) -> str:
    """
    Pick a primary metric for collapse & score normalization.
    - Regression: RMSE (lower is better)
    - Classification: accuracy if present else first metric
    """
    if task_type == "regression":
        return "rmse"
    # classification
    if isinstance(metrics, dict) and "accuracy" in metrics:
        return "accuracy"
    return next(iter(metrics.keys())) if metrics else "accuracy"


def _get_metric_fns(task_type: str) -> Dict[str, Any]:
    """
    Your compute_metrics() needs a dict[str, MetricFn].

    In your repo, classification_metrics / regression_metrics likely return that dict,
    sometimes directly, sometimes via config objects. We support common patterns.
    """
    if task_type == "classification":
        # Pattern A: classification_metrics() -> dict
        try:
            out = classification_metrics()
            if isinstance(out, dict):
                return out
        except TypeError:
            pass

        # Pattern B: classification_metrics(config) -> dict
        try:
            from ml_stress_test.metrics import ClassificationConfig  # optional
            out = classification_metrics(ClassificationConfig())
            if isinstance(out, dict):
                return out
        except Exception:
            pass

        raise TypeError("Could not build classification metric functions dict from ml_stress_test.metrics.")

    # regression
    try:
        out = regression_metrics()
        if isinstance(out, dict):
            return out
    except TypeError:
        pass

    try:
        from ml_stress_test.metrics import RegressionConfig  # optional
        out = regression_metrics(RegressionConfig())
        if isinstance(out, dict):
            return out
    except Exception:
        pass

    raise TypeError("Could not build regression metric functions dict from ml_stress_test.metrics.")


def _compute_collapse_threshold(
    task_type: str,
    baseline_metrics: Dict[str, float],
    curve: List[Dict[str, Any]],
    *,
    collapse_rmse_increase_pct: float,
    collapse_drop_pct: float,
) -> Optional[float]:
    """
    Returns the first missingness level at which performance is considered "collapsed".
    If never collapses, returns None.
    """
    base_primary = _primary_metric_name(task_type, baseline_metrics)
    base_val = baseline_metrics.get(base_primary)

    if base_val is None:
        return None

    for point in curve:
        lvl = point["missingness_level"]
        m = point["metrics"]
        val = m.get(base_primary)
        if val is None:
            continue

        if task_type == "regression":
            # RMSE higher is worse
            # collapse when rmse >= base * (1 + pct)
            if float(base_val) > 0 and float(val) >= float(base_val) * (1.0 + collapse_rmse_increase_pct):
                return float(lvl)
        else:
            # accuracy higher is better
            # collapse when accuracy <= base - drop_pct
            if float(val) <= float(base_val) - collapse_drop_pct:
                return float(lvl)

    return None


def _missingness_robustness_score(
    task_type: str,
    baseline_metrics: Dict[str, float],
    curve: List[Dict[str, Any]],
    *,
    score_floor: float,
    score_ceiling: float,
) -> float:
    """
    Score 0â€“100 based on area under a normalized curve:
    - Regression: normalize RMSE relative to baseline (1.0 = baseline; larger = worse)
      Convert to "stability" = 1 / normalized_rmse, clipped [0, 1]
    - Classification: normalize primary metric relative to baseline (1.0 = baseline)
      stability = normalized_metric, clipped [0, 1]
    Then average across levels (including baseline as 1.0), convert to 0â€“100.
    """
    base_primary = _primary_metric_name(task_type, baseline_metrics)
    base_val = baseline_metrics.get(base_primary)

    if base_val is None or float(base_val) == 0:
        return float(score_floor)

    stabilities: List[float] = [1.0]  # baseline stability

    for point in curve:
        m = point["metrics"]
        val = m.get(base_primary)
        if val is None:
            continue

        if task_type == "regression":
            norm = float(val) / float(base_val)  # >= 1 is worse
            stability = 1.0 / max(norm, 1e-9)
            stability = float(np.clip(stability, 0.0, 1.0))
        else:
            norm = float(val) / float(base_val)  # <=1 is worse
            stability = float(np.clip(norm, 0.0, 1.0))

        stabilities.append(stability)

    avg_stability = float(np.mean(stabilities)) if stabilities else 0.0
    score = avg_stability * 100.0
    return float(np.clip(score, score_floor, score_ceiling))


def run_missingness_shock_test(
    *,
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    config: MissingnessShockConfig = MissingnessShockConfig(),
) -> Dict[str, Any]:
    """
    Executes Missingness Shock Test and returns a structured result.

    Expected model:
      - sklearn-like estimator or Pipeline with .predict
      - If Pipeline, imputation should happen inside preprocessor
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in df.columns")

    task_type = detect_task_type(df[target_col])

    # Split X / y (this test is NOT responsible for data splitting; it evaluates on provided df)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # âœ… Metric-function dict for your compute_metrics() signature
    metric_fns = _get_metric_fns(task_type)

    # Baseline (no missingness injected)
    y_pred_base = model.predict(X)
    baseline_metrics = compute_metrics(metric_fns, y_true=y, y_pred=y_pred_base)

    curve: List[Dict[str, Any]] = []
    for i, lvl in enumerate(config.levels):
        X_masked = _mask_features_randomly(X, level=float(lvl), seed=int(config.seed) + i)

        # Predict using same model; pipeline should handle NaNs via imputers
        y_pred = model.predict(X_masked)

        metrics = compute_metrics(metric_fns, y_true=y, y_pred=y_pred)

        curve.append(
            {
                "missingness_level": float(lvl),
                "masked_cells_est": int(round(float(lvl) * X.shape[0] * X.shape[1])) if X.shape[1] else 0,
                "metrics": metrics,
            }
        )

    collapse_at = _compute_collapse_threshold(
        task_type=task_type,
        baseline_metrics=baseline_metrics,
        curve=curve,
        collapse_rmse_increase_pct=config.collapse_rmse_increase_pct,
        collapse_drop_pct=config.collapse_drop_pct,
    )

    score = _missingness_robustness_score(
        task_type=task_type,
        baseline_metrics=baseline_metrics,
        curve=curve,
        score_floor=config.score_floor,
        score_ceiling=config.score_ceiling,
    )

    return {
        "name": "missingness_shock",
        "task_type": task_type,
        "baseline_metrics": baseline_metrics,
        "curve": curve,
        "collapse_threshold": collapse_at,   # None if never collapsed
        "robustness_score": score,          # 0â€“100
        "config": {
            "levels": list(config.levels),
            "seed": int(config.seed),
            "collapse_rmse_increase_pct": float(config.collapse_rmse_increase_pct),
            "collapse_drop_pct": float(config.collapse_drop_pct),
        },
    }
