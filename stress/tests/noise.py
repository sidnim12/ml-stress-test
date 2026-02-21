from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from stress.metrics import (
    ClassificationConfig,
    RegressionConfig,
    classification_metrics,
    regression_metrics,
    compute_metrics,
)
from stress.schemas import detect_task_type


# -------------------------
# Configuration
# -------------------------
@dataclass(frozen=True)
class NoiseStressConfig:
    noise_levels: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.50)
    seed: int = 42

    # Which numeric columns to perturb: "all_numeric" or pass explicit list in function
    numeric_mode: str = "all_numeric"

    # How to scale noise per feature:
    # - "std": scale = std(feature)
    # - "iqr": scale = IQR(feature)/1.349 (robust std-ish)
    # - "absolute": scale = 1.0 (noise_level is absolute units)
    scale_mode: str = "std"

    # If True, skip columns with 0 scale (constant columns)
    skip_constant_cols: bool = True

    # Clip feature values after noise to reduce extreme outliers (optional)
    clip_quantiles: Optional[Tuple[float, float]] = None  # e.g., (0.001, 0.999)

    # If True, compute baseline metrics (no noise) and deltas
    include_baseline: bool = True


# -------------------------
# Helpers
# -------------------------
def _validate_inputs(df: pd.DataFrame, target_col: str, noise_levels: Sequence[float]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' not found in df columns.")
    if df[target_col].isna().any():
        raise ValueError("Target column contains NaNs. Please clean/impute target first.")

    if not noise_levels:
        raise ValueError("noise_levels must be a non-empty sequence.")
    if any(n < 0 for n in noise_levels):
        raise ValueError("All noise_levels must be >= 0.")


def _get_numeric_columns(X: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None) -> List[str]:
    if numeric_cols is not None:
        missing = [c for c in numeric_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Provided numeric_cols not in X: {missing}")
        # Keep only truly numeric among the provided columns (guardrail)
        cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(X[c])]
        return cols

    # Broad numeric selection (int, float, nullable, etc.)
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]


def _robust_scale_iqr(series: pd.Series) -> float:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    # Convert IQR to a std-like scale for normal-ish data
    # (IQR ≈ 1.349 * std for normal)
    return float(iqr / 1.349) if iqr and iqr > 0 else 0.0


def _feature_scale(series: pd.Series, mode: str) -> float:
    if mode == "std":
        val = float(series.std(ddof=0))  # stable, population std
        return 0.0 if np.isnan(val) else val
    if mode == "iqr":
        return _robust_scale_iqr(series)
    if mode == "absolute":
        return 1.0
    raise ValueError(f"Unknown scale_mode='{mode}'. Use 'std', 'iqr', or 'absolute'.")


def add_gaussian_noise(
    X: pd.DataFrame,
    noise_level: float,
    *,
    seed: int = 42,
    numeric_cols: Optional[Sequence[str]] = None,
    scale_mode: str = "std",
    skip_constant_cols: bool = True,
    clip_quantiles: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Add Gaussian noise to numeric columns only.

    noise_level meaning depends on scale_mode:
      - std/iqr: per-feature sigma = noise_level * scale(feature)
      - absolute: sigma = noise_level (same units as feature)

    clip_quantiles optionally clips each perturbed numeric column to original quantile bounds
    (useful to avoid unrealistic stress values).
    """
    if noise_level < 0:
        raise ValueError("noise_level must be >= 0")

    X_noisy = X.copy(deep=True)
    cols = _get_numeric_columns(X, numeric_cols)

    if not cols or noise_level == 0:
        return X_noisy

    rng = np.random.default_rng(seed)

    # Precompute clip bounds if needed
    clip_bounds: Dict[str, Tuple[float, float]] = {}
    if clip_quantiles is not None:
        lo_q, hi_q = clip_quantiles
        if not (0 <= lo_q < hi_q <= 1):
            raise ValueError("clip_quantiles must be like (low, high) with 0<=low<high<=1.")
        for c in cols:
            lo = float(X[c].quantile(lo_q))
            hi = float(X[c].quantile(hi_q))
            clip_bounds[c] = (lo, hi)

    for col in cols:
        scale = _feature_scale(X[col], scale_mode)

        if skip_constant_cols and scale <= 0:
            continue

        sigma = noise_level * scale
        if sigma <= 0:
            continue

        noise = rng.normal(loc=0.0, scale=sigma, size=len(X))
        # Ensure numeric ops work even with nullable ints by working in float space
        X_noisy[col] = X[col].astype("float64") + noise

        if clip_quantiles is not None:
            lo, hi = clip_bounds[col]
            X_noisy[col] = X_noisy[col].clip(lower=lo, upper=hi)

    return X_noisy


def _predict_with_optional_proba(model: Any, X: pd.DataFrame, task: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Predict y_pred, and y_proba if classification and available.
    """
    try:
        y_pred = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Model predict() failed: {e}") from e

    y_proba = None
    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            y_proba = None

    return y_pred, y_proba


def _compute_task_metrics(task: str, y_true: Union[pd.Series, np.ndarray], y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Uses your existing stress.metrics registry in a consistent way.
    """
    if task == "classification":
        metrics = classification_metrics(ClassificationConfig())
        return compute_metrics(metrics, y_true=y_true, y_pred=y_pred, y_proba=y_proba)
    else:
        metrics = regression_metrics(RegressionConfig())
        return compute_metrics(metrics, y_true=y_true, y_pred=y_pred)


def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested metric dicts into a single level for easy DataFrame creation.
    Keeps it simple (1-level recursion).
    """
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{key}.{kk}"] = vv
        else:
            out[key] = v
    return out


# -------------------------
# Main runner
# -------------------------
def run_noise_stress_test(
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    *,
    config: NoiseStressConfig = NoiseStressConfig(),
    numeric_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run a Gaussian-noise stress test on numeric features only.

    Returns a dict with:
      - task: 'classification' or 'regression'
      - baseline (optional): baseline metrics
      - noise_results: list of results per noise level (metrics + optionally deltas)
      - summary_df: pandas DataFrame (flattened metrics)
    """
    _validate_inputs(df, target_col, config.noise_levels)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    task = detect_task_type(y)

    # Baseline (no noise)
    baseline_metrics: Optional[Dict[str, Any]] = None
    if config.include_baseline:
        y_pred0, y_proba0 = _predict_with_optional_proba(model, X, task)
        baseline_metrics = _compute_task_metrics(task, y_true=y, y_pred=y_pred0, y_proba=y_proba0)

    results: List[Dict[str, Any]] = []
    rows_for_df: List[Dict[str, Any]] = []

    for nl in config.noise_levels:
        X_noisy = add_gaussian_noise(
            X,
            nl,
            seed=config.seed,
            numeric_cols=numeric_cols,
            scale_mode=config.scale_mode,
            skip_constant_cols=config.skip_constant_cols,
            clip_quantiles=config.clip_quantiles,
        )

        y_pred, y_proba = _predict_with_optional_proba(model, X_noisy, task)
        metric_out = _compute_task_metrics(task, y_true=y, y_pred=y_pred, y_proba=y_proba)

        entry: Dict[str, Any] = {
            "noise_level": float(nl),
            "metrics": metric_out,
        }

        # Optional deltas vs baseline (numeric keys only)
        if baseline_metrics is not None:
            flat_base = _flatten_metrics(baseline_metrics)
            flat_curr = _flatten_metrics(metric_out)

            deltas: Dict[str, Any] = {}
            for k, v in flat_curr.items():
                bv = flat_base.get(k)
                if isinstance(v, (int, float, np.number)) and isinstance(bv, (int, float, np.number)):
                    deltas[k] = float(v) - float(bv)
            entry["delta_vs_baseline"] = deltas

        results.append(entry)

        # Build DataFrame row
        flat_metrics = _flatten_metrics(metric_out, prefix="metrics")
        row = {"task": task, "noise_level": float(nl), **flat_metrics}
        if baseline_metrics is not None:
            flat_deltas = {f"delta.{k}": v for k, v in entry.get("delta_vs_baseline", {}).items()}
            row.update(flat_deltas)
        rows_for_df.append(row)

    summary_df = pd.DataFrame(rows_for_df).sort_values("noise_level").reset_index(drop=True)

    out: Dict[str, Any] = {
        "task": task,
        "noise_results": results,
        "summary_df": summary_df,
    }
    if baseline_metrics is not None:
        out["baseline"] = baseline_metrics

    return out