# stress/tests/covariate_shift.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import inspect
import numpy as np
import pandas as pd

from ml_stress_test.schemas import detect_task_type
from ml_stress_test.metrics import (
    ClassificationConfig,
    RegressionConfig,
    classification_metrics,
    regression_metrics,
)


@dataclass(frozen=True)
class CovariateShiftConfig:
    """
    Phase 5 â€” Covariate Shift Simulation
    """
    levels: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)

    # Numeric drift
    scale_factors: Tuple[float, ...] = (1.0, 1.05, 1.10, 1.20)
    mean_shift_fracs: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.40)
    clip_quantiles: Optional[Tuple[float, float]] = (0.005, 0.995)

    # Categorical drift
    dropout_probs: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    substitute_probs: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.06)

    seed: int = 42
    missing_token: str = "__MISSING_CAT__"


# -----------------------------
# Helpers
# -----------------------------
def _split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe.")
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y


def _get_col_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, cat_cols


def _pick(arr: Tuple[float, ...], i: int) -> float:
    return float(arr[i]) if i < len(arr) else float(arr[-1])


def _apply_numeric_shift(
    X: pd.DataFrame,
    numeric_cols: List[str],
    *,
    scale_factor: float,
    mean_shift_frac: float,
    clip_quantiles: Optional[Tuple[float, float]],
) -> pd.DataFrame:
    Xs = X.copy()

    for col in numeric_cols:
        s = Xs[col].astype(float)

        # 1) scaling perturbation
        if scale_factor != 1.0:
            s = s * scale_factor

        # 2) mean shifting (add frac * std)
        if mean_shift_frac != 0.0:
            std = float(np.nanstd(s))
            if std > 0:
                s = s + (mean_shift_frac * std)

        # 3) range modification (clamp)
        if clip_quantiles is not None:
            ql, qh = clip_quantiles
            lo = float(np.nanquantile(s, ql))
            hi = float(np.nanquantile(s, qh))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                s = s.clip(lower=lo, upper=hi)

        Xs[col] = s

    return Xs


def _apply_categorical_shift(
    X: pd.DataFrame,
    cat_cols: List[str],
    rng: np.random.Generator,
    *,
    dropout_prob: float,
    substitute_prob: float,
    missing_token: str,
) -> pd.DataFrame:
    Xs = X.copy()

    for col in cat_cols:
        s = Xs[col].astype("object")
        n = len(s)
        if n == 0:
            continue

        # 4) category dropout -> missing token
        if dropout_prob > 0:
            mask = rng.random(n) < dropout_prob
            if mask.any():
                s.loc[mask] = missing_token

        # 5) category substitution
        if substitute_prob > 0:
            vals = pd.Series(s.unique()).dropna().astype(str)
            vals = vals[vals != missing_token].tolist()

            if len(vals) >= 2:
                mask = rng.random(n) < substitute_prob
                if mask.any():
                    current = s.loc[mask].astype(str).values
                    replacements = []
                    for v in current:
                        choices = [x for x in vals if x != v]
                        replacements.append(rng.choice(choices))
                    s.loc[mask] = replacements

        Xs[col] = s

    return Xs


def _primary_metric_info(task_type: str) -> Tuple[str, bool]:
    # Matches your report.html expectation
    if task_type == "regression":
        return "mae", False
    return "accuracy", True


def _degradation_pct(baseline: float, shifted: float, *, higher_is_better: bool) -> float:
    """
    Positive % means worse.
    """
    if baseline is None or shifted is None:
        return float("nan")

    try:
        baseline = float(baseline)
        shifted = float(shifted)
    except Exception:
        return float("nan")

    if not np.isfinite(baseline) or not np.isfinite(shifted) or baseline == 0:
        return float("nan")

    if higher_is_better:
        return max(0.0, (baseline - shifted) / abs(baseline)) * 100.0
    return max(0.0, (shifted - baseline) / abs(baseline)) * 100.0


def _shift_sensitivity_index(levels: List[float], degradations: List[float]) -> float:
    """
    AUC-like degradation summary normalized by max level.
    """
    if len(levels) < 2:
        return float("nan")

    L = np.array(levels, dtype=float)
    D = np.array(degradations, dtype=float)

    mask = np.isfinite(L) & np.isfinite(D)
    if mask.sum() < 2:
        return float("nan")

    L = L[mask]
    D = D[mask]

    idx = np.argsort(L)
    L, D = L[idx], D[idx]

    maxL = float(np.max(L))
    if maxL <= 0:
        return float("nan")

    area = float(np.trapz(D, L))
    return area / maxL


def _safe_call_metrics(fn, *, y_true, y_pred, config, y_proba=None) -> Dict[str, Any]:
    """
    Calls a metrics function safely even if its signature differs (y vs y_true, etc.).
    ALSO materializes callable metric values into numeric results.
    """
    candidates = {
        # common true/pred names
        "y_true": y_true,
        "y": y_true,
        "target": y_true,
        "labels": y_true,
        "actual": y_true,

        "y_pred": y_pred,
        "pred": y_pred,
        "preds": y_pred,
        "prediction": y_pred,

        "config": config,
        "cfg": config,

        # optional probabilities for classification
        "y_proba": y_proba,
        "proba": y_proba,
        "y_score": y_proba,
        "scores": y_proba,
    }

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())

    call_kwargs = {k: v for k, v in candidates.items() if (k in accepted and v is not None)}

    # 1) Get raw output (may be dict of numbers OR dict of callables)
    try:
        raw = fn(**call_kwargs)
    except TypeError:
        # fallback positional pattern
        raw = fn(y_true, y_pred, config)

    raw_dict = dict(raw) if not isinstance(raw, dict) else raw

    # 2) Materialize callable values (your screenshot shows this is needed)
    out: Dict[str, Any] = {}
    for k, v in raw_dict.items():
        if callable(v):
            # try calling with whatever signature it supports
            try:
                vsig = inspect.signature(v)
                vparams = set(vsig.parameters.keys())
                v_kwargs = {}
                # support both y/y_true, y_pred/pred, etc.
                if "y_true" in vparams:
                    v_kwargs["y_true"] = y_true
                if "y" in vparams:
                    v_kwargs["y"] = y_true
                if "y_pred" in vparams:
                    v_kwargs["y_pred"] = y_pred
                if "pred" in vparams:
                    v_kwargs["pred"] = y_pred
                if "y_proba" in vparams and y_proba is not None:
                    v_kwargs["y_proba"] = y_proba
                if "proba" in vparams and y_proba is not None:
                    v_kwargs["proba"] = y_proba

                # if it accepts kwargs, use them; else try positional
                if v_kwargs:
                    out[k] = v(**v_kwargs)
                else:
                    out[k] = v(y_true, y_pred)
            except TypeError:
                # last resort patterns
                try:
                    out[k] = v(y_true, y_pred)
                except Exception:
                    try:
                        out[k] = v()
                    except Exception:
                        out[k] = np.nan
            except Exception:
                out[k] = np.nan
        else:
            out[k] = v

    return out

def _compute_metrics_direct(model: Any, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
    """
    âœ… NO compute_metrics()
    Uses regression_metrics / classification_metrics directly (signature-safe).
    """
    y_pred = model.predict(X)

    if task_type == "regression":
        cfg = RegressionConfig()
        return _safe_call_metrics(regression_metrics, y_true=y, y_pred=y_pred, config=cfg)

    # classification
    cfg = ClassificationConfig()

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            y_proba = None

    # Try with probabilities; if your metrics fn doesn't accept them, wrapper will drop them
    return _safe_call_metrics(classification_metrics, y_true=y, y_pred=y_pred, y_proba=y_proba, config=cfg)


# -----------------------------
# Main entry point
# -----------------------------
def run_covariate_shift_test(
    *,
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    config: CovariateShiftConfig = CovariateShiftConfig(),
) -> Dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) < 5:
        raise ValueError("df is empty or too small for covariate shift test.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in df.")

    X, y = _split_X_y(df, target_col)
    numeric_cols, cat_cols = _get_col_types(X)

    task_type = detect_task_type(y)
    primary_metric, higher_is_better = _primary_metric_info(task_type)

    rng = np.random.default_rng(config.seed)

    # --- baseline metrics ---
    base_metrics = _compute_metrics_direct(model, X, y, task_type)
    baseline_primary = base_metrics.get(primary_metric, np.nan)

    runs: List[Dict[str, Any]] = []
    curve_records: List[Dict[str, Any]] = []
    degradations: List[float] = []
    level_list: List[float] = []

    for i, lvl in enumerate(config.levels):
        scale_factor = _pick(config.scale_factors, i)
        mean_shift_frac = _pick(config.mean_shift_fracs, i)
        dropout_prob = _pick(config.dropout_probs, i)
        substitute_prob = _pick(config.substitute_probs, i)

        Xs = X
        if numeric_cols:
            Xs = _apply_numeric_shift(
                Xs,
                numeric_cols,
                scale_factor=scale_factor,
                mean_shift_frac=mean_shift_frac,
                clip_quantiles=config.clip_quantiles,
            )
        if cat_cols:
            Xs = _apply_categorical_shift(
                Xs,
                cat_cols,
                rng,
                dropout_prob=dropout_prob,
                substitute_prob=substitute_prob,
                missing_token=config.missing_token,
            )

        m = _compute_metrics_direct(model, Xs, y, task_type)
        primary_val = m.get(primary_metric, np.nan)

        deg = _degradation_pct(baseline_primary, primary_val, higher_is_better=higher_is_better)

        runs.append({
            "level": float(lvl),
            "scale_factor": float(scale_factor),
            "mean_shift_frac": float(mean_shift_frac),
            "clip_quantiles": config.clip_quantiles,
            "cat_dropout_prob": float(dropout_prob),
            "cat_substitute_prob": float(substitute_prob),
            "primary_metric": primary_metric,
            "primary_value": primary_val,
            "degradation_pct": float(deg),
            "metrics": m,
        })

        # Flat row for report.html
        rec = {"shift_level": float(lvl), "degradation_pct": float(deg)}
        for k, v in (m or {}).items():
            rec[f"metrics.{k}"] = v
        curve_records.append(rec)

        level_list.append(float(lvl))
        degradations.append(float(deg))

    ssi = _shift_sensitivity_index(level_list, degradations)

    if np.isfinite(ssi):
        if ssi <= 5.0:
            stability = "stable"
        elif ssi <= 15.0:
            stability = "moderate"
        else:
            stability = "fragile"
    else:
        stability = "unknown"

    return {
        "name": "covariate_shift",
        "task_type": task_type,
        "columns": {"numeric": numeric_cols, "categorical": cat_cols},
        "baseline": {
            "primary_metric": primary_metric,
            "primary_value": baseline_primary,
            "metrics": base_metrics,
        },
        "runs": runs,
        "curve_records": curve_records,
        "summary": {
            "shift_sensitivity_index": float(ssi),
            "max_degradation_pct": float(np.nanmax(degradations)) if degradations else float("nan"),
            "stability": stability,
        },
        "config": {
            "levels": list(config.levels),
            "scale_factors": list(config.scale_factors),
            "mean_shift_fracs": list(config.mean_shift_fracs),
            "clip_quantiles": config.clip_quantiles,
            "dropout_probs": list(config.dropout_probs),
            "substitute_probs": list(config.substitute_probs),
            "seed": config.seed,
            "missing_token": config.missing_token,
        },
    }
