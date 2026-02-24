# stress/runner.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd

from stress.schemas import detect_task_type

from stress.tests.noise import run_noise_stress_test, NoiseStressConfig
from stress.tests.feature_drop import run_feature_drop_test
from stress.tests.missingness import run_missingness_shock_test, MissingnessShockConfig


def _safe_block(fn, name: str) -> Dict[str, Any]:
    t0 = time.time()
    try:
        out = fn()
        return {"status": "ok", "duration_s": round(time.time() - t0, 4), "output": out}
    except Exception as e:
        return {"status": "error", "duration_s": round(time.time() - t0, 4), "error": f"{name} failed: {e}"}


def run_all_stress_tests(
    *,
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    split: Optional[dict] = None,
    noise_config: NoiseStressConfig = NoiseStressConfig(),
    missingness_config: MissingnessShockConfig = MissingnessShockConfig(),
) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    task = detect_task_type(df[target_col])

    results: Dict[str, Any] = {}

    # -------------------------
    # Phase 2 — Noise
    # -------------------------
    def _noise():
        return run_noise_stress_test(
            model=model,
            df=df,
            target_col=target_col,
            config=noise_config,
        )

    results["noise"] = _safe_block(_noise, "noise")

    # -------------------------
    # Phase 4 — Missingness Shock
    # -------------------------
    def _missingness():
        return run_missingness_shock_test(
            model=model,
            df=df,
            target_col=target_col,
            config=missingness_config,
        )

    results["missingness"] = _safe_block(_missingness, "missingness")

    # -------------------------
    # Phase 3 — Feature Drop (leakage-safe if split exists)
    # -------------------------
    def _feature_drop():
        if isinstance(split, dict) and all(k in split for k in ("X_train", "y_train", "X_test", "y_test")):
            return run_feature_drop_test(
                model=model,
                X_train=split["X_train"],
                y_train=split["y_train"],
                X_test=split["X_test"],
                y_test=split["y_test"],
                top_k=5,
                seed=42,
                n_repeats=5,
            )

        # fallback: use df
        X = df.drop(columns=[target_col])
        y = df[target_col]
        if X.shape[1] == 0:
            return {"status": "skip", "reason": "No feature columns available (only target present)."}

        return run_feature_drop_test(
            model=model,
            X=X,
            y=y,
            top_k=min(5, X.shape[1]),
            seed=42,
            n_repeats=5,
        )

    results["feature_drop"] = _safe_block(_feature_drop, "feature_drop")

    return {"status": "ok", "task": task, "results": results}