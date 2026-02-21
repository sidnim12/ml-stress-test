# stress/runner.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional, TypedDict

from stress.tests.noise import NoiseStressConfig, run_noise_stress_test


# -------------------------
# Output schema (stable contract)
# -------------------------
class StressTestResult(TypedDict, total=False):
    status: str                 # "ok" | "error" | "skipped"
    duration_s: float
    error: str
    output: Dict[str, Any]


class StressSuiteResult(TypedDict):
    suite: str
    target_col: str
    rows: int
    cols: int
    results: Dict[str, StressTestResult]


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class StressRunnerConfig:
    """
    Controls suite behavior in production.
    - enabled_tests: choose which tests to run (default: noise only, Phase 2)
    - fail_fast: stop on first failure (useful in CI)
    """
    enabled_tests: tuple[str, ...] = ("noise",)
    fail_fast: bool = False


# -------------------------
# Internal registry of tests
# -------------------------
StressFn = Callable[[Any, Any, str], Dict[str, Any]]


def _validate_inputs(model: Any, df: Any, target_col: str) -> None:
    if model is None:
        raise ValueError("model must not be None")
    if df is None:
        raise ValueError("df must not be None")
    if not isinstance(target_col, str) or not target_col.strip():
        raise ValueError("target_col must be a non-empty string")

    # Lightweight, pandas-friendly checks without importing pandas here
    if not hasattr(df, "columns") or not hasattr(df, "shape"):
        raise TypeError("df must be a tabular object with .columns and .shape (e.g., pandas.DataFrame)")

    if target_col not in getattr(df, "columns"):
        raise ValueError(f"target_col='{target_col}' not found in df.columns")

    rows, cols = df.shape
    if rows == 0 or cols == 0:
        raise ValueError(f"df must be non-empty, got shape={df.shape}")


def run_all_stress_tests(
    model: Any,
    df: Any,
    target_col: str,
    *,
    noise_config: NoiseStressConfig = NoiseStressConfig(),
    runner_config: StressRunnerConfig = StressRunnerConfig(),
    logger: Optional[Any] = None,  # standard logging.Logger compatible
) -> StressSuiteResult:
    """
    Central orchestrator for all stress tests.

    Phase 2 default:
      - noise stress

    Designed for future phases:
      - missingness shock
      - feature drop
      - covariate shift
      - class imbalance shift

    Returns a stable, JSON-serializable dict with per-test status + timings.
    """
    _validate_inputs(model=model, df=df, target_col=target_col)

    # registry is defined *after* validation so configs are definitely valid
    registry: Mapping[str, StressFn] = {
        "noise": lambda m, d, t: run_noise_stress_test(
            model=m,
            df=d,
            target_col=t,
            config=noise_config,
        ),
        # Future:
        # "missingness": ...
        # "feature_drop": ...
        # "covariate_shift": ...
    }

    suite_result: StressSuiteResult = {
        "suite": "stress_suite",
        "target_col": target_col,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "results": {},
    }

    enabled = set(runner_config.enabled_tests)

    for test_name, fn in registry.items():
        if test_name not in enabled:
            suite_result["results"][test_name] = {
                "status": "skipped",
                "duration_s": 0.0,
            }
            continue

        start = perf_counter()
        try:
            if logger:
                logger.info("Running stress test: %s", test_name)

            out = fn(model, df, target_col)

            suite_result["results"][test_name] = {
                "status": "ok",
                "duration_s": perf_counter() - start,
                "output": out,
            }

        except Exception as e:
            duration = perf_counter() - start
            if logger:
                logger.exception("Stress test failed: %s", test_name)

            suite_result["results"][test_name] = {
                "status": "error",
                "duration_s": duration,
                "error": f"{type(e).__name__}: {e}",
            }

            if runner_config.fail_fast:
                break

    return suite_result