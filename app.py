# app.py
from flask import Flask, render_template, request
import pandas as pd
import traceback

from ml_stress_test.schemas import validate_inputs, clean_target_nans
from models.baseline import train_and_evaluate_baseline
from ml_stress_test.runner import run_all_stress_tests

app = Flask(__name__)


def _df_to_records(df: pd.DataFrame, limit: int = 300):
    """
    Convert DataFrame to list-of-dicts for Jinja rendering.
    Prevents UI freeze on very large tables.
    """
    if df is None:
        return []
    if len(df) > limit:
        df = df.head(limit)
    return df.to_dict(orient="records")


def _normalize_baseline_output(out):
    """
    Supports baseline returning:
      - (model, results)
      - (model, results, split)
    Returns:
      model, results, split_dict_or_none
    """
    if isinstance(out, tuple):
        if len(out) == 3:
            model, results, split = out
            return model, results, split
        if len(out) == 2:
            model, results = out
            return model, results, None

    raise ValueError(
        "train_and_evaluate_baseline must return (model, results) or (model, results, split)"
    )


def _get_noise_block(stress_suite: dict):
    """
    Support both stress_suite schemas:
      A) {"results": {"noise": {...}}}
      B) {"noise": {...}}
    """
    if not isinstance(stress_suite, dict):
        return None

    results_block = stress_suite.get("results")
    if isinstance(results_block, dict) and "noise" in results_block:
        return results_block.get("noise")

    if "noise" in stress_suite:
        return stress_suite.get("noise")

    return None


def _get_missingness_block(stress_suite: dict):
    """
    Support both stress_suite schemas:
      A) {"results": {"missingness": {...}}}
      B) {"missingness": {...}}
    """
    if not isinstance(stress_suite, dict):
        return None

    results_block = stress_suite.get("results")
    if isinstance(results_block, dict) and "missingness" in results_block:
        return results_block.get("missingness")

    if "missingness" in stress_suite:
        return stress_suite.get("missingness")

    return None


def _get_covshift_block(stress_suite: dict):
    """
    Phase 5 block getter:
      A) {"results": {"covariate_shift": {...}}}
      B) {"covariate_shift": {...}}
    """
    if not isinstance(stress_suite, dict):
        return None

    results_block = stress_suite.get("results")
    if isinstance(results_block, dict) and "covariate_shift" in results_block:
        return results_block.get("covariate_shift")

    if "covariate_shift" in stress_suite:
        return stress_suite.get("covariate_shift")

    return None


def _flatten_metrics(metrics: dict) -> dict:
    """
    Flatten metric dict into keys that are Jinja-friendly.
    Example: {"rmse": 1.2, "mae": 0.9} stays as-is.
    """
    if not isinstance(metrics, dict):
        return {}
    return dict(metrics)


def _clip_0_100(x):
    try:
        x = float(x)
    except Exception:
        return None
    return max(0.0, min(100.0, x))


def _primary_metric_key(task: str):
    # baseline keys in your report.html: regression uses rmse, classification uses accuracy
    return "accuracy" if task == "classification" else "rmse"


def _score_from_ratio(task: str, ratio: float) -> float:
    """
    Turn performance ratio into a 0â€“100 score.
    - Classification: ratio = stressed / baseline, higher better
    - Regression: ratio = baseline / stressed, higher better
    """
    ratio = max(0.0, min(2.0, float(ratio)))  # cap extreme improvement
    return _clip_0_100(ratio * 100.0)


def _compute_noise_score(task: str, baseline: dict, noise_block: dict):
    """
    Uses the highest noise-level point (last row) compared to baseline.
    Expects noise_block["output"]["summary_records"] already prepared by app.py.
    """
    if not noise_block or noise_block.get("status") != "ok":
        return None

    out = noise_block.get("output", {}) or {}
    rows = out.get("summary_records")
    if not rows or len(rows) == 0:
        return None

    primary = _primary_metric_key(task)
    base_val = (baseline or {}).get(primary)
    if base_val is None:
        return None

    last = rows[-1]
    stressed_val = last.get(f"metrics.{primary}")
    if stressed_val is None:
        return None

    base_val = float(base_val)
    stressed_val = float(stressed_val)

    if task == "classification":
        ratio = stressed_val / base_val if base_val != 0 else 0.0
        return _score_from_ratio(task, ratio)
    else:
        ratio = base_val / stressed_val if stressed_val != 0 else 0.0
        return _score_from_ratio(task, ratio)


def _compute_missingness_score(missing_block: dict):
    """
    Uses missingness output robustness_score (already computed in your missingness test).
    """
    if not missing_block or missing_block.get("status") != "ok":
        return None
    out = missing_block.get("output", {}) or {}
    return _clip_0_100(out.get("robustness_score"))


def _compute_feature_drop_score(task: str, baseline: dict, fd_block: dict):
    """
    Converts feature-drop sensitivity into a stability score.
    Uses worst impact among ranked features.
    score = 100 * (1 - worst_impact / |baseline_primary|)
    """
    if not fd_block or fd_block.get("status") != "ok":
        return None

    out = fd_block.get("output", {}) or {}
    if isinstance(out, dict) and out.get("status") == "skip":
        return None

    ranked = out.get("ranked")
    if not ranked or len(ranked) == 0:
        return None

    primary = _primary_metric_key(task)
    base_val = (baseline or {}).get(primary)
    if base_val is None:
        return None

    base_val = float(base_val)

    worst_impact = None
    for r in ranked:
        try:
            imp = float(r.get("impact"))
        except Exception:
            continue
        if worst_impact is None or imp > worst_impact:
            worst_impact = imp

    if worst_impact is None:
        return None

    denom = abs(base_val) if abs(base_val) > 1e-9 else 1.0
    score = 100.0 * (1.0 - (worst_impact / denom))
    return _clip_0_100(score)


def _compute_covshift_score(cov_block: dict):
    """
    Phase 5 score from covariate shift output.
    Uses:
      - max_degradation_pct (0..100, higher worse)
      - shift_sensitivity_index (SSI, higher worse)
    Score idea (traditional, practical):
      score = 100 - (0.7 * max_deg + 0.3 * min(100, SSI))
    """
    if not cov_block or cov_block.get("status") != "ok":
        return None

    out = cov_block.get("output", {}) or {}
    summary = out.get("summary", {}) or {}

    try:
        max_deg = float(summary.get("max_degradation_pct"))
    except Exception:
        max_deg = None

    try:
        ssi = float(summary.get("shift_sensitivity_index"))
    except Exception:
        ssi = None

    if max_deg is None and ssi is None:
        return None

    if max_deg is None:
        max_deg = 0.0
    if ssi is None:
        ssi = 0.0

    ssi_capped = max(0.0, min(100.0, ssi))
    penalty = 0.7 * max(0.0, max_deg) + 0.3 * ssi_capped
    score = 100.0 - penalty
    return _clip_0_100(score)


def compute_overall_robustness(results: dict):
    """
    Returns:
      {
        "overall": 0â€“100,
        "components": {"noise":..., "missingness":..., "feature_drop":..., "covariate_shift":...},
        "weights_used": {...}
      }
    """
    if not isinstance(results, dict):
        return None

    task = results.get("task")
    baseline = results.get("baseline") or {}

    stress = results.get("stress") or {}
    blocks = (stress.get("results") or {}) if isinstance(stress, dict) else {}

    noise_block = blocks.get("noise")
    missing_block = blocks.get("missingness")
    fd_block = blocks.get("feature_drop")
    cov_block = blocks.get("covariate_shift")

    noise_score = _compute_noise_score(task, baseline, noise_block)
    missing_score = _compute_missingness_score(missing_block)
    fd_score = _compute_feature_drop_score(task, baseline, fd_block)
    cov_score = _compute_covshift_score(cov_block)

    components = {
        "noise": noise_score,
        "missingness": missing_score,
        "feature_drop": fd_score,
        "covariate_shift": cov_score,
    }

    # base weights (re-normalized based on availability)
    weights = {
        "noise": 0.30,
        "missingness": 0.30,
        "feature_drop": 0.20,
        "covariate_shift": 0.20,
    }

    available = {k: v for k, v in components.items() if v is not None}
    if not available:
        return None

    wsum = sum(weights[k] for k in available.keys())
    weights_used = {k: weights[k] / wsum for k in available.keys()}

    overall = 0.0
    for k, score in available.items():
        overall += float(score) * weights_used[k]

    return {
        "overall": _clip_0_100(overall),
        "components": components,
        "weights_used": weights_used,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("dataset")
        target_col = request.form.get("target", "").strip()

        # checkbox (optional stress)
        run_stress = request.form.get("run_stress") == "1"

        # ---------- Basic form validation ----------
        if not file or file.filename == "":
            return render_template(
                "error.html",
                title="Dataset Missing",
                code=400,
                message="Please upload a CSV file.",
                hint="Click Return Home and choose a valid .csv dataset.",
                path=request.path,
            ), 400

        if target_col == "":
            return render_template(
                "error.html",
                title="Target Column Missing",
                code=400,
                message="Please specify the target column.",
                hint="Enter the exact column name (case-sensitive) and try again.",
                path=request.path,
            ), 400

        # ---------- Read CSV ----------
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template(
                "error.html",
                title="CSV Read Error",
                code=400,
                message="Could not read the uploaded CSV file.",
                hint="Ensure it is a valid CSV (comma-separated) and not an Excel file.",
                path=request.path,
                details=str(e) if app.debug else None,
            ), 400

        # ---------- Basic validation ----------
        try:
            validate_inputs(df, target_col)
        except Exception as e:
            return render_template(
                "error.html",
                title="Input Validation Error",
                code=400,
                message="Your dataset or target column did not pass validation.",
                hint="Confirm the target column exists and dataset is usable.",
                path=request.path,
                details=str(e) if app.debug else None,
            ), 400

        # ---------- Drop target NaNs ----------
        df_clean, dropped = clean_target_nans(df, target_col)

        if df_clean.shape[0] < 30:
            return render_template(
                "error.html",
                title="Not Enough Rows",
                code=400,
                message=(
                    f"Dropped {dropped} rows with missing target values. "
                    f"Only {df_clean.shape[0]} rows remain."
                ),
                hint="Use a dataset with more rows or fewer missing values in the target.",
                path=request.path,
            ), 400

        # ---------- Train baseline (ALWAYS) ----------
        try:
            baseline_out = train_and_evaluate_baseline(df_clean, target_col)
            model, results, split = _normalize_baseline_output(baseline_out)
        except Exception as e:
            return render_template(
                "error.html",
                title="Model Training Error",
                code=500,
                message="The model failed during training/evaluation.",
                hint="Try a simpler dataset or review logs.",
                path=request.path,
                details=str(e) if app.debug else None,
            ), 500

        # ---------- Enrich baseline results ----------
        results["target"] = target_col
        results["dropped_target_nan_rows"] = int(dropped)
        results["run_stress"] = bool(run_stress)

        # Preview for UI (audit-friendly)
        preview_df = df_clean.head(25).copy()
        results["preview_cols"] = list(preview_df.columns)
        results["preview_rows"] = _df_to_records(preview_df, limit=25)

        # ---------- OPTIONAL: Run Stress Suite ----------
        stress_suite = None
        stress_warning = None

        if run_stress:
            # Prefer test split if provided, else fall back to full clean df
            test_df = None
            if isinstance(split, dict):
                test_df = split.get("test_df")

            if test_df is None:
                test_df = df_clean
                stress_warning = (
                    "Note: No test split provided by baseline. "
                    "Stress ran on full cleaned dataset."
                )

            try:
                stress_suite = run_all_stress_tests(
                    model=model,
                    df=test_df,
                    target_col=target_col,
                    split=split,  # âœ… enables Phase 3 leakage-safe mode
                )
            except Exception as e:
                stress_warning = f"Stress tests failed: {e}"
                stress_suite = None

        results["stress"] = stress_suite
        results["stress_warning"] = stress_warning

        # ---------- Convert summary_df for UI (Noise table) ----------
        noise_block = _get_noise_block(stress_suite) if stress_suite else None
        if (
            noise_block
            and isinstance(noise_block, dict)
            and noise_block.get("status") == "ok"
        ):
            noise_output = noise_block.get("output", {}) or {}
            summary_df = noise_output.get("summary_df")

            if summary_df is not None:
                noise_output["summary_records"] = _df_to_records(summary_df)
                noise_output.pop("summary_df", None)
                noise_block["output"] = noise_output

        # ---------- Normalize Missingness output for UI (Phase 4) ----------
        ms_block = _get_missingness_block(stress_suite) if stress_suite else None
        if (
            ms_block
            and isinstance(ms_block, dict)
            and ms_block.get("status") == "ok"
        ):
            ms_output = ms_block.get("output", {}) or {}

            curve = ms_output.get("curve")
            if isinstance(curve, list) and len(curve) > 0:
                curve_records = []
                for p in curve:
                    if not isinstance(p, dict):
                        continue
                    lvl = p.get("missingness_level")
                    m = _flatten_metrics(p.get("metrics", {}))

                    row = {
                        "missingness_level": lvl,
                        **m,
                        "metrics": m,
                    }
                    curve_records.append(row)

                ms_output["curve_records"] = curve_records

            ms_block["output"] = ms_output

        # ---------- Phase 5: Covariate Shift ----------
        # Your covariate_shift test already returns `curve_records` in the right flat form,
        # so we don't need extra conversions here. This is just a safe hook in case you
        # later want to normalize fields.
        cs_block = _get_covshift_block(stress_suite) if stress_suite else None
        if cs_block and isinstance(cs_block, dict) and cs_block.get("status") == "ok":
            cs_output = cs_block.get("output", {}) or {}
            cs_block["output"] = cs_output

        # ---------- Compute robustness (NOW ACTUALLY ATTACH IT) ----------
        results["robustness"] = compute_overall_robustness(results)

        return render_template("report.html", results=results)

    return render_template("index.html")


# ---------- Global Error Handlers ----------

@app.errorhandler(404)
def not_found(e):
    return render_template(
        "error.html",
        title="Page Not Found",
        code=404,
        message="The page you're looking for doesn't exist.",
        hint="Check the URL or return home.",
        path=request.path,
    ), 404


@app.errorhandler(400)
def bad_request(e):
    return render_template(
        "error.html",
        title="Bad Request",
        code=400,
        message="The request could not be processed.",
        hint="Return home and try again.",
        path=request.path,
    ), 400


@app.errorhandler(500)
def server_error(e):
    debug_details = traceback.format_exc() if app.debug else None
    return render_template(
        "error.html",
        title="Server Error",
        code=500,
        message="Something went wrong while processing your request.",
        hint="Return home and retry. If it persists, check logs.",
        path=request.path,
        details=debug_details,
    ), 500


if __name__ == "__main__":
    app.run(debug=True)
