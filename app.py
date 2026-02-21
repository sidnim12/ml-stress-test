# app.py
from flask import Flask, render_template, request
import pandas as pd
import traceback

from stress.schemas import validate_inputs, clean_target_nans
from models.baseline import train_and_evaluate_baseline
from stress.runner import run_all_stress_tests

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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("dataset")
        target_col = request.form.get("target", "").strip()

        # checkbox (optional stress)
        run_stress = request.form.get("run_stress") == "1"

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

        # ✅ NEW: Preview for UI (audit-friendly + first impression)
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
                test_df = df_clean  # safe fallback
                stress_warning = (
                    "Note: No test split provided by baseline. "
                    "Stress ran on full cleaned dataset."
                )

            try:
                stress_suite = run_all_stress_tests(
                    model=model,
                    df=test_df,
                    target_col=target_col,
                )
            except Exception as e:
                # Do NOT kill baseline report; show warning
                stress_warning = f"Stress tests failed: {e}"
                stress_suite = None

        results["stress"] = stress_suite
        results["stress_warning"] = stress_warning

        # ---------- Convert summary_df for UI ----------
        noise_block = _get_noise_block(stress_suite) if stress_suite else None
        if (
            noise_block
            and isinstance(noise_block, dict)
            and noise_block.get("status") == "ok"
        ):
            noise_output = noise_block.get("output", {})
            summary_df = noise_output.get("summary_df")

            if summary_df is not None:
                noise_output["summary_records"] = _df_to_records(summary_df)
                noise_output.pop("summary_df", None)

                # ✅ IMPORTANT: ensure the nested dict is updated for the template
                noise_block["output"] = noise_output

        return render_template("report.html", results=results)

    return render_template("index.html")


# ---------- Global Error Handlers ----------

@app.errorhandler(404)
def not_found(e):
    return render_template(
        "error.html",
        title="Page Not Found",
        code=404,
        message="The page you’re looking for doesn’t exist.",
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