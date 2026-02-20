# app.py
from flask import Flask, render_template, request
import pandas as pd
import traceback

from stress.schemas import validate_inputs, clean_target_nans
from models.baseline import train_and_evaluate_baseline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("dataset")
        target_col = request.form.get("target", "").strip()

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

        # Read CSV
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

        # Basic validation
        try:
            validate_inputs(df, target_col)
        except Exception as e:
            return render_template(
                "error.html",
                title="Input Validation Error",
                code=400,
                message="Your dataset or target column did not pass validation.",
                hint="Confirm the target column exists and the dataset has usable rows/columns.",
                path=request.path,
                details=str(e) if app.debug else None,
            ), 400

        # Drop target NaNs (do not impute target)
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
                hint="Use a dataset with more rows or fewer missing values in the target column.",
                path=request.path,
            ), 400

        # Train + Evaluate baseline
        try:
            _, results = train_and_evaluate_baseline(df_clean, target_col)
        except Exception as e:
            return render_template(
                "error.html",
                title="Model Training Error",
                code=500,
                message="The model failed during training/evaluation.",
                hint="Try a simpler dataset, check target type, or review logs.",
                path=request.path,
                details=str(e) if app.debug else None,
            ), 500

        # Enrich results for report
        results["target"] = target_col
        results["dropped_target_nan_rows"] = int(dropped)

        return render_template("report.html", results=results)

    return render_template("index.html")


# ---------- Global error handlers (themed) ----------

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
    # Covers malformed requests (rare, but good to have)
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