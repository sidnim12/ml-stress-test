# app.py
from flask import Flask, render_template, request
import pandas as pd

from stress.schemas import validate_inputs, clean_target_nans
from models.baseline import train_and_evaluate_baseline

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("dataset")
        target_col = request.form.get("target", "").strip()

        if not file or file.filename == "":
            return "Please upload a CSV file."

        if target_col == "":
            return "Please specify the target column."

        # Read CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return f"CSV Error: {e}"

        # Basic validation
        try:
            validate_inputs(df, target_col)
        except Exception as e:
            return f"Input Error: {e}"

        # Drop target NaNs (do not impute target)
        df_clean, dropped = clean_target_nans(df, target_col)

        if df_clean.shape[0] < 30:
            return (
                f"Input Error: Dropped {dropped} rows with missing target values. "
                f"Only {df_clean.shape[0]} rows remain."
            )

        # Train + Evaluate baseline
        try:
            _, results = train_and_evaluate_baseline(df_clean, target_col)
        except Exception as e:
            return f"Model training error: {e}"

        # Enrich results for report
        results["target"] = target_col
        results["dropped_target_nan_rows"] = int(dropped)

        return render_template("report.html", results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)