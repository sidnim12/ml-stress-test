from flask import Flask, render_template, request
from models.baseline import train_baseline_model

import pandas as pd

app = Flask(__name__)

@app.route("/", methods= ["GET", "POST"])
def index():
    
    if request.method == "POST":
        
        file = request.files.get("dataset")
        target_col = request.form.get("target", "").strip()
        
        if not file or file.filename == "":
            return "Please upload a CSV file."
        
        if target_col == "":
            return "Please specify the target column."
        
        
        try:
            df = pd.read_csv(file)
            
        except Exception as e:
            return f"CSV error : {e}"
        
        if target_col not in df.columns:
            return f"Target column '{target_col}' not found in dataset."
        
        try:
            baseline_accuracy = train_baseline_model(df, target_col)
            
        except Exception as e:
            return f"Model training error : {e}"
        
        
        return render_template(
            "report.html",
            rows = df.shape[0],
            cols = df.shape[1],
            target = target_col,
            baseline_metric = round(baseline_accuracy, 4)
        )
        
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)