# models/baseline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge

from stress.schemas import detect_task_type
from stress.metrics import (
    ClassificationConfig,
    RegressionConfig,
    classification_metrics,
    regression_metrics,
    compute_metrics,
)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocessing:
    - Numeric: median impute + standard scale
    - Categorical: most_frequent impute + one-hot encode (ignore unknown)
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def train_and_evaluate_baseline(
    df: pd.DataFrame,
    target_col: str,
    seed: int = 42,
):
    """
    Returns:
      (model, results, split)

    split contains:
      X_train, y_train, X_test, y_test, test_df
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    task = detect_task_type(y)

    stratify = None
    if task == "classification":
        stratify = y  # may still fail if class counts are tiny; sklearn will raise

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train)

    if task == "classification":
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )
    else:
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", Ridge()),
            ]
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_proba = None
    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    if task == "classification":
        metrics = classification_metrics(ClassificationConfig())
        metric_out = compute_metrics(metrics, y_true=y_test, y_pred=y_pred, y_proba=y_proba)

        baseline = {
            "accuracy": metric_out.get("accuracy"),
            "balanced_accuracy": metric_out.get("balanced_accuracy"),
            "f1": metric_out.get("f1"),
            "precision": metric_out.get("precision"),
            "recall": metric_out.get("recall"),
            "roc_auc": metric_out.get("roc_auc"),
            "pr_auc": metric_out.get("pr_auc"),
            "log_loss": metric_out.get("log_loss"),
        }
    else:
        metrics = regression_metrics(RegressionConfig())
        metric_out = compute_metrics(metrics, y_true=y_test, y_pred=y_pred, y_proba=None)

        baseline = {
            "rmse": metric_out.get("rmse"),
            "mae": metric_out.get("mae"),
            "r2": metric_out.get("r2"),
            "mape_percent": metric_out.get("mape_percent"),
            "smape_percent": metric_out.get("smape_percent"),
        }

    def _round_or_nan(v, nd=4):
        try:
            if v is None:
                return None
            v = float(v)
            if v != v:  # NaN
                return float("nan")
            return round(v, nd)
        except Exception:
            return float("nan")

    baseline = {k: _round_or_nan(v) for k, v in baseline.items()}

    results = {
        "task": task,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "baseline": baseline,
    }

    split = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_df": pd.concat([X_test, y_test.rename(target_col)], axis=1),
    }

    return model, results, split