# stress/schemas.py
import pandas as pd


def detect_task_type(y: pd.Series) -> str:
    """
    Heuristic task detection:
    - object/category/bool => classification
    - numeric with small unique count (<=20) => classification
    - else => regression
    """
    if y.dtype.name in ("object", "category", "bool"):
        return "classification"

    nunique = y.nunique(dropna=True)
    if nunique <= 20:
        return "classification"

    return "regression"


def validate_inputs(df: pd.DataFrame, target_col: str) -> None:
    """
    Validates basic inputs (does NOT mutate the dataframe).
    Target NaN cleaning is handled separately via clean_target_nans().
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    if df.shape[0] < 30:
        raise ValueError("Dataset too small. Please provide at least ~30 rows.")


def clean_target_nans(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, int]:
    """
    Drops rows where the target is NaN.
    We do NOT impute targets, because missing labels are unknown labels.
    Returns:
      (clean_df, dropped_rows_count)
    """
    before = df.shape[0]
    df2 = df.dropna(subset=[target_col]).copy()
    dropped = before - df2.shape[0]
    return df2, int(dropped)