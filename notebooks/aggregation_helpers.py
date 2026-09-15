"""
aggregation_helpers.py

Helper functions for creating the aggregated datasets (train/test).
"""

import pandas as pd

def concat_and_sort(df_0: pd.DataFrame, df_1: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two dataframes and sort by case_id."""
    return pd.concat([df_0, df_1], axis=0).sort_values(by='case_id').reset_index(drop=True)

def fit_categories(train_df: pd.DataFrame, col: str) -> list:
    """
    Get the fixed list of categories for a column from TRAINING data.
    Apply this same list when aggregating both train & test so one-hot
    columns line up across both sets.
    """
    return sorted(train_df[col].dropna().unique().tolist())

def mean_median_max_of(col: str) -> dict:
    """Return a dict of aggregation functions for a numeric column, to be used in groupby.agg()."""
    return {
        f"{col}_mean": (col, "mean"),
        f"{col}_max": (col, "max"),
        f"{col}_median": (col, "median"),
    }

def onehot_encode_and_sum(df: pd.DataFrame, col: str, category_list=None) -> pd.DataFrame:
    """
    One-hot encode a categorical column & sum one-hot rows into one row per case_id.
    If category_list provided, use it as fixed list of categories for one-hot encoding.
    """
    if category_list is not None:
        cat_series = pd.Categorical(df[col], categories=category_list)
    else:
        cat_series = df[col]

    onehot = pd.get_dummies(cat_series, prefix=col)
    onehot["case_id"] = df["case_id"]
    onehot_agg = onehot.groupby("case_id").sum().reset_index()

    return onehot_agg