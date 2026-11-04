"""
prepare_lgbm_data.py

Builds the LightGBM dataset from data/aggregated and saves it in data/lgbm.

LightGBM handles missing values and text categories itself, so unlike clean_data.py ->
preprocess_data.py (built for logistic regression) this keeps unknowns as NaN: no -1 fill,
no imputing, scaling or one-hot encoding. Other teams still use data/cleaned, so this is a
separate script instead of a change to clean_data.py.

Steps
1. Fill structural counts with 0: NaN there means "no records", not "unknown".
2. Add the features from features.py, with NaN where a feature cannot be calculated.
3. Drop n_unknown: it counts -1 markers, which this data does not have.
"""

import os
import numpy as np
import pandas as pd
from features import add_features, last_prior_approval

INPUT_DIR = "data/aggregated"
RAW_DIR = "data"
OUTPUT_DIR = "data/lgbm"

# NaN in these columns means "no records", so the true value is 0
STRUCTURAL_COUNT_COLS = [
    "applprev_rowcount",
    "person_rowcount",
    "tax_rowcount",
    "bureau_rowcount",
    "payment_rowcount_sum",
]
STRUCTURAL_COUNT_PREFIXES = [
    "status_219L_",
    "name_4527232M_",
    "classificationofcontr_13M_",
]


def _check_data_files_exist(input_dir: str, raw_dir: str, output_dir: str) -> None:
    """Verify the aggregated files and the raw applprev tables exist. Creates output_dir if needed."""
    required = []g
    for prefix in ("train", "test"):
        required.append(f"{input_dir}/{prefix}_aggregated.csv")
        required.append(f"{raw_dir}/{prefix}/{prefix}_applprev_1.csv")
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required data file(s):\n  " + "\n  ".join(missing) +
            "\n\nRun notebooks/data.py first."
        )
    os.makedirs(output_dir, exist_ok=True)


def fill_structural_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the row-count and category-count columns with 0. Every other NaN is left as is."""
    df = df.copy()
    # list() makes a new list, so extend() does not change STRUCTURAL_COUNT_COLS itself
    columns = list(STRUCTURAL_COUNT_COLS)
    for p in STRUCTURAL_COUNT_PREFIXES:
        columns.extend([c for c in df.columns if c.startswith(p)])
    df[columns] = df[columns].fillna(0)
    return df


def build_lgbm_data(df: pd.DataFrame, last_approval: pd.Series) -> pd.DataFrame:
    """Aggregated dataframe -> LightGBM-ready dataframe."""
    # structural zeros first: is_thin_file in add_features checks bureau_rowcount == 0
    df = fill_structural_zeros(df)
    df = add_features(df, last_approval, missing_value=np.nan)
    return df.drop(columns=["n_unknown"])


def _verify(train: pd.DataFrame, test: pd.DataFrame, n_train: int, n_test: int) -> None:
    """Sanity checks before anything is written."""
    assert len(train) == n_train and len(test) == n_test, "row count changed"
    assert set(train.columns) - {"target"} == set(test.columns), "train/test columns differ"

    for df, name in ((train, "train"), (test, "test")):
        numeric = df.select_dtypes(include="number")
        assert not (numeric == -1).any().any(), f"{name}: -1 marker found, should be NaN"
        assert np.isfinite(numeric.fillna(0).to_numpy()).all(), f"{name}: inf found"
        assert not df["bureau_rowcount"].isna().any(), f"{name}: structural counts not filled"


def prepare_datasets(input_dir: str, raw_dir: str, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the aggregated splits, build the LightGBM datasets, write the result."""
    _check_data_files_exist(input_dir, raw_dir, output_dir)

    train = pd.read_csv(f"{input_dir}/train_aggregated.csv")
    test = pd.read_csv(f"{input_dir}/test_aggregated.csv")
    n_train, n_test = len(train), len(test)

    train = build_lgbm_data(train, last_prior_approval(raw_dir, "train"))
    test = build_lgbm_data(test, last_prior_approval(raw_dir, "test"))
    _verify(train, test, n_train, n_test)

    train.to_csv(f"{output_dir}/train_lgbm.csv", index=False)
    test.to_csv(f"{output_dir}/test_lgbm.csv", index=False)

    print(f"-> wrote LightGBM datasets ({train.shape[1]} columns) to {output_dir}")
    print(f"   thin-file share in train: {train['is_thin_file'].mean():.1%}")
    return train, test


if __name__ == "__main__":
    prepare_datasets(INPUT_DIR, RAW_DIR, OUTPUT_DIR)
