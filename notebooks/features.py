"""
features.py

This file adds extra columns to the cleaned datasets and saves the result in data/features.
Run it after clean_data.py and before preprocess_data.py.

All features are calculated straight from each applicant's own information using simple rules.
Nothing is learned from the dataset so there is nothing to train or fit. The same code runs on
both the training and test data.

Same as in clean_data.py: -1 means the value is unknown. Any input that is -1 is treated as
unknown and if a feature cannot be calculated it is stored as -1.

Features

is_thin_file: 1 if the credit bureau has no records for the applicant. This is the group the
    whole project is about.
has_prior_application: 1 if the applicant has applied to us before.
has_tax_record: 1 if the tax registry has information about the applicant.
external_coverage: how many of the three outside sources have information about the applicant
    (0 to 3).
total_external_records: total number of records across the credit bureau + previous applications
    + tax registry.
n_unknown: how many of the applicant's numeric values are unknown (-1).
age_in_years: the applicant's age on the day the decision was made.
annuity_to_income: the loan payment compared with income
    (annuity_780A / mainoccupationinc_384A).
credit_to_income: the requested credit amount compared with income
    (credamount_770A / mainoccupationinc_384A).
credit_to_annuity: the requested credit amount compared with the loan payment. Roughly how many
    payments it would take (credamount_770A / annuity_780A).
income_vs_tax: declared income compared with the average income in the tax registry
    (mainoccupationinc_384A / amount_4527230A_mean).
requested_vs_past_credit: the requested amount compared with the applicant's average past loan at
    the bureau (credamount_770A / credamount_770A_mean).
requested_vs_prior_application: the requested amount compared with the average amount in the
    applicant's earlier applications to us (credamount_770A / credamount_590A_mean).
employment_share_of_life: years employed compared with age.
overdue_to_credit: the largest overdue amount compared with the largest past loan at the bureau
    (overdueamountmax_950A_max / credamount_770A_max).
payments_per_contract: recorded payments divided by the number of past loans at the bureau
    (payment_rowcount_sum / bureau_rowcount).
dpd_bucket: the worst late payment on a past loan grouped into four buckets.
    0 never late. 1 up to 30 days late. 2 up to 60 days late. 3 more than 60 days late.
months_since_last_application: months since the applicant's most recent approved application.
    Read from the raw applprev_1 table because data.py drops the date.

Not included

"months since last bureau payment" is not calculated. The pmts_month_158T column only gives a
month number from 1 to 12 with no year so there is no way to tell how long ago a payment was.
"""

import os
import numpy as np
import pandas as pd

INPUT_DIR = "data/cleaned"
RAW_DIR = "data"
OUTPUT_DIR = "data/features"

# clean_data.py writes -1 wherever a numeric value is unknown
MISSING_VALUE = -1

# ids and time columns, never used inside a feature
ID_COLS = ["case_id", "target", "MONTH", "WEEK_NUM"]

DAYS_PER_YEAR = 365.2425   # average year, leap years included
DAYS_PER_MONTH = 30.4375   # average month

# edges for dpd_bucket: never late / up to 30 days / up to 60 / over 60
DPD_BINS = [-np.inf, 0, 30, 60, np.inf]


def _check_data_files_exist(input_dir: str, raw_dir: str, output_dir: str) -> None:
    """Verify the cleaned files and the raw applprev tables exist. Creates output_dir if needed."""
    required = []
    for prefix in ("train", "test"):
        required.append(f"{input_dir}/{prefix}_cleaned.csv")
        required.append(f"{raw_dir}/{prefix}/{prefix}_applprev_1.csv")
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Missing required data file(s):\n  " + "\n  ".join(missing) +
            "\n\nRun notebooks/data.py and notebooks/clean_data.py first."
        )
    os.makedirs(output_dir, exist_ok=True)


def unknown_to_nan(values: pd.Series) -> pd.Series:
    """Turn the -1 unknown marker into NaN, so arithmetic on it cannot produce a real-looking number."""
    values = values.astype(float)
    return values.mask(values == MISSING_VALUE)


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator. Written as -1 where either side is unknown or the denominator is 0."""
    num = unknown_to_nan(numerator)
    den = unknown_to_nan(denominator)
    den = den.mask(den == 0)
    return (num / den).fillna(MISSING_VALUE)


def last_prior_approval(raw_dir: str, prefix: str) -> pd.Series:
    """Latest prior approval date per case_id from the raw applprev_1 table. NaT when no date is known."""
    applprev = pd.read_csv(
        f"{raw_dir}/{prefix}/{prefix}_applprev_1.csv",
        usecols=["case_id", "approvaldate_319D"],
    )
    applprev["approvaldate_319D"] = pd.to_datetime(applprev["approvaldate_319D"])
    return applprev.groupby("case_id")["approvaldate_319D"].max()


def add_features(df: pd.DataFrame, last_approval: pd.Series) -> pd.DataFrame:
    """Add every feature to a cleaned dataframe. Drops birth_259D once age has been worked out."""
    df = df.copy()

    # how sparse is this applicant: count the unknowns before we add anything
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in ID_COLS]
    df["n_unknown"] = (df[numeric_cols] == MISSING_VALUE).sum(axis=1)

    # who has a record where. The row counts are 0 (not -1) when there are no records,
    # so a plain comparison is enough here.
    df["is_thin_file"] = (df["bureau_rowcount"] == 0).astype(int)
    df["has_prior_application"] = (df["applprev_rowcount"] > 0).astype(int)
    df["has_tax_record"] = (df["tax_rowcount"] > 0).astype(int)
    df["external_coverage"] = (1 - df["is_thin_file"]) + df["has_prior_application"] + df["has_tax_record"]
    df["total_external_records"] = df["bureau_rowcount"] + df["applprev_rowcount"] + df["tax_rowcount"]

    # age on the decision date. The raw birth date is not needed after this.
    decision_date = pd.to_datetime(df["date_decision"])
    df["age_in_years"] = (decision_date - pd.to_datetime(df["birth_259D"])).dt.days / DAYS_PER_YEAR
    df = df.drop(columns=["birth_259D"])

    # affordability. These come from the applicant's own form, so thin-file applicants have them too.
    df["annuity_to_income"] = ratio(df["annuity_780A"], df["mainoccupationinc_384A"])
    df["credit_to_income"] = ratio(df["credamount_770A"], df["mainoccupationinc_384A"])
    df["credit_to_annuity"] = ratio(df["credamount_770A"], df["annuity_780A"])
    df["income_vs_tax"] = ratio(df["mainoccupationinc_384A"], df["amount_4527230A_mean"])
    df["requested_vs_past_credit"] = ratio(df["credamount_770A"], df["credamount_770A_mean"])
    df["requested_vs_prior_application"] = ratio(df["credamount_770A"], df["credamount_590A_mean"])
    years_employed = unknown_to_nan(df["days_employed_700P"]) / DAYS_PER_YEAR
    df["employment_share_of_life"] = ratio(years_employed, df["age_in_years"])

    # past-loan behaviour. Unknown (-1) for thin-file applicants, since they have no past loans.
    df["overdue_to_credit"] = ratio(df["overdueamountmax_950A_max"], df["credamount_770A_max"])
    df["payments_per_contract"] = ratio(df["payment_rowcount_sum"], df["bureau_rowcount"])

    worst_dpd = unknown_to_nan(df["pmts_dpdvalue_108P_max"])
    bucket = pd.cut(worst_dpd, bins=DPD_BINS, labels=[0, 1, 2, 3]).astype(float)   # NaN stays NaN
    df["dpd_bucket"] = bucket.fillna(MISSING_VALUE).astype(int)

    # how long ago the applicant last got a loan from us. -1 if never, or if the date is unknown.
    days_since = (decision_date - df["case_id"].map(last_approval)).dt.days
    df["months_since_last_application"] = (days_since / DAYS_PER_MONTH).fillna(MISSING_VALUE)

    return df


def _verify(train: pd.DataFrame, test: pd.DataFrame, n_train: int, n_test: int, new_cols: list) -> None:
    """Sanity checks on the new columns before anything is written."""
    assert len(train) == n_train and len(test) == n_test, "row count changed"
    assert set(train.columns) - {"target"} == set(test.columns), "train/test columns differ"

    for df, name in ((train, "train"), (test, "test")):
        features = df[new_cols]
        assert not features.isna().any().any(), f"{name}: NaN in new features"
        assert np.isfinite(features.to_numpy()).all(), f"{name}: inf in new features"
        # every feature is a count, a flag, a ratio of positive amounts or a time gap,
        # so the only value allowed below zero is the -1 unknown marker
        negative = ((features < 0) & (features != MISSING_VALUE)).any()
        assert not negative.any(), f"{name}: negative values in {negative[negative].index.tolist()}"


def prepare_datasets(input_dir: str, raw_dir: str, output_dir: str) -> None:
    """Load the cleaned splits, add the features, write the result."""
    _check_data_files_exist(input_dir, raw_dir, output_dir)

    train = pd.read_csv(f"{input_dir}/train_cleaned.csv")
    test = pd.read_csv(f"{input_dir}/test_cleaned.csv")
    n_train, n_test = len(train), len(test)

    train_features = add_features(train, last_prior_approval(raw_dir, "train"))
    test_features = add_features(test, last_prior_approval(raw_dir, "test"))

    new_cols = [c for c in train_features.columns if c not in train.columns]
    _verify(train_features, test_features, n_train, n_test, new_cols)

    train_features.to_csv(f"{output_dir}/train_features.csv", index=False)
    test_features.to_csv(f"{output_dir}/test_features.csv", index=False)

    print(f"-> wrote features datasets ({len(new_cols)} new columns, {train_features.shape[1]} total) to {output_dir}")
    print(f"   thin-file share in train: {train_features['is_thin_file'].mean():.1%}")


if __name__ == "__main__":
    prepare_datasets(INPUT_DIR, RAW_DIR, OUTPUT_DIR)
