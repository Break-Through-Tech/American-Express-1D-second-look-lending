"""
splits.py
 
Every model uses this so all models are trained, tuned and scored on the same applications.
 
* splits
    train     weeks 0-63    to fit the model
    tune      weeks 64-75   early stopping, hyperparameter search, calibration, thresholds
    holdout   weeks 76-91   testing with labeled data; to measure AUC, LogLoss, Brier, inclusion gap
    score     test CSV      unlabeled; predictions for final scoring / the demo
 
    Split by time because the model is trained on past applications and used on future ones,
    and thin-file default rates drift over time; a random split would hide that.
 
* how to use
    from splits import make_splits, get_features_target_segment
 
    splits = make_splits(train_df, test_df)    # both loaded from data/features
    X_train, y_train, seg_train = get_features_target_segment(splits["train"])
    # same for "tune" and "holdout"; y is None for "score"
 
    X    features (IDs, dates, target removed; still raw) -> model.fit / model.predict
    y    1 = defaulted, 0 = repaid                        -> training and AUC / LogLoss / Brier
    seg  is_thin_file: 0 = established, 1 = thin-file     -> NOT for training; measures
                                                             results by group for fairness
    Row i of X, y and seg is the same applicant. Do specific model's prep after splitting.
 
* segments
    Every split contains both segment. Filter with seg for per-segment results:
        roc_auc_score(y_hold[seg_hold == 1], pred_hold[seg_hold == 1])   # thin-file AUC
 
    Inclusion gap = approval rate among repayers, established minus thin-file:
        approved, repaid = pred_hold < threshold, y_hold == 0
        gap = approved[repaid & (seg_hold == 0)].mean() - approved[repaid & (seg_hold == 1)].mean()
 
    is_thin_file is also in X; drop it if unwanted.
"""

from typing import Optional

import pandas as pd

WEEK_COL = "WEEK_NUM"
TARGET = "target"
SEGMENT = "is_thin_file"
ID_COL = "case_id"

TRAIN_END_WEEK = 63   # train:   WEEK_NUM <= 63
TUNE_END_WEEK = 75    # tune:    64-75
                      # holdout: >= 76
LABELED_SPLITS = ("train", "tune", "holdout")


def _check_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Fail early if dataframes are missing expected columns"""
    for col in (WEEK_COL, TARGET, SEGMENT, ID_COL):
        if col not in train_df.columns:
            raise KeyError(f"train_df is missing '{col}'")

    if TARGET in test_df.columns:
        print("NOTE: test_df has a target column; it is still treated as the unlabeled score set.")

    missing_in_test = set(train_df.columns) - {TARGET} - set(test_df.columns)
    if missing_in_test:
        raise ValueError(f"test_df is missing columns that train_df has: {sorted(missing_in_test)}")

    overlap = set(train_df[ID_COL]) & set(test_df[ID_COL])
    if overlap:
        raise ValueError(f"{len(overlap)} case_ids appear in both train_df and test_df")


def make_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = True) -> dict:
    """Time-based split of the labeled data.

    Returns {"train", "tune", "holdout", "score"} dataframes with all original columns.
    """
    _check_inputs(train_df, test_df)

    week = train_df[WEEK_COL]
    splits = {
        "train": train_df[week <= TRAIN_END_WEEK].copy(),
        "tune": train_df[(week > TRAIN_END_WEEK) & (week <= TUNE_END_WEEK)].copy(),
        "holdout": train_df[week > TUNE_END_WEEK].copy(),
        "score": test_df.copy(),
    }

    for name in LABELED_SPLITS:
        df = splits[name]
        if df.empty:
            raise ValueError(f"'{name}' split is empty; check WEEK_NUM range")
        # inclusion gap compares approval rates among repayers in each segment, so each split needs enough of both
        for seg_value in (0, 1):
            repayers = ((df[SEGMENT] == seg_value) & (df[TARGET] == 0)).sum()
            # could make this stricter by requiring a higher min number of repayers,
            # so that model can learn a real pattern
            if repayers < 100:
                raise ValueError(f"'{name}' has no repayers with {SEGMENT}={seg_value}")

    if verbose:
        print(summarize(splits).round(4).to_string())
    return splits


def summarize(splits: dict) -> pd.DataFrame:
    """Rows, week range, default rate and thin-file share per split."""
    rows = []
    for name, df in splits.items():
        labeled = TARGET in df.columns and name in LABELED_SPLITS
        rows.append({
            "split": name,
            "rows": len(df),
            "first_week": df[WEEK_COL].min() if WEEK_COL in df else None,
            "last_week": df[WEEK_COL].max() if WEEK_COL in df else None,
            "default_rate": df[TARGET].mean() if labeled else None,
            "thin_file_share": df[SEGMENT].mean() if SEGMENT in df else None,
        })
    return pd.DataFrame(rows).set_index("split")

def get_features_target_segment(df: pd.DataFrame,
                                drop_cols=(ID_COL, "date_decision", WEEK_COL, "MONTH")
                                ) -> tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
    """Separate one split's dataframe into model inputs, labels, and fairness segments.
 
    Returns:
        X:   feature columns (IDs, dates, WEEK_NUM, MONTH and target removed)
        y:   target as int (1 = defaulted, 0 = repaid), or None for the unlabeled "score" split
        seg: is_thin_file as int (0 = established, 1 = thin-file)
    """
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    seg = df[SEGMENT].astype(int)
    X = df.drop(columns=[c for c in (*drop_cols, TARGET) if c in df.columns])
    return X, y, seg