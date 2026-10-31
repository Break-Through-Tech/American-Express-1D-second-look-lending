"""
First-pass logistic regression for American Express second-look lending.

Run from repository root:
    python models/logistic_baseline.py

Optional:
    python models/logistic_baseline.py --max-default-rate 0.10

Uses models/splits.py for chronological splits.

Target:
    0 = repaid
    1 = defaulted

Stage 1:
    Low predicted default risk -> first-pass approval
    Higher predicted default risk -> second-look referral

Exports predictions for train, tune, holdout, and score.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Paths and imports
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(MODELS_DIR))

from splits import make_splits, get_features_target_segment


# --------------------------------------------------
# Feature preprocessing
# --------------------------------------------------

def clean_X(X):
    """
    Prepare features for logistic regression.

    Exclude explicit thin-file status from predictions.
    Keep the segment separately for fairness evaluation.
    """
    X = X.copy()

    X = X.drop(
        columns=["is_thin_file"],
        errors="ignore"
    )

    # Preserve existing baseline's -1 handling.
    # Review sentinel semantics before finalizing the model.
    for col in X.select_dtypes(include=["number"]).columns:
        X[col] = X[col].replace(-1, np.nan)

    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def make_preprocessor(X):
    """
    Build preprocessing pipeline.

    Fit transformations on training data only.
    """

    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        col for col in X.columns
        if col not in numeric_cols
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])


# --------------------------------------------------
# Approval threshold
# --------------------------------------------------

def choose_threshold(
    y,
    probabilities,
    max_default_rate,
    min_approvals=100
):
    """
    Find the threshold allowing the most first-pass
    approvals while satisfying the observed default-rate
    ceiling on the tune dataset.

    This is a demonstration policy, not a guarantee
    about future default rates.
    """

    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities)

    order = np.argsort(
        probabilities,
        kind="stable"
    )

    sorted_probs = probabilities[order]
    sorted_y = y[order]

    cumulative_defaults = np.cumsum(sorted_y)

    # Consider complete groups of equal probability scores.
    ends = np.r_[
        np.flatnonzero(np.diff(sorted_probs) != 0),
        len(sorted_probs) - 1
    ]

    counts = ends + 1
    default_rates = cumulative_defaults[ends] / counts

    valid = ends[
        (counts >= min_approvals)
        & (default_rates <= max_default_rate)
    ]

    if len(valid) == 0:
        raise ValueError(
            "No approval threshold satisfies the "
            "default-rate ceiling with enough approvals."
        )

    best = valid[-1]

    return float(
        np.nextafter(sorted_probs[best], np.inf)
    )


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(
    name,
    y,
    probabilities,
    segments,
    threshold
):
    """
    Evaluate prediction quality, approvals, and
    thin-file inclusion.
    """

    y = np.asarray(y, dtype=int)
    segments = np.asarray(segments, dtype=int)

    approved = probabilities < threshold
    repaid = y == 0

    print(f"\n===== {name} =====")

    if len(np.unique(y)) == 2:
        print(
            "ROC-AUC:",
            round(roc_auc_score(y, probabilities), 4)
        )

    print(
        "LogLoss:",
        round(
            log_loss(y, probabilities, labels=[0, 1]),
            4
        )
    )

    print(
        "Brier score:",
        round(brier_score_loss(y, probabilities), 4)
    )

    print(
        "Approval rate:",
        f"{approved.mean():.2%}"
    )

    print(
        "Referred to second look:",
        f"{(~approved).sum():,}"
    )

    if approved.any():
        print(
            "Default rate among approved:",
            f"{y[approved].mean():.2%}"
        )

    established = repaid & (segments == 0)
    thin_file = repaid & (segments == 1)

    established_rate = (
        approved[established].mean()
        if established.any()
        else np.nan
    )

    thin_file_rate = (
        approved[thin_file].mean()
        if thin_file.any()
        else np.nan
    )

    print(
        "Established repayer approval rate:",
        f"{established_rate:.2%}"
    )

    print(
        "Thin-file repayer approval rate:",
        f"{thin_file_rate:.2%}"
    )

    print(
        "Inclusion gap:",
        f"{established_rate - thin_file_rate:+.2%}"
    )


# --------------------------------------------------
# Export predictions for teammates
# --------------------------------------------------

def save_predictions(
    name,
    df,
    X,
    model,
    threshold,
    output_dir
):
    """
    Export first-stage predictions for one split.

    Teammates can use refer_to_second_look == 1
    to identify applicants for second-stage scoring.

    Training predictions are in-sample and must NOT
    be used as unbiased referral labels for training
    a referral-specialized second-stage model.
    """

    probabilities = model.predict_proba(X)[:, 1]

    approved = (
        probabilities < threshold
    ).astype(int)

    referred = (
        probabilities >= threshold
    ).astype(int)

    results = pd.DataFrame({
        "case_id": df["case_id"].to_numpy(),
        "predicted_default_probability": probabilities,
        "first_pass_approve": approved,
        "refer_to_second_look": referred,
    })

    # Save actual labels for evaluation datasets.
    if name != "score" and "target" in df.columns:
        results["target"] = df["target"].to_numpy()

    # Include thin-file status for analysis only.
    if "is_thin_file" in df.columns:
        results["is_thin_file"] = (
            df["is_thin_file"].to_numpy()
        )

    output_path = (
        output_dir / f"logistic_{name}_predictions.csv"
    )

    results.to_csv(output_path, index=False)

    print(f"\nSaved {name} predictions:")
    print(f"  Total applicants: {len(results):,}")
    print(f"  Approved: {approved.sum():,}")
    print(f"  Referred: {referred.sum():,}")
    print(f"  File: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-default-rate",
        type=float,
        default=0.10,
        help="Illustrative tune-set default-rate ceiling"
    )

    args = parser.parse_args()

    if not 0 < args.max_default_rate < 1:
        parser.error(
            "--max-default-rate must be between 0 and 1"
        )

    # ----------------------------------------------
    # Load feature-engineered datasets
    # ----------------------------------------------

    train_path = (
        ROOT / "data/features/train_features.csv"
    )

    test_path = (
        ROOT / "data/features/test_features.csv"
    )

    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Feature datasets not found. "
            "Run python run_pipeline.py first."
        )

    print("Loading feature datasets...")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # ----------------------------------------------
    # Use existing splits.py
    # ----------------------------------------------

    splits = make_splits(train_df, test_df)

    X_train, y_train, seg_train = (
        get_features_target_segment(splits["train"])
    )

    X_tune, y_tune, seg_tune = (
        get_features_target_segment(splits["tune"])
    )

    X_hold, y_hold, seg_hold = (
        get_features_target_segment(splits["holdout"])
    )

    X_score, _, seg_score = (
        get_features_target_segment(splits["score"])
    )

    # ----------------------------------------------
    # Clean features
    # ----------------------------------------------

    X_train = clean_X(X_train)
    X_tune = clean_X(X_tune)
    X_hold = clean_X(X_hold)
    X_score = clean_X(X_score)

    for name, X in [
        ("tune", X_tune),
        ("holdout", X_hold),
        ("score", X_score),
    ]:
        if X.columns.tolist() != X_train.columns.tolist():
            raise ValueError(
                f"{name} features do not match training features."
            )

    # ----------------------------------------------
    # Hyperparameter tuning
    # ----------------------------------------------

    print("\nTraining logistic regression models...")

    best_loss = np.inf
    best_C = None
    best_model = None

    for C in [0.01, 0.1, 1.0, 10.0]:

        model = Pipeline([
            (
                "preprocess",
                make_preprocessor(X_train)
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    solver="lbfgs"
                )
            ),
        ])

        model.fit(X_train, y_train)

        tune_probs = model.predict_proba(X_tune)[:, 1]

        loss = log_loss(
            y_tune,
            tune_probs,
            labels=[0, 1]
        )

        print(
            f"C={C}: tune LogLoss={loss:.4f}"
        )

        if loss < best_loss:
            best_loss = loss
            best_C = C
            best_model = model

    print(f"\nSelected C: {best_C}")

    # ----------------------------------------------
    # Select stage-one approval threshold
    # ----------------------------------------------

    tune_probs = best_model.predict_proba(X_tune)[:, 1]

    threshold = choose_threshold(
        y_tune,
        tune_probs,
        args.max_default_rate
    )

    print(
        f"Selected risk threshold: {threshold:.2%}"
    )

    # ----------------------------------------------
    # Evaluate
    # ----------------------------------------------

    evaluate(
        "Tune",
        y_tune,
        tune_probs,
        seg_tune,
        threshold
    )

    hold_probs = best_model.predict_proba(X_hold)[:, 1]

    evaluate(
        "Holdout",
        y_hold,
        hold_probs,
        seg_hold,
        threshold
    )

    # ----------------------------------------------
    # Export all predictions
    # ----------------------------------------------

    output_dir = ROOT / "reports/models"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n===== EXPORTING PREDICTIONS =====")

    for name, X in [
        ("train", X_train),
        ("tune", X_tune),
        ("holdout", X_hold),
        ("score", X_score),
    ]:

        save_predictions(
            name=name,
            df=splits[name],
            X=X,
            model=best_model,
            threshold=threshold,
            output_dir=output_dir
        )

    print("\nAll first-stage predictions exported.")


if __name__ == "__main__":
    main()