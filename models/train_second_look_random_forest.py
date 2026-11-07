"""
train_second_look_random_forest.py

Second model in the Advisor's design: a random forest for thin-file applicants only, trained on the
columns they actually have. Thin-file applicants the first model rejected are scored by it and a
second cutoff decides.

Run after train_random_forest.py:

    python models/train_second_look_random_forest.py          # about 10 minutes
    python models/train_second_look_random_forest.py --quick  # skip the tuning loop
    python models/train_second_look_random_forest.py --stage1 other_first_model_holdout.csv

--stage1 is the first model's holdout predictions (case_id, is_thin_file, target, p_default).
Same splits as the first model, thin-file rows only. Tables and plots go to
reports/second_look_random_forest, the model to models/artifacts (gitignored).
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import SEGMENT, get_features_target_segment, make_splits
from train_random_forest import (
    ARTIFACTS_DIR, BOOL_COLS, CUTOFF, DEFAULT_SETTINGS, EXCLUDED_COLS, FEATURES_DIR, PREDICTIONS_DIR,
    PROFIT_DEFAULT, PROFIT_REPAID, ROOT_DIR, SWEEP_CUTOFFS, expected_profit, fit_forest, tune,
)

REPORTS_DIR = ROOT_DIR / "reports" / "second_look_random_forest"
STAGE1_FILE = PREDICTIONS_DIR / "random_forest_holdout.csv"
# description_5085714M is a bureau column and always MISSING for thin-file, so only these two
TEXT_COLS = ["education_927M", "maritalstatus_703M"]


# ------------- prep --------------------------------
def thin_file_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[SEGMENT] == 1].reset_index(drop=True)


def dead_columns(X_train: pd.DataFrame) -> list:
    return [col for col in X_train.columns if X_train[col].nunique(dropna=False) <= 1]


def fit_encoder(X_train: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[TEXT_COLS])
    return encoder


def fit_education_order(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    rates = y_train.groupby(X_train["education_927M"], observed=True).mean().sort_values()
    return {cat: rank for rank, cat in enumerate(rates.index)}


def prep(X: pd.DataFrame, encoder: OneHotEncoder, dead: list, edu_map: dict) -> pd.DataFrame:
    X = X.drop(columns=[col for col in dead + EXCLUDED_COLS if col in X.columns])
    X["education_ordinal"] = X["education_927M"].map(edu_map).fillna(-1).astype(float)
    onehot = pd.DataFrame(
        encoder.transform(X[TEXT_COLS]),
        columns=encoder.get_feature_names_out(TEXT_COLS),
        index=X.index,
    )
    X = X.drop(columns=TEXT_COLS)
    for col in BOOL_COLS:
        X[col] = X[col].astype(int)
    return pd.concat([X, onehot], axis=1)

# ------------- evaluation --------------------------------
def thin_metrics(p: np.ndarray, y: pd.Series, cutoff: float = CUTOFF) -> dict:
    approved = p < cutoff
    repaid = (y == 0).to_numpy()
    return {
        "rows": int(len(p)),
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "default_rate": float(y.mean()),
        "mean_predicted_default": float(p.mean()),
        "calibration_gap": float(p.mean() - y.mean()),   # positive = over-predicts risk
        "approval_rate": float(approved.mean()),
        "default_rate_of_approved": float(y[approved].mean()) if approved.any() else np.nan,
        "repayers_approved": float(approved[repaid].mean()),
        "expected_profit": expected_profit(p, y, cutoff),
    }


def two_stage(stage1: pd.DataFrame, p_second: np.ndarray, second_cutoff: float) -> dict:
    y = stage1["target"].to_numpy()
    thin = (stage1[SEGMENT] == 1).to_numpy()
    first = stage1["p_default"].to_numpy() < CUTOFF
    rescued = thin & ~first & (p_second < second_cutoff)   # p_second is NaN for established rows, so they stay out
    approved = first | rescued
    repaid = y == 0
    established_ok = approved[repaid & ~thin].mean()
    thin_ok = approved[repaid & thin].mean()
    return {
        "second_cutoff": second_cutoff,
        "approval_rate": float(approved.mean()),
        "default_rate_of_approved": float(y[approved].mean()),
        "expected_profit": float(np.where(y[approved] == 1, PROFIT_DEFAULT, PROFIT_REPAID).sum()),
        "established_repayers_approved": float(established_ok),
        "thin_file_repayers_approved": float(thin_ok),
        "inclusion_gap": float(established_ok - thin_ok),
        "rescued": int(rescued.sum()),
        "rescued_default_rate": float(y[rescued].mean()) if rescued.any() else np.nan,
    }


def plot_reliability(curves: dict, y: pd.Series, path: str) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5))
    for label, p in curves.items():
        actual, predicted = calibration_curve(y, p, n_bins=15, strategy="quantile")
        axis.plot(predicted, actual, marker="o", ms=3, label=label)
    axis.plot([0, 0.7], [0, 0.7], linestyle="--", color="gray")
    axis.axvline(CUTOFF, linestyle=":", color="red")
    axis.set(title="Thin-file holdout, weeks 76-91", xlabel="predicted default probability",
             ylabel="actual default rate", xlim=(0, 0.7), ylim=(0, 0.7))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_two_stage(sweep: pd.DataFrame, path: str) -> None:
    first_only = sweep[sweep["second_cutoff"] == 0].iloc[0]
    curve = sweep[sweep["second_cutoff"] > 0]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("expected_profit", "profit on holdout ($M)", 1e-6),
        ("inclusion_gap", "inclusion gap", 1),
        ("default_rate_of_approved", "default rate of approved loans", 1),
    ]
    for axis, (col, ylabel, scale) in zip(axes, panels):
        axis.plot(curve["second_cutoff"], curve[col] * scale, marker="o", ms=3, label="first model + second look")
        axis.axhline(first_only[col] * scale, color="grey", linestyle=":", linewidth=1, label="first model only")
        axis.axvline(CUTOFF, color="grey", linestyle="--", linewidth=0.8)
        axis.set(xlabel="second cutoff for thin-file rejects", ylabel=ylabel)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ------------- main --------------------------------
def main(quick: bool, stage1_path: Path) -> None:
    for path in (FEATURES_DIR / "train_features.csv", FEATURES_DIR / "test_features.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run notebooks/features.py first.")
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Missing {stage1_path}. Run models/train_random_forest.py first, or pass --stage1.")
    for folder in (ARTIFACTS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(FEATURES_DIR / "train_features.csv")
    test_df = pd.read_csv(FEATURES_DIR / "test_features.csv")
    splits = {name: thin_file_rows(df) for name, df in make_splits(train_df, test_df).items()}
    print("\nThin-file rows:", {name: len(df) for name, df in splits.items()})

    X_train, y_train, _ = get_features_target_segment(splits["train"])
    X_tune, y_tune, _ = get_features_target_segment(splits["tune"])
    X_hold, y_hold, _ = get_features_target_segment(splits["holdout"])
    X_score, _, _ = get_features_target_segment(splits["score"])

    dead = dead_columns(X_train)
    print(f"\n{len(dead)} columns never vary for thin-file applicants and are dropped:")
    print("  " + ", ".join(dead))
    encoder = fit_encoder(X_train)
    edu_map = fit_education_order(X_train, y_train)
    X_train, X_tune, X_hold, X_score = (
        prep(X, encoder, dead, edu_map) for X in (X_train, X_tune, X_hold, X_score)
    )
    print(f"\nFeatures going into the second-look forest: {X_train.shape[1]}")

    print("\nFitting with default settings...")
    model = fit_forest(X_train, y_train, DEFAULT_SETTINGS)
    p_tune = model.predict_proba(X_tune)[:, 1]
    print(f"default settings  tune AUC {roc_auc_score(y_tune, p_tune):.4f}")

    settings = DEFAULT_SETTINGS
    if not quick:
        print("\nTuning on train, scored on tune...")
        best = tune(X_train, y_train, X_tune, y_tune)
        best_model = fit_forest(X_train, y_train, best)
        p_best = best_model.predict_proba(X_tune)[:, 1]
        print(f"best settings on all of train  tune AUC {roc_auc_score(y_tune, p_best):.4f}")
        if roc_auc_score(y_tune, p_best) > roc_auc_score(y_tune, p_tune):
            model, p_tune, settings = best_model, p_best, best
        else:
            print("tuned settings did not beat the defaults, keeping the defaults")

    # one segment, one calibrator, fitted on tune only
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(p_tune, y_tune)

    # holdout, scored once
    p_hold_raw = model.predict_proba(X_hold)[:, 1]
    p_hold_cal = calibrator.predict(p_hold_raw)
    results = pd.DataFrame([
        {"split": "tune", "version": "raw", **thin_metrics(p_tune, y_tune)},
        {"split": "tune", "version": "calibrated", **thin_metrics(calibrator.predict(p_tune), y_tune)},
        {"split": "holdout", "version": "raw", **thin_metrics(p_hold_raw, y_hold)},
        {"split": "holdout", "version": "calibrated", **thin_metrics(p_hold_cal, y_hold)},
    ])
    print("\nSettings used:", settings)
    print(results.round(4).to_string(index=False))

    # first model's predictions for the same applicants
    stage1 = pd.read_csv(stage1_path)
    needed = ["case_id", SEGMENT, "target", "p_default"]
    missing = [col for col in needed if col not in stage1.columns]
    if missing:
        raise ValueError(f"{stage1_path} needs the columns {needed}, missing {missing}")
    holdout = splits["holdout"][["case_id", "WEEK_NUM", "target"]].copy()
    holdout["p_default_raw"] = p_hold_raw
    holdout["p_default"] = p_hold_cal
    holdout = holdout.merge(stage1[["case_id", "p_default"]].rename(columns={"p_default": "p_first"}),
                            on="case_id", validate="one_to_one")
    if len(holdout) != len(X_hold):
        raise ValueError(f"{stage1_path} does not cover every thin-file holdout applicant")
    rejected = (holdout["p_first"] >= CUTOFF).to_numpy()
    comparison = []
    for name, p in (("first model, calibrated", holdout["p_first"].to_numpy()),
                    ("second look, raw", p_hold_raw), ("second look, calibrated", p_hold_cal)):
        comparison.append({"model": name, "group": "all thin-file holdout", **thin_metrics(p, y_hold)})
        comparison.append({"model": name, "group": "thin-file rejected by the first model",
                           **thin_metrics(p[rejected], y_hold[rejected].reset_index(drop=True))})
    comparison = pd.DataFrame(comparison)
    print("\nSame thin-file applicants, both models:")
    print(comparison.round(4).to_string(index=False))

    stage1 = stage1.merge(holdout[["case_id", "p_default"]].rename(columns={"p_default": "p_second"}),
                          on="case_id", how="left")
    sweep = pd.DataFrame([two_stage(stage1, stage1["p_second"].to_numpy(), c) for c in (0.0, *SWEEP_CUTOFFS)])
    print("\nTwo stages on the whole holdout (second cutoff 0 = first model only):")
    print(sweep.round(4).to_string(index=False))

    importance = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    joblib.dump(
        {"model": model, "encoder": encoder, "edu_map": edu_map, "calibrator": calibrator,
         "settings": settings, "feature_columns": list(X_train.columns), "dropped_columns": dead},
        ARTIFACTS_DIR / "second_look_random_forest.joblib",
    )
    holdout["score"] = 1 - holdout["p_default"]   # chance they pay it back
    holdout["first_model_approved"] = holdout["p_first"] < CUTOFF
    holdout["second_look_approved"] = ~holdout["first_model_approved"] & (holdout["p_default"] < CUTOFF)
    holdout.to_csv(PREDICTIONS_DIR / "second_look_holdout.csv", index=False)

    p_score_raw = model.predict_proba(X_score)[:, 1]
    score = splits["score"][["case_id"]].copy()
    score["p_default_raw"] = p_score_raw
    score["p_default"] = calibrator.predict(p_score_raw)
    score["score"] = 1 - score["p_default"]
    first_scores = PREDICTIONS_DIR / "random_forest_score.csv"
    if first_scores.is_file():
        full_score = pd.read_csv(first_scores)
        score = score.merge(
            full_score[["case_id", "p_default"]].rename(columns={"p_default": "p_first"}),
            on="case_id",
            how="left",
        )
        score["first_model_approved"] = score["p_first"] < CUTOFF
        score["second_look_approved"] = ~score["first_model_approved"] & (score["p_default"] < CUTOFF)

        full_score = full_score.merge(
            score[["case_id", "p_default"]].rename(columns={"p_default": "p_second_look"}),
            on="case_id",
            how="left",
        )
        rejected_thin = (full_score[SEGMENT] == 1) & (full_score["p_default"] >= CUTOFF)
        full_score["p_final"] = np.where(
            rejected_thin, full_score["p_second_look"], full_score["p_default"]
        )
        full_score["score_final"] = 1 - full_score["p_final"]
        full_score["approved_final"] = (full_score["p_default"] < CUTOFF) | (
            rejected_thin & (full_score["p_second_look"] < CUTOFF)
        )
        full_score.to_csv(PREDICTIONS_DIR / "two_stage_combined_score.csv", index=False)
    score.to_csv(PREDICTIONS_DIR / "second_look_score.csv", index=False)

    results.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    comparison.to_csv(REPORTS_DIR / "comparison.csv", index=False)
    sweep.to_csv(REPORTS_DIR / "two_stage.csv", index=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    plot_reliability({"first model, calibrated": holdout["p_first"].to_numpy(), "second look, raw": p_hold_raw,
                      "second look, calibrated": p_hold_cal}, y_hold, REPORTS_DIR / "calibration.png")
    plot_two_stage(sweep, REPORTS_DIR / "two_stage.png")
    print(f"\n-> saved model to {ARTIFACTS_DIR}, predictions to {PREDICTIONS_DIR}, tables and plots to {REPORTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the second-look random forest for thin-file applicants.")
    parser.add_argument("--quick", action="store_true", help="skip the tuning loop")
    parser.add_argument("--stage1", type=Path, default=STAGE1_FILE,
                        help="holdout predictions of the first model (case_id, is_thin_file, target, p_default)")
    args = parser.parse_args()
    main(quick=args.quick, stage1_path=args.stage1)