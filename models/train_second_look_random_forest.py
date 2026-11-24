"""
train_second_look_random_forest.py

Second look for thin-file applicants that the logistic regression (stage 1) does not approve. They
get scored again by a random forest trained only on thin-file applicants and a second cutoff
decides. Applicants with a credit history only go through stage 1.

Run after features.py and train_random_forest.py:

    python models/train_second_look_random_forest.py          # about 13 minutes
    python models/train_second_look_random_forest.py --quick  # skip the tuning loop

Forest settings are picked on weeks 64-69 and the calibration is fit on 70-72. The second cutoff is
picked on 73-75 and the holdout is only used at the end. Tables and plots go to
reports/second_look_random_forest.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logistic_baseline import apply_calibrator, fit_calibrator
from predict_logistic import score_dataframe
from splits import SEGMENT, get_features_target_segment, make_splits
from train_random_forest import (
    ARTIFACTS_DIR, BOOL_COLS, CUTOFF, DEFAULT_SETTINGS, EXCLUDED_COLS, FEATURES_DIR, PREDICTIONS_DIR,
    PROFIT_DEFAULT, PROFIT_REPAID, ROOT_DIR, SWEEP_CUTOFFS, fit_forest, tune,
)

REPORTS_DIR = ROOT_DIR / "reports" / "second_look_random_forest"
LOGISTIC_FILE = ARTIFACTS_DIR / "logistic_pipeline.joblib"
ALL_APPLICANTS_FILE = PREDICTIONS_DIR / "random_forest_holdout.csv"
# description_5085714M is always MISSING for thin-file applicants so only these two
TEXT_COLS = ["education_927M", "maritalstatus_703M"]

SELECTION_END_WEEK = 69      # 64-69 forest settings
CALIBRATION_END_WEEK = 72    # 70-72 calibration and 73-75 second cutoff
CALIBRATION = "sigmoid"      # isotonic gave only a few dozen different scores here


# ------------- stage 1 --------------------------------
def stage1_decisions(df: pd.DataFrame, logistic: dict) -> pd.DataFrame:
    scored = score_dataframe(df, logistic)
    keep = [col for col in ("case_id", "WEEK_NUM", SEGMENT, "target") if col in df.columns]
    out = df[keep].reset_index(drop=True)
    out["p_first"] = scored["predicted_default_probability"].to_numpy()
    out["first_pass_approve"] = scored["first_pass_approve"].to_numpy().astype(bool)
    # about 7% of applicants land exactly on the stage 1 cutoff and rounding picks their side
    out["stage1_tie"] = np.isclose(out["p_first"], float(logistic["stage1_threshold"]), rtol=0, atol=1e-6)
    return out


def referred_thin_file(df: pd.DataFrame, stage1: pd.DataFrame) -> pd.DataFrame:
    ids = stage1.loc[(stage1[SEGMENT] == 1) & ~stage1["first_pass_approve"], "case_id"]
    return df[df["case_id"].isin(ids)].reset_index(drop=True)


def stage1_scores(referred: pd.DataFrame, stage1: pd.DataFrame) -> np.ndarray:
    merged = referred[["case_id"]].merge(stage1[["case_id", "p_first"]], on="case_id", validate="one_to_one")
    return merged["p_first"].to_numpy()


def forest_on_all_applicants(holdout: pd.DataFrame):
    if not ALL_APPLICANTS_FILE.is_file():
        print(f"\nno {ALL_APPLICANTS_FILE.name} so skipping the all-applicants rows (run train_random_forest.py first)")
        return None
    saved = pd.read_csv(ALL_APPLICANTS_FILE)[["case_id", "p_default"]]
    p = holdout[["case_id"]].merge(saved, on="case_id", how="left", validate="one_to_one")["p_default"]
    if p.isna().any():
        print(f"\n{ALL_APPLICANTS_FILE.name} does not match these holdout rows so skipping the all-applicants rows "
              "(rerun train_random_forest.py)")
        return None
    return p.to_numpy()


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


# ------------- calibration --------------------------------
def calibrate(p_raw: np.ndarray, calibrator) -> np.ndarray:
    return apply_calibrator(CALIBRATION, calibrator, p_raw)


# ------------- evaluation --------------------------------
def model_metrics(p: np.ndarray, y: pd.Series) -> dict:
    return {
        "rows": int(len(p)),
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "default_rate": float(y.mean()),
        "mean_predicted_default": float(p.mean()),
        "calibration_gap": float(p.mean() - y.mean()),   # positive = over-predicts risk
    }


def add_second_look(stage1: pd.DataFrame, referred: pd.DataFrame, p_raw: np.ndarray,
                    p_calibrated: np.ndarray) -> pd.DataFrame:
    scored = pd.DataFrame({
        "case_id": referred["case_id"].to_numpy(),
        "p_second_raw": p_raw,
        "p_second": p_calibrated,
    })
    return stage1.merge(scored, on="case_id", how="left", validate="one_to_one")


def two_stage(table: pd.DataFrame, second_cutoff: float) -> dict:
    y = table["target"].to_numpy()
    thin = (table[SEGMENT] == 1).to_numpy()
    first = table["first_pass_approve"].to_numpy()
    # p_second is NaN for anyone the second look did not score so they stay out
    rescued = thin & ~first & (table["p_second"].to_numpy() < second_cutoff)
    approved = first | rescued
    repaid = y == 0
    established_ok = approved[repaid & ~thin].mean()
    thin_ok = approved[repaid & thin].mean()
    return {
        "second_cutoff": second_cutoff,
        "rows": int(len(table)),
        "approved_count": int(approved.sum()),
        "approval_rate": float(approved.mean()),
        "default_rate_of_approved": float(y[approved].mean()),
        "expected_profit": float(np.where(y[approved] == 1, PROFIT_DEFAULT, PROFIT_REPAID).sum()),
        "established_repayers_approved": float(established_ok),
        "thin_file_repayers_approved": float(thin_ok),
        "inclusion_gap": float(established_ok - thin_ok),
        "rescued": int(rescued.sum()),
        "rescued_default_rate": float(y[rescued].mean()) if rescued.any() else np.nan,
        "rescued_from_stage1_tie": int((rescued & table["stage1_tie"].to_numpy()).sum()),
    }


def cutoff_sweep(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([two_stage(table, cutoff) for cutoff in (0.0, *SWEEP_CUTOFFS)])   # 0 = stage 1 only


def pick_cutoff(sweep: pd.DataFrame, max_default_rate: float) -> float:
    allowed = sweep[sweep["default_rate_of_approved"] <= max_default_rate]
    if allowed.empty:
        return 0.0
    best = allowed.sort_values(["expected_profit", "inclusion_gap", "second_cutoff"],
                               ascending=[False, True, True]).iloc[0]
    return float(best["second_cutoff"])


def final_table(table: pd.DataFrame, second_cutoff: float) -> pd.DataFrame:
    out = table.rename(columns={"p_first": "p_default", "p_second_raw": "p_second_look_raw",
                                "p_second": "p_second_look"})
    out["decided_by"] = np.where(out["p_second_look"].notna(), "second look", "stage 1")
    out["p_final"] = out["p_second_look"].fillna(out["p_default"])
    out["score_final"] = 1 - out["p_final"]   # chance they pay it back
    # p_final comes from two models with different cutoffs so use approved_final for decisions
    out["approved_final"] = out["first_pass_approve"] | (out["p_second_look"] < second_cutoff)
    return out


def plot_reliability(curves: dict, y: pd.Series, path: str) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5))
    for label, p in curves.items():
        actual, predicted = calibration_curve(y, p, n_bins=15, strategy="quantile")
        axis.plot(predicted, actual, marker="o", ms=3, label=label)
    axis.plot([0, 0.7], [0, 0.7], linestyle="--", color="gray")
    axis.axvline(CUTOFF, linestyle=":", color="gray", label=f"{CUTOFF:.0%} break-even")
    axis.set(title="Thin-file applicants stage 1 referred, weeks 76-91",
             xlabel="predicted default probability", ylabel="actual default rate",
             xlim=(0, 0.7), ylim=(0, 0.7))
    axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_two_stage(sweep: pd.DataFrame, picked: float, path: str) -> None:
    first_only = sweep[sweep["second_cutoff"] == 0].iloc[0]
    curve = sweep[sweep["second_cutoff"] > 0]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("expected_profit", "profit on holdout ($M)", 1e-6),
        ("inclusion_gap", "inclusion gap", 1),
        ("default_rate_of_approved", "default rate of approved loans", 1),
    ]
    for axis, (col, ylabel, scale) in zip(axes, panels):
        axis.plot(curve["second_cutoff"], curve[col] * scale, marker="o", ms=3, label="stage 1 + second look")
        axis.axhline(first_only[col] * scale, color="grey", linestyle=":", linewidth=1, label="stage 1 only")
        if picked > 0:
            axis.axvline(picked, color="grey", linestyle="--", linewidth=0.8, label="cutoff picked on weeks 73-75")
        axis.set(xlabel="second cutoff for referred thin-file applicants", ylabel=ylabel)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ------------- main --------------------------------
def main(quick: bool, max_default_rate: float | None) -> None:
    for path in (FEATURES_DIR / "train_features.csv", FEATURES_DIR / "test_features.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run notebooks/features.py first.")
    if not LOGISTIC_FILE.is_file():
        raise FileNotFoundError(f"Missing {LOGISTIC_FILE}. Get it back from git.")
    for folder in (ARTIFACTS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(FEATURES_DIR / "train_features.csv")
    test_df = pd.read_csv(FEATURES_DIR / "test_features.csv")
    splits = make_splits(train_df, test_df)

    logistic = joblib.load(LOGISTIC_FILE)
    if max_default_rate is None:
        max_default_rate = float(logistic["policy_assumptions"]["max_approved_default_rate"])
    limit = f"{max_default_rate * 100:g}%"
    stage1 = {name: stage1_decisions(splits[name], logistic) for name in ("tune", "holdout", "score")}
    p_all_applicants = forest_on_all_applicants(stage1["holdout"])

    train = thin_file_rows(splits["train"])
    tune_referred = referred_thin_file(splits["tune"], stage1["tune"])
    week = tune_referred["WEEK_NUM"]
    selection_referred = tune_referred[week <= SELECTION_END_WEEK]
    calibration_referred = tune_referred[(week > SELECTION_END_WEEK) & (week <= CALIBRATION_END_WEEK)]
    cutoff_referred = tune_referred[week > CALIBRATION_END_WEEK]
    hold_referred = referred_thin_file(splits["holdout"], stage1["holdout"])
    score_referred = referred_thin_file(splits["score"], stage1["score"])
    stage1_approved = {name: int(table["first_pass_approve"].sum()) for name, table in stage1.items()}
    print(f"\nStage 1 approves under {logistic['stage1_threshold']:.2%}: "
          f"{stage1_approved['holdout']:,} of {len(stage1['holdout']):,} holdout applicants in this run")
    print(f"Thin-file applicants it referred: {len(selection_referred):,} in weeks 64-69, "
          f"{len(calibration_referred):,} in 70-72, {len(cutoff_referred):,} in 73-75, "
          f"{len(hold_referred):,} in the holdout, {len(score_referred):,} in the test file")
    print(f"Thin-file applicants in the train weeks: {len(train):,}")

    X_train, y_train, _ = get_features_target_segment(train)
    X_selection, y_selection, _ = get_features_target_segment(selection_referred)
    X_calibration, y_calibration, _ = get_features_target_segment(calibration_referred)
    X_cutoff, y_cutoff, _ = get_features_target_segment(cutoff_referred)
    X_hold, _, _ = get_features_target_segment(hold_referred)
    X_score, _, _ = get_features_target_segment(score_referred)

    dead = dead_columns(X_train)
    print(f"\n{len(dead)} columns never vary for thin-file applicants and are dropped:")
    print("  " + ", ".join(dead))
    encoder = fit_encoder(X_train)
    edu_map = fit_education_order(X_train, y_train)
    X_train, X_selection, X_calibration, X_cutoff, X_hold, X_score = (
        prep(X, encoder, dead, edu_map) for X in (X_train, X_selection, X_calibration, X_cutoff, X_hold, X_score)
    )
    print(f"\nFeatures going into the second-look forest: {X_train.shape[1]}")

    print("\nFitting with default settings...")
    model = fit_forest(X_train, y_train, DEFAULT_SETTINGS)
    p_selection = model.predict_proba(X_selection)[:, 1]
    print(f"default settings  AUC on weeks 64-69 {roc_auc_score(y_selection, p_selection):.4f}")

    settings = DEFAULT_SETTINGS
    if not quick:
        print("\nTuning on a sample of train and scoring on the referred applicants of weeks 64-69 (the tune AUC below)...")
        best = tune(X_train, y_train, X_selection, y_selection)
        best_model = fit_forest(X_train, y_train, best)
        p_best = best_model.predict_proba(X_selection)[:, 1]
        print(f"best settings on all of train  AUC on weeks 64-69 {roc_auc_score(y_selection, p_best):.4f}")
        if roc_auc_score(y_selection, p_best) > roc_auc_score(y_selection, p_selection):
            model, p_selection, settings = best_model, p_best, best
        else:
            print("tuned settings did not beat the defaults so keeping the defaults")

    # fitted on weeks 70-72 only
    calibrator = fit_calibrator(CALIBRATION, model.predict_proba(X_calibration)[:, 1], y_calibration)

    # the limit counts all approved loans so this uses everyone in weeks 73-75
    p_cutoff_raw = model.predict_proba(X_cutoff)[:, 1]
    p_cutoff = calibrate(p_cutoff_raw, calibrator)
    cutoff_weeks = stage1["tune"][stage1["tune"]["WEEK_NUM"] > CALIBRATION_END_WEEK]
    cutoff_weeks = add_second_look(cutoff_weeks, cutoff_referred, p_cutoff_raw, p_cutoff)
    policy_thresholds = cutoff_sweep(cutoff_weeks)
    second_cutoff = pick_cutoff(policy_thresholds, max_default_rate)
    profit_cutoff = pick_cutoff(policy_thresholds, 1.0)   # limit of 1 = no limit
    policy_thresholds["meets_default_rate_limit"] = policy_thresholds["default_rate_of_approved"] <= max_default_rate
    policy_thresholds["selected"] = policy_thresholds["second_cutoff"] == second_cutoff
    print(f"\nWeeks 73-75 at each second cutoff (limit {limit}, second cutoff 0 = stage 1 only):")
    print(policy_thresholds.round(4).to_string(index=False))
    picked = policy_thresholds[policy_thresholds["selected"]].iloc[0]
    print(f"stage 1 alone is at {policy_thresholds['default_rate_of_approved'].iloc[0]:.2%} on these weeks")
    print(f"picked second cutoff: {second_cutoff:.2f} ({int(picked['rescued']):,} of the {len(cutoff_referred):,} "
          f"referred get approved). Most profitable without the limit: {profit_cutoff:.2f}")
    if not policy_thresholds["meets_default_rate_limit"].any():
        print(f"even stage 1 alone is over the {limit} limit on these weeks so the second look approves nobody")

    # only scoring the holdout from here on
    p_hold_raw = model.predict_proba(X_hold)[:, 1]
    holdout = add_second_look(stage1["holdout"], hold_referred, p_hold_raw, calibrate(p_hold_raw, calibrator))
    hold_scored = holdout[holdout["p_second"].notna()]
    y_hold = hold_scored["target"]

    results = pd.DataFrame([
        {"weeks": "64-69", "picked_here": "forest settings", "version": "raw",
         **model_metrics(p_selection, y_selection)},
        {"weeks": "73-75", "picked_here": "second cutoff", "version": "raw",
         **model_metrics(p_cutoff_raw, y_cutoff)},
        {"weeks": "73-75", "picked_here": "second cutoff", "version": "calibrated",
         **model_metrics(p_cutoff, y_cutoff)},
        {"weeks": "76-91 holdout", "picked_here": "nothing", "version": "raw",
         **model_metrics(hold_scored["p_second_raw"].to_numpy(), y_hold)},
        {"weeks": "76-91 holdout", "picked_here": "nothing", "version": "calibrated",
         **model_metrics(hold_scored["p_second"].to_numpy(), y_hold)},
    ])
    print("\nSettings used:", settings)
    print("Second look on the thin-file applicants stage 1 referred:")
    print(results.round(4).to_string(index=False))

    scorers = {"logistic regression (stage 1), calibrated": hold_scored["p_first"].to_numpy()}
    if p_all_applicants is not None:
        scorers["random forest trained on all applicants, calibrated"] = p_all_applicants[hold_scored.index]
    scorers["second look, raw"] = hold_scored["p_second_raw"].to_numpy()
    scorers["second look, calibrated"] = hold_scored["p_second"].to_numpy()
    comparison = pd.DataFrame([{"model": name, **model_metrics(p, y_hold)} for name, p in scorers.items()])
    # only the auc column is a fair comparison here
    comparison = comparison[["model", "rows", "auc", "default_rate", "mean_predicted_default", "calibration_gap"]]
    print("\nReferred thin-file holdout applicants, every model:")
    print(comparison.round(4).to_string(index=False))

    # same rule but with stage 1's own score instead of the forest
    control = fit_calibrator(CALIBRATION, stage1_scores(calibration_referred, stage1["tune"]), y_calibration)
    p_control = calibrate(stage1_scores(hold_referred, stage1["holdout"]), control)
    no_second_model = add_second_look(stage1["holdout"], hold_referred, p_control, p_control)

    two_stages = [
        {"policy": "stage 1 only", **two_stage(holdout, 0.0)},
        {"policy": f"second look, cutoff from weeks 73-75 with the {limit} limit", **two_stage(holdout, second_cutoff)},
        {"policy": f"second look, {CUTOFF:.0%} cutoff", **two_stage(holdout, CUTOFF)},
        {"policy": f"no second model, stage 1 score recalibrated for the referred, {CUTOFF:.0%} cutoff",
         **two_stage(no_second_model, CUTOFF)},
    ]
    if p_all_applicants is not None:
        alone = holdout.assign(first_pass_approve=p_all_applicants < CUTOFF)
        two_stages.append({"policy": f"random forest trained on all applicants alone, {CUTOFF:.0%} cutoff",
                           **two_stage(alone, 0.0)})
    two_stages = pd.DataFrame(two_stages)
    two_stages["meets_default_rate_limit"] = two_stages["default_rate_of_approved"] <= max_default_rate
    print("\nTwo stages on the whole holdout:")
    print(two_stages.round(4).to_string(index=False))
    holdout_sweep = cutoff_sweep(holdout)

    final_holdout = final_table(holdout, second_cutoff)
    scores = {"stage 1 only": final_holdout["p_default"].to_numpy(),
              "final, second look where it applies": final_holdout["p_final"].to_numpy()}
    if p_all_applicants is not None:
        scores["random forest trained on all applicants"] = p_all_applicants
    final_scores = pd.DataFrame([{"score": name, **model_metrics(p, final_holdout["target"])}
                                 for name, p in scores.items()])
    print("\nScores on the whole holdout:")
    print(final_scores.round(4).to_string(index=False))

    importance = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    joblib.dump(
        {"model": model, "encoder": encoder, "edu_map": edu_map, "calibration_method": CALIBRATION,
         "calibrator": calibrator, "settings": settings, "feature_columns": list(X_train.columns),
         "dropped_columns": dead, "second_look_cutoff": second_cutoff,
         "stage1_threshold": float(logistic["stage1_threshold"])},
        ARTIFACTS_DIR / "second_look_random_forest.joblib",
    )
    final_holdout.to_csv(PREDICTIONS_DIR / "two_stage_random_forest_holdout.csv", index=False)
    p_score_raw = model.predict_proba(X_score)[:, 1]
    score = add_second_look(stage1["score"], score_referred, p_score_raw, calibrate(p_score_raw, calibrator))
    final_table(score, second_cutoff).to_csv(PREDICTIONS_DIR / "two_stage_random_forest_score.csv", index=False)

    config = {
        "stage1_threshold": float(logistic["stage1_threshold"]),
        "stage1_approved": stage1_approved,
        "weeks": {"train": "0-63", "settings": "64-69", "calibration": "70-72",
                  "second_cutoff": "73-75", "holdout": "76-91"},
        "train_rows": len(train),
        "referred_rows": {"settings": len(selection_referred), "calibration": len(calibration_referred),
                          "second_cutoff": len(cutoff_referred), "holdout": len(hold_referred),
                          "score": len(score_referred)},
        "quick": quick,
        "settings": settings,
        "calibration_method": CALIBRATION,
        "max_approved_default_rate": max_default_rate,
        "selected_second_cutoff": second_cutoff,
        "most_profitable_second_cutoff": profit_cutoff,
        "all_applicants_rows": p_all_applicants is not None,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "scipy": scipy.__version__},
    }
    (REPORTS_DIR / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    results.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    comparison.to_csv(REPORTS_DIR / "comparison.csv", index=False)
    policy_thresholds.to_csv(REPORTS_DIR / "policy_thresholds.csv", index=False)
    two_stages.to_csv(REPORTS_DIR / "two_stage.csv", index=False)
    final_scores.to_csv(REPORTS_DIR / "final_score.csv", index=False)
    holdout_sweep.to_csv(REPORTS_DIR / "cutoff_sweep.csv", index=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    plot_reliability(scorers, y_hold, REPORTS_DIR / "calibration.png")
    plot_two_stage(holdout_sweep, second_cutoff, REPORTS_DIR / "two_stage.png")
    print(f"\n-> saved model to {ARTIFACTS_DIR}, predictions to {PREDICTIONS_DIR}, tables and plots to {REPORTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the random forest second look.")
    parser.add_argument("--quick", action="store_true", help="skip the tuning loop")
    parser.add_argument("--max-default-rate", type=float, default=None,
                        help="max default rate of all approved loans when picking the second cutoff "
                             "(default: stage 1's limit)")
    args = parser.parse_args()
    if args.max_default_rate is not None and not 0 < args.max_default_rate <= 1:
        parser.error("--max-default-rate must be between 0 and 1")
    main(quick=args.quick, max_default_rate=args.max_default_rate)
