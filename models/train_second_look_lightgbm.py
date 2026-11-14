"""Train and evaluate LightGBM as Stage 2 for thin-file applicants.

Stage 1 Logistic Regression scores all applicants. This script trains LightGBM
only on thin-file applicants that time-forward Stage 1 models would refer, then
scores the thin-file applicants actually referred in holdout and test data.
It uses weeks 64-67 for early stopping, 68-69 for parameter selection, 70-72
for probability calibration, and 73-75 to select the second-look cutoff under
the approved-pool default-rate limit. Weeks 76-91 are held out for final
evaluation. The selected model, predictions, metrics, and run configuration
are saved under models/artifacts, data/predictions, and reports/second_look_lightgbm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "models"))

from splits import SEGMENT, get_features_target_segment, make_splits
from logistic_baseline import build_preprocessor, clean_X, fit_calibrator
from notebooks.train_lightgbm import DEFAULT_PARAMS, apply_prep, fit_prep, train_lgbm


FEATURES_DIR = ROOT_DIR / "data" / "features"
PREDICTIONS_DIR = ROOT_DIR / "data" / "predictions"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports" / "second_look_lightgbm"
STAGE1_PATH = PREDICTIONS_DIR / "logistic_labeled_predictions.csv"
STAGE1_SCORE_PATH = PREDICTIONS_DIR / "logistic_score_predictions.csv"
LOGISTIC_ARTIFACT_PATH = ARTIFACTS_DIR / "logistic_pipeline.joblib"
SECOND_LOOK_CUTOFF = 0.20
REPAID_VALUE = 2_000
DEFAULT_COST = -8_000
EXCLUDED_FEATURES = [SEGMENT, "external_coverage"]
EARLY_STOP_END_WEEK = 69
MODEL_SELECTION_START_WEEK = 68
CALIBRATION_END_WEEK = 72
TRAIN_ROUTE_WARMUP_END_WEEK = 15
TRAIN_ROUTE_CALIBRATION_WEEKS = 8
TRAIN_ROUTE_FOLD_WEEKS = 8


def load_stage1_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    if "p_default" not in predictions and "predicted_default_probability" in predictions:
        predictions = predictions.rename(
            columns={"predicted_default_probability": "p_default"}
        )
    required = ["case_id", "p_default", "first_pass_approve"]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if predictions["case_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate case_id values")
    predictions["first_pass_approve"] = predictions["first_pass_approve"].astype(bool)
    optional = [column for column in (SEGMENT,) if column in predictions]
    return predictions[required + optional]


def attach_stage1_routes(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Join first-stage routes to a labeled time split and verify complete coverage."""
    route_columns = ["case_id", "p_default", "first_pass_approve"]
    routed = frame.merge(
        predictions[route_columns],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    if routed["first_pass_approve"].isna().any():
        raise ValueError("Stage 1 predictions do not cover every requested applicant")
    routed["first_pass_approve"] = routed["first_pass_approve"].astype(bool)
    return routed


def make_time_forward_train_routes(
    train_df: pd.DataFrame,
    logistic_artifact: dict,
) -> pd.DataFrame:
    """Generate out-of-time Stage 1 routes for Stage 2 training rows."""
    threshold = float(logistic_artifact["stage1_threshold"])
    regularization = float(logistic_artifact["model"].C)
    routed_folds = []
    first_score_week = TRAIN_ROUTE_WARMUP_END_WEEK + TRAIN_ROUTE_CALIBRATION_WEEKS + 1
    last_train_week = int(train_df["WEEK_NUM"].max())

    for score_start in range(first_score_week, last_train_week + 1, TRAIN_ROUTE_FOLD_WEEKS):
        score_end = min(score_start + TRAIN_ROUTE_FOLD_WEEKS - 1, last_train_week)
        calibration_start = score_start - TRAIN_ROUTE_CALIBRATION_WEEKS
        historical = train_df.loc[train_df["WEEK_NUM"] < calibration_start]
        calibration_frame = train_df.loc[
            train_df["WEEK_NUM"].between(calibration_start, score_start - 1)
        ]
        score_frame = train_df.loc[
            train_df["WEEK_NUM"].between(score_start, score_end)
        ]
        X_historical, y_historical, _ = get_features_target_segment(historical)
        X_calibration, y_calibration, _ = get_features_target_segment(calibration_frame)
        X_score, _, _ = get_features_target_segment(score_frame)
        X_historical = clean_X(X_historical)
        X_calibration = clean_X(X_calibration)
        X_score = clean_X(X_score)

        preprocessor, _, _ = build_preprocessor(X_historical)
        historical_matrix = np.asarray(
            preprocessor.fit_transform(X_historical), dtype=np.float32
        )
        calibration_matrix = np.asarray(
            preprocessor.transform(X_calibration), dtype=np.float32
        )
        score_matrix = np.asarray(preprocessor.transform(X_score), dtype=np.float32)
        model = LogisticRegression(
            C=regularization,
            solver="lbfgs",
            max_iter=1_000,
            random_state=42,
        )
        model.fit(historical_matrix, y_historical)
        calibration_raw = model.predict_proba(calibration_matrix)[:, 1]
        raw_probability = model.predict_proba(score_matrix)[:, 1]
        calibrator = fit_calibrator(
            "isotonic",
            calibration_raw,
            y_calibration,
        )
        probability = calibrator.predict(raw_probability)
        fold_routes = score_frame[["case_id", "WEEK_NUM", SEGMENT, "target"]].copy()
        fold_routes["p_stage1"] = probability
        fold_routes["first_pass_approve"] = probability < threshold
        routed_folds.append(fold_routes)
        print(
            f"Stage 1 time-forward routes weeks {score_start}-{score_end}: "
            f"fit through week {int(historical['WEEK_NUM'].max())}, "
            f"calibrated on weeks {calibration_start}-{score_start - 1}, "
            f"scored {len(score_frame):,} applicants"
        )

    if not routed_folds:
        raise ValueError("Training data is too short to create time-forward Stage 1 routes")
    return pd.concat(routed_folds, ignore_index=True)


def load_stage1_holdout(path: Path, holdout: pd.DataFrame) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    if "p_default" not in predictions and "predicted_default_probability" in predictions:
        predictions = predictions.rename(
            columns={"predicted_default_probability": "p_default"}
        )
    required = ["case_id", "p_default", "first_pass_approve"]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    holdout_metadata = holdout[["case_id", "target", SEGMENT]]
    routed = predictions[required].merge(
        holdout_metadata,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(routed) != len(holdout_metadata):
        raise ValueError(f"{path} does not cover every holdout applicant")
    routed["first_pass_approve"] = routed["first_pass_approve"].astype(bool)
    return routed


def model_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features, target, _ = get_features_target_segment(frame)
    features = features.drop(columns=EXCLUDED_FEATURES, errors="ignore")
    return features, target


def metrics_for_group(
    target: pd.Series,
    probabilities: np.ndarray,
    group: str,
    version: str,
    cutoff: float,
) -> dict:
    approved = probabilities < cutoff
    return {
        "group": group,
        "version": version,
        "rows": len(target),
        "default_rate": float(target.mean()),
        "mean_predicted_default": float(probabilities.mean()),
        "roc_auc": roc_auc_score(target, probabilities),
        "pr_auc": average_precision_score(target, probabilities),
        "logloss": log_loss(target, probabilities, labels=[0, 1]),
        "brier": brier_score_loss(target, probabilities),
        "second_look_approval_rate": float(approved.mean()),
        "default_rate_of_second_look_approvals": (
            float(target[approved].mean()) if approved.any() else np.nan
        ),
    }


def two_stage_metrics(
    stage1: pd.DataFrame,
    second_probabilities: pd.Series,
    second_cutoff: float,
) -> dict:
    frame = stage1.merge(
        second_probabilities.rename("p_second"),
        left_on="case_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    thin = frame[SEGMENT].eq(1).to_numpy()
    first_approved = frame["first_pass_approve"].to_numpy()
    rescued = thin & ~first_approved & (frame["p_second"].to_numpy() < second_cutoff)
    approved = first_approved | rescued
    target = frame["target"].to_numpy()
    segment = frame[SEGMENT].to_numpy()
    repaid = target == 0
    established_repaid = repaid & (segment == 0)
    thin_repaid = repaid & (segment == 1)
    approved_repaid = approved & repaid
    approved_default = approved & (target == 1)
    established_approval = float(approved[established_repaid].mean())
    thin_approval = float(approved[thin_repaid].mean())
    return {
        "second_cutoff": second_cutoff,
        "rows": len(frame),
        "approved_count": int(approved.sum()),
        "approval_rate": float(approved.mean()),
        "default_rate_of_approved": float(target[approved].mean()) if approved.any() else np.nan,
        "expected_profit": float(
            approved_repaid.sum() * REPAID_VALUE + approved_default.sum() * DEFAULT_COST
        ),
        "established_repayers_approved": established_approval,
        "thin_file_repayers_approved": thin_approval,
        "inclusion_gap": established_approval - thin_approval,
        "rescued": int(rescued.sum()),
        "rescued_default_rate": float(target[rescued].mean()) if rescued.any() else np.nan,
    }


def run(
    stage1_path: Path = STAGE1_PATH,
    second_cutoff: float = SECOND_LOOK_CUTOFF,
    logistic_artifact_path: Path = LOGISTIC_ARTIFACT_PATH,
    max_approved_default_rate: float = 0.10,
) -> None:
    train_df = pd.read_csv(FEATURES_DIR / "train_features.csv", low_memory=False)
    test_df = pd.read_csv(FEATURES_DIR / "test_features.csv", low_memory=False)
    splits = make_splits(train_df, test_df)
    if not logistic_artifact_path.is_file():
        raise FileNotFoundError(
            f"Missing {logistic_artifact_path}. Run models/logistic_baseline.py first."
        )
    if not STAGE1_SCORE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {STAGE1_SCORE_PATH}. Score the unlabeled test split with the logistic model first."
        )

    logistic_artifact = joblib.load(logistic_artifact_path)
    stage1_predictions = load_stage1_predictions(stage1_path)
    train_routes = make_time_forward_train_routes(splits["train"], logistic_artifact)
    train_referrals = train_routes.loc[
        (train_routes[SEGMENT] == 1) & ~train_routes["first_pass_approve"]
    ]
    train_ids = train_referrals["case_id"]
    train_referral_frame = splits["train"].loc[
        splits["train"]["case_id"].isin(train_ids)
    ].copy()

    tune_routes = attach_stage1_routes(splits["tune"], stage1_predictions)
    early_stop_frame = tune_routes.loc[tune_routes["WEEK_NUM"] < MODEL_SELECTION_START_WEEK]
    model_selection_frame = tune_routes.loc[
        tune_routes["WEEK_NUM"].between(MODEL_SELECTION_START_WEEK, EARLY_STOP_END_WEEK)
    ]
    calibration_frame = tune_routes.loc[
        (tune_routes["WEEK_NUM"] > EARLY_STOP_END_WEEK)
        & (tune_routes["WEEK_NUM"] <= CALIBRATION_END_WEEK)
    ]
    policy_frame = tune_routes.loc[tune_routes["WEEK_NUM"] > CALIBRATION_END_WEEK]

    def referred_thin(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[(frame[SEGMENT] == 1) & ~frame["first_pass_approve"]].copy()

    early_stop_referrals = referred_thin(early_stop_frame)
    model_selection_referrals = referred_thin(model_selection_frame)
    calibration_referrals = referred_thin(calibration_frame)
    policy_referrals = referred_thin(policy_frame)
    if not len(train_referral_frame) or not len(early_stop_referrals):
        raise ValueError("No routed thin-file applicants are available for model fitting")
    if not len(model_selection_referrals) or model_selection_referrals["target"].nunique() < 2:
        raise ValueError("Candidate-selection weeks need both repayment and default referrals")
    if calibration_referrals["target"].nunique() < 2:
        raise ValueError("Calibration weeks need both repayment and default outcomes")
    if policy_referrals["target"].nunique() < 2:
        raise ValueError("Policy-validation weeks need both repayment and default outcomes")

    X_train, y_train = model_data(train_referral_frame)
    X_early, y_early = model_data(early_stop_referrals)
    X_selection, y_selection = model_data(model_selection_referrals)
    X_calibration, y_calibration = model_data(calibration_referrals)
    prep_rules = fit_prep(X_train)
    X_train = apply_prep(X_train, prep_rules)
    X_early = apply_prep(X_early, prep_rules)
    X_selection = apply_prep(X_selection, prep_rules)
    X_calibration = apply_prep(X_calibration, prep_rules)

    candidates = (
        {},
        {"num_leaves": 15, "min_child_samples": 100},
        {"num_leaves": 31, "min_child_samples": 100, "lambda_l2": 5.0},
        {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 100},
    )
    tuning_rows = []
    best_model = None
    best_params = None
    best_loss = float("inf")
    print(
        "Routed thin-file applicants: "
        f"train={len(y_train):,}, early-stop={len(y_early):,}, "
        f"candidate-selection={len(y_selection):,}, "
        f"calibration={len(calibration_referrals):,}, "
        f"policy-validation={len(policy_referrals):,}"
    )
    for candidate_id, candidate in enumerate(candidates):
        candidate_model = train_lgbm(
            X_train,
            y_train,
            X_early,
            y_early,
            params=candidate,
            num_boost_round=3000,
            early_stopping=100,
            log_every=0,
        )
        selection_probability = candidate_model.predict(
            X_selection,
            num_iteration=candidate_model.best_iteration,
        )
        candidate_loss = log_loss(y_selection, selection_probability, labels=[0, 1])
        tuning_rows.append(
            {
                "candidate": candidate_id,
                "params": repr(candidate),
                "best_iteration": candidate_model.best_iteration,
                "selection_logloss": candidate_loss,
                "selection_roc_auc": roc_auc_score(y_selection, selection_probability),
            }
        )
        print(
            f"Candidate {candidate_id}: best_iteration={candidate_model.best_iteration}, "
            f"candidate-selection LogLoss={candidate_loss:.5f}"
        )
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_params = candidate.copy()
            best_model = candidate_model
    if best_model is None or best_params is None:
        raise RuntimeError("LightGBM tuning did not select a model")

    final_training_frame = pd.concat(
        [train_referral_frame, early_stop_referrals, model_selection_referrals],
        ignore_index=True,
    )
    X_final_train, y_final_train = model_data(final_training_frame)
    X_final_train = apply_prep(X_final_train, prep_rules)
    final_model = lgb.train(
        {**DEFAULT_PARAMS, **best_params},
        lgb.Dataset(X_final_train, y_final_train),
        num_boost_round=max(1, int(best_model.best_iteration)),
    )
    final_iteration = final_model.current_iteration()
    p_calibration_raw = final_model.predict(
        X_calibration,
        num_iteration=final_iteration,
    )
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        p_calibration_raw,
        y_calibration,
    )

    policy_routes = attach_stage1_routes(policy_frame.drop(columns=[
        "p_default", "first_pass_approve"
    ], errors="ignore"), stage1_predictions)
    policy_referrals = referred_thin(policy_routes)
    X_policy, y_policy = model_data(policy_referrals)
    X_policy = apply_prep(X_policy, prep_rules)
    p_policy_raw = final_model.predict(X_policy, num_iteration=final_iteration)
    p_policy = calibrator.predict(p_policy_raw)
    policy_probabilities = pd.Series(
        p_policy,
        index=policy_referrals["case_id"].to_numpy(),
        name="p_second",
    )
    policy_threshold_rows = [
        {
            "policy": "stage1_only",
            **two_stage_metrics(policy_routes, policy_probabilities, 0.0),
        },
        *[
        {
            "policy": f"second_cutoff_{cutoff:.2f}",
            **two_stage_metrics(policy_routes, policy_probabilities, cutoff),
        }
        for cutoff in np.round(np.arange(0.08, 0.401, 0.02), 2)
        ],
    ]
    policy_thresholds = pd.DataFrame(policy_threshold_rows)
    feasible = policy_thresholds.loc[
        policy_thresholds["default_rate_of_approved"] <= max_approved_default_rate
    ]
    if feasible.empty:
        selected_cutoff = 0.0
        selected_policy_row = policy_thresholds.loc[
            policy_thresholds["second_cutoff"] == 0.0
        ].iloc[0]
    else:
        selected_policy_row = feasible.sort_values(
            ["expected_profit", "inclusion_gap", "rescued"],
            ascending=[False, True, False],
        ).iloc[0]
        selected_cutoff = float(selected_policy_row["second_cutoff"])
    policy_thresholds["approved_default_rate_limit"] = max_approved_default_rate
    policy_thresholds["meets_default_rate_limit"] = (
        policy_thresholds["default_rate_of_approved"] <= max_approved_default_rate
    )
    policy_thresholds["selected"] = np.isclose(
        policy_thresholds["second_cutoff"], selected_cutoff
    )

    holdout_routes = attach_stage1_routes(splits["holdout"], stage1_predictions)
    holdout_referrals = referred_thin(holdout_routes)
    X_hold, y_hold = model_data(holdout_referrals)
    X_hold = apply_prep(X_hold, prep_rules)
    p_hold_raw = final_model.predict(X_hold, num_iteration=final_iteration)
    p_hold = calibrator.predict(p_hold_raw)
    metric_rows = [
        metrics_for_group(
            y_hold.reset_index(drop=True),
            p_hold_raw,
            "stage1_rejected_thin_file_holdout",
            "raw",
            second_cutoff,
        ),
        metrics_for_group(
            y_hold.reset_index(drop=True),
            p_hold,
            "stage1_rejected_thin_file_holdout",
            "calibrated",
            second_cutoff,
        ),
    ]
    metrics = pd.DataFrame(metric_rows)
    holdout_probabilities = pd.Series(
        p_hold,
        index=holdout_referrals["case_id"].to_numpy(),
        name="p_second",
    )
    base_policy = two_stage_metrics(holdout_routes, holdout_probabilities, 0.0)
    second_policy = two_stage_metrics(
        holdout_routes,
        holdout_probabilities,
        selected_cutoff,
    )
    reference_policy = two_stage_metrics(
        holdout_routes,
        holdout_probabilities,
        second_cutoff,
    )
    two_stage = pd.DataFrame(
        [
            {"policy": "logistic_stage1_only", **base_policy},
            {
                "policy": "logistic_plus_lightgbm_selected_validation_cutoff",
                **second_policy,
            },
            {
                "policy": f"logistic_plus_lightgbm_{second_cutoff:.0%}_reference",
                **reference_policy,
            },
        ]
    )

    score_stage1 = load_stage1_predictions(STAGE1_SCORE_PATH)
    score_routes = attach_stage1_routes(splits["score"], score_stage1)
    score_referrals = referred_thin(score_routes)
    X_score, _ = model_data(score_referrals)
    X_score = apply_prep(X_score, prep_rules)
    p_score_raw = final_model.predict(X_score, num_iteration=final_iteration)
    score_predictions = score_referrals[["case_id"]].copy()
    score_predictions["p_default_raw"] = p_score_raw
    score_predictions["p_default"] = calibrator.predict(p_score_raw)
    score_predictions["score"] = 1 - score_predictions["p_default"]
    if score_predictions["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id values in second-look score predictions")

    combined_score_path = PREDICTIONS_DIR / "two_stage_lightgbm_score.csv"
    combined_score = score_stage1.merge(
        score_predictions[["case_id", "p_default"]].rename(
            columns={"p_default": "p_second_look"}
        ),
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    referred_test = (combined_score[SEGMENT] == 1) & ~combined_score[
        "first_pass_approve"
    ].astype(bool)
    if combined_score.loc[referred_test, "p_second_look"].isna().any():
        raise ValueError("LightGBM did not score every referred thin-file test applicant")
    combined_score["p_final"] = np.where(
        referred_test,
        combined_score["p_second_look"],
        combined_score["p_default"],
    )
    combined_score["score_final"] = 1 - combined_score["p_final"]
    combined_score["approved_final"] = combined_score["first_pass_approve"].astype(bool) | (
        referred_test & (combined_score["p_second_look"] < selected_cutoff)
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "prep_rules": prep_rules,
            "calibrator": calibrator,
            "best_iteration": final_iteration,
            "best_params": best_params,
            "second_look_cutoff": selected_cutoff,
            "reference_cutoff": second_cutoff,
            "max_approved_default_rate": max_approved_default_rate,
            "excluded_features": EXCLUDED_FEATURES,
        },
        ARTIFACTS_DIR / "second_look_lightgbm.joblib",
    )
    metrics.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    two_stage.to_csv(REPORTS_DIR / "two_stage.csv", index=False)
    policy_thresholds.to_csv(REPORTS_DIR / "policy_thresholds.csv", index=False)
    pd.DataFrame(tuning_rows).assign(
        selected=lambda frame: frame["candidate"].eq(
            min(tuning_rows, key=lambda row: row["selection_logloss"])["candidate"]
        )
    ).to_csv(REPORTS_DIR / "tuning_results.csv", index=False)
    pd.Series(
        {
            "train_referrals": len(train_referral_frame),
            "early_stop_referrals": len(early_stop_referrals),
            "candidate_selection_referrals": len(model_selection_referrals),
            "calibration_referrals": len(calibration_referrals),
            "policy_validation_referrals": len(policy_referrals),
            "holdout_referrals": len(holdout_referrals),
            "selected_params": repr(best_params),
            "selected_iteration": int(final_iteration),
            "selected_second_cutoff": selected_cutoff,
            "reference_second_cutoff": second_cutoff,
            "max_approved_default_rate": max_approved_default_rate,
            "stage1_threshold": float(logistic_artifact["stage1_threshold"]),
            "train_route_weeks": (
                f"{TRAIN_ROUTE_WARMUP_END_WEEK + TRAIN_ROUTE_CALIBRATION_WEEKS + 1}-"
                f"{int(splits['train']['WEEK_NUM'].max())}"
            ),
            "early_stop_weeks": f"64-{MODEL_SELECTION_START_WEEK - 1}",
            "candidate_selection_weeks": f"{MODEL_SELECTION_START_WEEK}-{EARLY_STOP_END_WEEK}",
            "calibration_weeks": f"{EARLY_STOP_END_WEEK + 1}-{CALIBRATION_END_WEEK}",
            "policy_validation_weeks": f"{CALIBRATION_END_WEEK + 1}-75",
            "holdout_weeks": "76-91",
        },
        name="value",
    ).to_json(REPORTS_DIR / "run_config.json", indent=2)
    importance = pd.DataFrame(
        {
            "feature": final_model.feature_name(),
            "gain_importance": final_model.feature_importance(importance_type="gain"),
            "split_importance": final_model.feature_importance(importance_type="split"),
        }
    ).sort_values("gain_importance", ascending=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    holdout_referrals.assign(
        p_default_raw=p_hold_raw,
        p_default=p_hold,
    ).to_csv(PREDICTIONS_DIR / "second_look_lightgbm_holdout.csv", index=False)
    score_predictions.to_csv(PREDICTIONS_DIR / "second_look_lightgbm_score.csv", index=False)
    combined_score.to_csv(combined_score_path, index=False)
    print(f"Selected parameters: {best_params}")
    print(f"Best iteration: {final_iteration}")
    print(f"Validation-selected second-look cutoff: {selected_cutoff:.2%}")
    print("\nHoldout model metrics:")
    print(metrics.round(4).to_string(index=False))
    print("\nTwo-stage holdout comparison:")
    print(two_stage.round(4).to_string(index=False))
    print(f"\nSaved reports to {REPORTS_DIR}")
    if combined_score_path.is_file():
        print(f"Combined test scores saved to {combined_score_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path, default=STAGE1_PATH)
    parser.add_argument("--second-cutoff", type=float, default=SECOND_LOOK_CUTOFF)
    parser.add_argument("--logistic-artifact", type=Path, default=LOGISTIC_ARTIFACT_PATH)
    parser.add_argument("--max-approved-default-rate", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 < args.second_cutoff < 1:
        parser.error("--second-cutoff must be between 0 and 1")
    if not 0 < args.max_approved_default_rate < 1:
        parser.error("--max-approved-default-rate must be between 0 and 1")
    run(
        args.stage1,
        args.second_cutoff,
        args.logistic_artifact,
        args.max_approved_default_rate,
    )