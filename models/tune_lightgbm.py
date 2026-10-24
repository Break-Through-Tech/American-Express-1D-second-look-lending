"""

Run from the repository root with:
    python models/tune_lightgbm.py

"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss

from splits import TUNE_END_WEEK, TRAIN_END_WEEK, get_features_target_segment
from train_lightgbm import DEFAULT_PARAMS, apply_prep, fit_prep, train_lgbm

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "data/features/train_features.csv"
OUTPUT_PATH = ROOT_DIR / "reports/model/tuning_results.csv"

PARAMETER_CANDIDATES = (
    {},
    {"num_leaves": 15, "min_child_samples": 50},
    {"num_leaves": 63, "min_child_samples": 100},
    {"num_leaves": 31, "min_child_samples": 100, "lambda_l2": 5.0},
    {"num_leaves": 31, "min_child_samples": 200, "lambda_l2": 10.0},
    {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 100},
)


def tune_parameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    candidates: tuple[dict, ...] = PARAMETER_CANDIDATES,
) -> tuple[dict, lgb.Booster, pd.DataFrame]:
    """Choose the candidate with the lowest tune-set LogLoss."""
    if not candidates:
        raise ValueError("At least one parameter candidate is required")

    results = []
    best_params = None
    best_model = None
    best_loss = float("inf")

    for trial, candidate in enumerate(candidates):
        model = train_lgbm(
            X_train,
            y_train,
            X_tune,
            y_tune,
            params={
                **candidate,
                "objective": "binary",
                "metric": "binary_logloss",
            },
            early_stopping=50,
            log_every=0,
        )
        probabilities = model.predict(X_tune, num_iteration=model.best_iteration)
        tune_loss = log_loss(y_tune, probabilities, labels=[0, 1])
        results.append({
            "trial": trial,
            "is_default_params": not candidate,
            "params": repr({**DEFAULT_PARAMS, **candidate}),
            "best_iteration": model.best_iteration,
            "tune_logloss": tune_loss,
        })
        if tune_loss < best_loss:
            best_loss = tune_loss
            best_params = candidate.copy()
            best_model = model

    if best_params is None or best_model is None:
        raise RuntimeError("Parameter tuning completed without selecting a model")

    results_df = pd.DataFrame(results).sort_values("tune_logloss").reset_index(drop=True)
    results_df.insert(0, "selected", results_df["tune_logloss"].eq(best_loss))
    return best_params, best_model, results_df


def run(train_path: Path = TRAIN_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Tune candidates on thin-file train and tune period applicants."""
    data = pd.read_csv(train_path)
    thin_file = data.loc[data["is_thin_file"] == 1]
    train_df = thin_file.loc[thin_file["WEEK_NUM"] <= TRAIN_END_WEEK]
    tune_df = thin_file.loc[
        (thin_file["WEEK_NUM"] > TRAIN_END_WEEK)
        & (thin_file["WEEK_NUM"] <= TUNE_END_WEEK)
    ]
    if train_df.empty or tune_df.empty:
        raise ValueError("Train or tune period has no thin-file applicants")

    X_train, y_train, _ = get_features_target_segment(train_df)
    X_tune, y_tune, _ = get_features_target_segment(tune_df)

    # Drop is_thin_file from feature set to avoid target/segment column leakage
    if "is_thin_file" in X_train.columns:
        X_train = X_train.drop(columns=["is_thin_file"])
    if "is_thin_file" in X_tune.columns:
        X_tune = X_tune.drop(columns=["is_thin_file"])

    X_tune = X_tune.reindex(columns=X_train.columns)

    prep_rules = fit_prep(X_train)
    X_train = apply_prep(X_train, prep_rules)
    X_tune = apply_prep(X_tune, prep_rules)

    print(f"Thin-file train applicants: {len(train_df):,}")
    print(f"Thin-file tune applicants: {len(tune_df):,}")
    best_params, _, results = tune_parameters(X_train, y_train, X_tune, y_tune)
    selected = results.loc[results["selected"]].iloc[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print("\nHyperparameter results (sorted by tune LogLoss):")
    print(results[["trial", "selected", "best_iteration", "tune_logloss"]].to_string(index=False))
    print(f"\nSelected parameters: {best_params or 'DEFAULT_PARAMS'}")
    print(f"Best tune LogLoss: {selected['tune_logloss']:.6f}")
    print(f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    run(args.train, args.output)
