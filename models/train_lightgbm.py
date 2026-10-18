"""
train_lightgbm.py

Trains a LightGBM model on the train split, early-stopping on the tune split, and evaluates on the tune split.

Returns (model, prep_rules) & can reuse prep_rules with apply_prep() for holdout, score, demo.
Holdout split but not scored here: to be used for final evaluation after tuning.
"""

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from splits import make_splits, get_features_target_segment

TRAIN_PATH = "data/features/train_features.csv"
TEST_PATH = "data/features/test_features.csv"
SEED = 42


# -------------  prep --------------------------------
def fit_prep(X_train: pd.DataFrame) -> dict:
    """Learn prep decisions from the train split only, so every split is prepped the same way."""
    # get category lists per text column
    text_columns = X_train.select_dtypes(include=["object", "string"]).columns
    categories_by_column = {col: pd.Categorical(X_train[col].dropna()).categories
                            for col in text_columns}
 
    # cols  with a single value in train not useful -> drop them
    constant_columns = [col for col in X_train.columns if X_train[col].nunique(dropna=False) <= 1]
 
    print(f"Features: {X_train.shape[1]} | categorical: {len(categories_by_column)} "
          f"| dropped constant: {len(constant_columns)}")
    # prep_rules = the decisions learned from train; pass to apply_prep for every split
    return {
        "feature_columns": list(X_train.columns),
        "categories_by_column": categories_by_column,
        "constant_columns": constant_columns,
    }


def apply_prep(X, prep_rules: dict) -> pd.DataFrame:
    """Apply train's prep decisions (from fit_prep) to any split: train, tune, holdout or score."""
    # same columns in the same order as train; any column missing is filled with NaN.
    X = X.reindex(columns=prep_rules["feature_columns"]).copy()

    # booleans: True/False -> 1/0 
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)


    # handle missing values in numeric columns: -1 -> NaN
    numeric_columns = X.select_dtypes(include="number").columns
    X[numeric_columns] = X[numeric_columns].mask(X[numeric_columns] == -1)

    # using train's category list; categories never seen in train become NaN.
    for col, categories in prep_rules["categories_by_column"].items():
        X[col] = pd.Categorical(X[col], categories=categories)

    # drop the columns that were constant in train
    return X.drop(columns=prep_rules["constant_columns"])


# ------------- training --------------------------------
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss", "average_precision"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,   # column subsampling
    "bagging_fraction": 0.8,   # row subsampling
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "is_unbalance": False,     # or set scale_pos_weight instead
    "verbose": -1,
    "seed": SEED,
}


def train_lgbm(X_train, y_train, X_tune, y_tune, params=None,
               num_boost_round=5000, early_stopping=100, log_every=200):
    """Train LightGBM on train, early-stopping on tune."""
    final_params = {**DEFAULT_PARAMS, **(params or {})} # merge default and user input params

    train_data = lgb.Dataset(X_train, y_train)
    tune_data = lgb.Dataset(X_tune, y_tune, reference=train_data)

    callbacks = [lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(log_every)]

    model = lgb.train(final_params, train_data, num_boost_round=num_boost_round,
                      valid_sets=[tune_data], valid_names=["tune"], callbacks=callbacks)
    return model


# ------------ evaluation --------------------------------
def evaluate(model, X, y, seg, name="tune"):
    """Metrics for one split, overall and per segment. Returns a metrics table and predictions."""
    default_prob = model.predict(X, num_iteration=model.best_iteration)
    segments = [("all", seg >= 0), ("established", seg == 0), ("thin-file", seg == 1)]
    metric_rows = []
    for segment, in_segment in segments:
        y_segment, prob_segment = y[in_segment], default_prob[in_segment]
        auc = roc_auc_score(y_segment, prob_segment)
        metrics = {
            "split": name,
            "segment": segment,
            "n": int(in_segment.sum()),
            "auc": auc,
            "gini": 2 * auc - 1,
            "pr_auc": average_precision_score(y_segment, prob_segment),
            "logloss": log_loss(y_segment, prob_segment),
            "best_iter": model.best_iteration,
        }
        metric_rows.append(metrics)   

    all_metrics = pd.DataFrame(metric_rows)
    print(f"best_iteration: {model.best_iteration}")
    print(all_metrics.round(4).to_string(index=False))

    return all_metrics, default_prob


# ------------ execution --------------------------------
def run(train_path=TRAIN_PATH, test_path=TEST_PATH) -> tuple[lgb.Booster, dict]:
    """Load, split, prep, train, evaluate on tune. Returns the model and prep_rules for reuse."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    splits = make_splits(train_df, test_df)
    X_train, y_train, _ = get_features_target_segment(splits["train"])
    X_tune, y_tune, seg_tune = get_features_target_segment(splits["tune"])

    prep_rules = fit_prep(X_train) 
    X_train = apply_prep(X_train, prep_rules)
    X_tune = apply_prep(X_tune, prep_rules)

    model = train_lgbm(X_train, y_train, X_tune, y_tune)
    evaluate(model, X_tune, y_tune, seg_tune, "tune")
    # model.save_model("lgbm_model.txt")
    return model, prep_rules


if __name__ == "__main__":
    run()