"""
lightgbm.py

"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

TRAIN_PATH = "data/cleaned/train_cleaned.csv"
TEST_PATH = "data/cleaned/test_cleaned.csv"
TARGET = "target"
DROP_COLS = ["case_id", "date_decision", "WEEK_NUM", "MONTH"]  # columns to drop
VAL_SIZE = 0.2  # fraction of train that's used for validation
SEED = 42


# -------------  prep --------------------------------
def prep(train_df, test_df, target=TARGET, drop_cols=DROP_COLS):
    """Prepares the data for training and testing."""
    y_train = train_df[target].astype(int)
    y_test = test_df[target].astype(int) if target in test_df else None

    drop = [c for c in drop_cols + [target] if c in train_df.columns]
    X_train = train_df.drop(columns=drop)
    X_test = test_df.reindex(columns=X_train.columns)  # same columns, same order

    # any leftover text columns -> pandas category
    obj_cols = X_train.select_dtypes(include=["object", "string"]).columns
    for c in obj_cols:
        cats = pd.Categorical(X_train[c].dropna()).categories
        X_train[c] = pd.Categorical(X_train[c], categories=cats)
        X_test[c] = pd.Categorical(X_test[c], categories=cats)

    # drop constant columns
    const = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    X_train = X_train.drop(columns=const)
    X_test = X_test.drop(columns=const)

    print(f"Features: {X_train.shape[1]} | categorical: {len(obj_cols)} | dropped constant: {len(const)}")
    return X_train, y_train, X_test, y_test


# ------------- data split --------------------------------
def make_splits(X, y, val_size=VAL_SIZE, seed=SEED):
    """Splits the data into train and validation sets."""
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=seed
    )
    # bad/defaulted rate (target = 1) similar in each split
    print(f"Train: {len(X_train)} (bad/defaulted rate {y_train.mean():.3%}) | "
          f"Val: {len(X_validation)} (bad/defaulted rate {y_validation.mean():.3%})")
    return X_train, X_validation, y_train, y_validation



# ------------- training --------------------------------
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
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


def train_lgbm(X_train, y_train, X_validation, y_validation, params=None,
               num_boost_round=5000, early_stopping=100, log_every=0):
    """Trains a LightGBM model."""
    p = {**DEFAULT_PARAMS, **(params or {})}

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_validation, y_validation, reference=dtrain)

    callbacks = [lgb.early_stopping(early_stopping, verbose=False)]
    if log_every:
        callbacks.append(lgb.log_evaluation(log_every))

    model = lgb.train(p, dtrain, num_boost_round=num_boost_round,
                      valid_sets=[dval], valid_names=["val"], callbacks=callbacks)
    return model


# ------------ evaluation --------------------------------
def evaluate(model, X, y, name="val"):
    """Evaluates a LightGBM model on a dataset."""
    pred = model.predict(X, num_iteration=model.best_iteration)
    auc = roc_auc_score(y, pred)
    metrics = {
        "split": name,
        "auc": auc,
        "gini": 2 * auc - 1,
        "pr_auc": average_precision_score(y, pred),
        "logloss": log_loss(y, pred),
        "best_iter": model.best_iteration,
    }
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    return metrics, pred




if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X, y, X_test, y_test = prep(train_df, test_df)
    X_train, X_validation, y_train, y_validation = make_splits(X, y)

    # --- baseline with default params ---
    base = train_lgbm(X_train, y_train, X_validation, y_validation, log_every=200)
    evaluate(base, X_validation, y_validation, "baseline val")

    # feature importance
    importance = pd.DataFrame({
        "feature": base.feature_name(),
        "gain": base.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    # importance.to_csv("lgbm_feature_importance.csv", index=False)
    print(importance.head(20))

    # base.save_model("lgbm_final.txt")  
    # to reload model: lgb.Booster(model_file="lgbm_final.txt")