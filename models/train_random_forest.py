"""
train_random_forest.py

Random forest trained on all applicants, the first model in the Advisor's design. Thin-file
applicants it rejects get a second look from train_second_look_random_forest.py, which reads the
holdout predictions this script writes.

Run after features.py:

    python models/train_random_forest.py          # about 20 minutes
    python models/train_random_forest.py --quick  # skip the tuning loop

Splits are from splits.py. Settings and calibration are picked on the tune weeks and the holdout is
scored once. Tables and plots go to reports/random_forest, the model to models/artifacts
(gitignored, about 75 MB).
"""

import argparse
import sys
from pathlib import Path

import joblib
import optuna
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splits import SEGMENT, get_features_target_segment, make_splits

ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT_DIR / "data" / "features"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"
PREDICTIONS_DIR = ROOT_DIR / "data" / "predictions"
REPORTS_DIR = ROOT_DIR / "reports" / "random_forest"

TEXT_COLS = ["education_927M", "maritalstatus_703M", "description_5085714M"]
BOOL_COLS = ["has_second_person"]
# not inputs. is_thin_file is only for checking results by group, and external_coverage counts the
# bureau as one of its sources so it would give the segment away
EXCLUDED_COLS = [SEGMENT, "external_coverage"]

SEED = 42
CUTOFF = 0.20            # approve if the chance of default is under this (Advisor's cost model)
PROFIT_REPAID = 2000     # earned on a loan that is paid back
PROFIT_DEFAULT = -8000   # lost on a loan that defaults
TUNE_SAMPLE_ROWS = 200_000
# same range as team 1C's sweep
SWEEP_CUTOFFS = np.round(np.arange(0.08, 0.42, 0.02), 2)

DEFAULT_SETTINGS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 100,
    "max_features": "sqrt",
}

# the Advisor's slide lists tree count, depth and leaf size. tree count stays fixed, more trees only cost time
GRID = [
    {"max_depth": d, "min_samples_leaf": leaf, "max_features": feats}
    for d in (None, 16)
    for leaf in (50, 200, 500)
    for feats in ("sqrt", 0.3)
]


# ------------- prep --------------------------------
def fit_encoder(X_train: pd.DataFrame) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[TEXT_COLS])
    return encoder


def prep(X: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    X = X.drop(columns=EXCLUDED_COLS)
    onehot = pd.DataFrame(
        encoder.transform(X[TEXT_COLS]),
        columns=encoder.get_feature_names_out(TEXT_COLS),
        index=X.index,
    )
    X = X.drop(columns=TEXT_COLS)
    for col in BOOL_COLS:
        X[col] = X[col].astype(int)
    return pd.concat([X, onehot], axis=1)


# ------------- training --------------------------------
def fit_forest(X: pd.DataFrame, y: pd.Series, settings: dict) -> RandomForestClassifier:
    model = RandomForestClassifier(n_jobs=-1, random_state=SEED, **settings)
    model.fit(X, y)
    return model

def tune(X_train: pd.DataFrame, y_train: pd.Series, X_tune: pd.DataFrame, y_tune: pd.Series, n_bayesian_trials: int = 12,) -> dict:
    sample = X_train.sample(min(TUNE_SAMPLE_ROWS, len(X_train)), random_state=SEED).index
    X_sub, y_sub = X_train.loc[sample], y_train.loc[sample]
    best_settings, best_auc = None, -1.0

    print("Grid Search")
    for settings in GRID:
        settings = {"n_estimators": 200, **settings}
        model = fit_forest(X_sub, y_sub, settings)
        auc = roc_auc_score(y_tune, model.predict_proba(X_tune)[:, 1])
        print(f"  [Grid] {settings}  tune AUC {auc:.4f}")
        if auc > best_auc:
            best_settings, best_auc = settings, auc

    print("Bayesian Search using optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        use_depth = trial.suggest_categorical("use_depth", [True, False])
        max_depth = trial.suggest_int("max_depth", 10, 28) if use_depth else None
        candidate = {
            "n_estimators": 200,
            "max_depth": max_depth,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 25, 500, log=True),
            "max_features": trial.suggest_float("max_features", 0.1, 0.4),
        }
        model = fit_forest(X_sub, y_sub, candidate)
        auc = roc_auc_score(y_tune, model.predict_proba(X_tune)[:, 1])
        print(f"  [Bayes {trial.number:02d}] {candidate}  tune AUC {auc:.4f}")
        return auc

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_bayesian_trials)

    if study.best_value > best_auc:
        best_auc = study.best_value
        bp = study.best_params
        best_settings = {
            "max_depth": bp["max_depth"] if bp["use_depth"] else None,
            "min_samples_leaf": bp["min_samples_leaf"],
            "max_features": bp["max_features"],
        }

    best_settings = {**best_settings, "n_estimators": DEFAULT_SETTINGS["n_estimators"]}
    print(f"best overall: {best_settings}  tune AUC {best_auc:.4f}")
    return best_settings


# ------------- calibration --------------------------------
def fit_calibrators(p: np.ndarray, y: pd.Series, seg: pd.Series) -> dict:
    calibrators = {}
    for value in (0, 1):
        rows = (seg == value).to_numpy()
        calibrators[value] = IsotonicRegression(out_of_bounds="clip").fit(p[rows], y[rows])
    return calibrators


def calibrate(p: np.ndarray, seg: pd.Series, calibrators: dict) -> np.ndarray:
    out = p.copy()
    for value, calibrator in calibrators.items():
        rows = (seg == value).to_numpy()
        out[rows] = calibrator.predict(p[rows])
    return out


# ------------- evaluation --------------------------------
def inclusion_gap(p: np.ndarray, y: pd.Series, seg: pd.Series, cutoff: float = CUTOFF) -> float:
    """Approval rate among repayers, established minus thin-file (formula from splits.py)."""
    approved = p < cutoff
    repaid = (y == 0).to_numpy()
    seg = seg.to_numpy()
    return approved[repaid & (seg == 0)].mean() - approved[repaid & (seg == 1)].mean()


def expected_profit(p: np.ndarray, y: pd.Series, cutoff: float = CUTOFF) -> float:
    """Advisor's slide 8: +2000 per repaid loan, -8000 per default."""
    approved = p < cutoff
    outcomes = np.where(y.to_numpy()[approved] == 1, PROFIT_DEFAULT, PROFIT_REPAID)
    return float(outcomes.sum())


def metrics(p: np.ndarray, y: pd.Series, seg: pd.Series, split: str, version: str) -> list[dict]:
    rows = []
    groups = {
        "all": np.ones(len(p), dtype=bool),
        "established": (seg == 0).to_numpy(),
        "thin_file": (seg == 1).to_numpy(),
    }
    for name, mask in groups.items():
        approved_mask = p[mask] < CUTOFF
        rows.append({
            "split": split,
            "version": version,
            "segment": name,
            "rows": int(mask.sum()),
            "auc": roc_auc_score(y[mask], p[mask]),
            "pr_auc": average_precision_score(y[mask], p[mask]),
            "logloss": log_loss(y[mask], p[mask], labels=[0, 1]),
            "brier": brier_score_loss(y[mask], p[mask]),
            "default_rate": float(y[mask].mean()),
            "mean_predicted_default": float(p[mask].mean()),
            "calibration_gap": float(p[mask].mean() - y[mask].mean()),   # positive = over-predicts risk
            "approval_rate": float(approved_mask.mean()),
            "default_rate_of_approved": float(y[mask][approved_mask].mean()) if approved_mask.any() else np.nan,
        })
    rows[0]["inclusion_gap"] = inclusion_gap(p, y, seg)
    rows[0]["expected_profit"] = expected_profit(p, y)
    return rows


def cutoff_sweep(p_raw: np.ndarray, p_cal: np.ndarray, y: pd.Series, seg: pd.Series) -> pd.DataFrame:
    rows = []
    repaid = (y == 0).to_numpy()
    established = (seg == 0).to_numpy()
    thin_file = (seg == 1).to_numpy()
    for version, p in (("raw", p_raw), ("calibrated", p_cal)):
        for cutoff in SWEEP_CUTOFFS:
            approved = p < cutoff
            rows.append({
                "cutoff": cutoff,
                "version": version,
                "approval_rate": float(approved.mean()),
                "default_rate_of_approved": float(y[approved].mean()) if approved.any() else np.nan,
                "expected_profit": expected_profit(p, y, cutoff),
                "established_repayers_approved": float(approved[repaid & established].mean()),
                "thin_file_repayers_approved": float(approved[repaid & thin_file].mean()),
                "inclusion_gap": inclusion_gap(p, y, seg, cutoff),
            })
    return pd.DataFrame(rows)


def plot_reliability(p_raw: np.ndarray, p_cal: np.ndarray, y: pd.Series, seg: pd.Series, path: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, (title, p) in zip(axes, [("Before calibration", p_raw), ("After per-segment calibration", p_cal)]):
        for value, label in ((0, "established"), (1, "thin-file")):
            rows = (seg == value).to_numpy()
            actual, predicted = calibration_curve(y[rows], p[rows], n_bins=15, strategy="quantile")
            axis.plot(predicted, actual, marker="o", label=label)
        axis.plot([0, 0.7], [0, 0.7], linestyle="--", color="gray")
        axis.axvline(CUTOFF, linestyle=":", color="red")
        axis.set(title=title, xlabel="predicted default probability", xlim=(0, 0.7))
    axes[0].set_ylabel("actual default rate")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_cutoff_sweep(sweep: pd.DataFrame, path: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("expected_profit", "profit on holdout ($M)", 1e-6),
        ("inclusion_gap", "inclusion gap", 1),
        ("default_rate_of_approved", "default rate of approved loans", 1),
    ]
    for version, label in (("raw", "before calibration"), ("calibrated", "after per-segment calibration")):
        part = sweep[sweep["version"] == version]
        for axis, (col, _, scale) in zip(axes, panels):
            axis.plot(part["cutoff"], part[col] * scale, marker="o", ms=3, label=label)
    for axis, (_, ylabel, _) in zip(axes, panels):
        axis.axvline(CUTOFF, color="grey", linestyle=":", linewidth=1)
        axis.set(xlabel="cutoff (approve if the chance of default is under it)", ylabel=ylabel)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ------------- main --------------------------------
def main(quick: bool) -> None:
    for prefix in ("train", "test"):
        path = FEATURES_DIR / f"{prefix}_features.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run notebooks/features.py first.")
    for folder in (ARTIFACTS_DIR, PREDICTIONS_DIR, REPORTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(FEATURES_DIR / "train_features.csv")
    test_df = pd.read_csv(FEATURES_DIR / "test_features.csv")
    splits = make_splits(train_df, test_df)

    X_train, y_train, seg_train = get_features_target_segment(splits["train"])
    X_tune, y_tune, seg_tune = get_features_target_segment(splits["tune"])
    X_hold, y_hold, seg_hold = get_features_target_segment(splits["holdout"])
    X_score, _, seg_score = get_features_target_segment(splits["score"])

    encoder = fit_encoder(X_train)
    X_train, X_tune, X_hold, X_score = (prep(X, encoder) for X in (X_train, X_tune, X_hold, X_score))
    print(f"\nFeatures going into the forest: {X_train.shape[1]}")

    print("\nFitting with default settings...")
    model = fit_forest(X_train, y_train, DEFAULT_SETTINGS)
    p_tune = model.predict_proba(X_tune)[:, 1]
    print(f"default settings  tune AUC {roc_auc_score(y_tune, p_tune):.4f}")

    settings = DEFAULT_SETTINGS
    if not quick:
        print("\nTuning on a sample of train, scored on tune...")
        best = tune(X_train, y_train, X_tune, y_tune)
        best_model = fit_forest(X_train, y_train, best)
        p_best = best_model.predict_proba(X_tune)[:, 1]
        print(f"best settings on all of train  tune AUC {roc_auc_score(y_tune, p_best):.4f}")
        if roc_auc_score(y_tune, p_best) > roc_auc_score(y_tune, p_tune):
            model, p_tune, settings = best_model, p_best, best
        else:
            print("tuned settings did not beat the defaults, keeping the defaults")

    # fitted on tune only
    calibrators = fit_calibrators(p_tune, y_tune, seg_tune)

    # holdout, scored once
    p_hold_raw = model.predict_proba(X_hold)[:, 1]
    p_hold_cal = calibrate(p_hold_raw, seg_hold, calibrators)
    results = pd.DataFrame(
        metrics(p_tune, y_tune, seg_tune, "tune", "raw")
        + metrics(calibrate(p_tune, seg_tune, calibrators), y_tune, seg_tune, "tune", "calibrated")
        + metrics(p_hold_raw, y_hold, seg_hold, "holdout", "raw")
        + metrics(p_hold_cal, y_hold, seg_hold, "holdout", "calibrated")
    )
    print("\nSettings used:", settings)
    print(results.round(4).to_string(index=False))

    sweep = cutoff_sweep(p_hold_raw, p_hold_cal, y_hold, seg_hold)
    print("\nHoldout at each cutoff, after calibration:")
    print(sweep[sweep["version"] == "calibrated"].round(4).to_string(index=False))

    importance = pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    print("\nTop 15 features:")
    print(importance.head(15).to_string(index=False))

    joblib.dump(
        {"model": model, "encoder": encoder, "calibrators": calibrators, "settings": settings,
         "feature_columns": list(X_train.columns)},
        ARTIFACTS_DIR / "random_forest.joblib",
    )
    holdout = splits["holdout"][["case_id", "WEEK_NUM", SEGMENT, "target"]].copy()
    holdout["p_default_raw"] = p_hold_raw
    holdout["p_default"] = p_hold_cal
    holdout.to_csv(PREDICTIONS_DIR / "random_forest_holdout.csv", index=False)

    p_score_raw = model.predict_proba(X_score)[:, 1]
    score = splits["score"][["case_id", SEGMENT]].copy()
    score["p_default_raw"] = p_score_raw
    score["p_default"] = calibrate(p_score_raw, seg_score, calibrators)
    score["score"] = 1 - score["p_default"]   # chance they pay it back
    score.to_csv(PREDICTIONS_DIR / "random_forest_score.csv", index=False)

    results.to_csv(REPORTS_DIR / "metrics.csv", index=False)
    importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    sweep.to_csv(REPORTS_DIR / "cutoff_sweep.csv", index=False)
    plot_reliability(p_hold_raw, p_hold_cal, y_hold, seg_hold, REPORTS_DIR / "calibration.png")
    plot_cutoff_sweep(sweep, REPORTS_DIR / "cutoff_sweep.png")
    print(f"\n-> saved model to {ARTIFACTS_DIR}, predictions to {PREDICTIONS_DIR}, metrics, sweep and plots to {REPORTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the random forest model.")
    parser.add_argument("--quick", action="store_true", help="skip the tuning loop")
    main(quick=parser.parse_args().quick)
