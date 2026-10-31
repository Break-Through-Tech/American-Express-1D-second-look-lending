"""
Train, tune, calibrate, evaluate, and save the Logistic Regression baseline.

Run from the repository root:

    python models/train_logistic.py

The script uses the shared time split in models/splits.py. All learned preprocessing
is fitted on weeks 0-63 only. Weeks 64-69 select regularization and fit probability
calibration, weeks 70-75 select the provisional lending threshold, and weeks 76-91
are scored once for the final holdout report.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from splits import get_features_target_segment, make_splits


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "data" / "features" / "train_features.csv"
TEST_PATH = ROOT_DIR / "data" / "features" / "test_features.csv"
REPORT_DIR = ROOT_DIR / "reports" / "logistic"
FIGURE_DIR = REPORT_DIR / "figures"
ARTIFACT_PATH = ROOT_DIR / "models" / "artifacts" / "logistic_pipeline.joblib"
PREDICTION_PATH = ROOT_DIR / "data" / "predictions" / "logistic_score_predictions.csv"

SEED = 42
C_VALUES = (0.01, 0.1, 1.0, 10.0)
MAX_ITERATIONS = 300
CALIBRATION_END_WEEK = 69
DEFAULT_RATE_CEILING = 0.20
REPAID_APPROVAL_VALUE = 1.0
DEFAULT_APPROVAL_COST = -4.0
EXCLUDED_MODEL_FEATURES = ("is_thin_file", "external_coverage")

SEGMENT_LABELS = {
    "all": "All applicants",
    "established": "Established",
    "thin_file": "Thin-file",
}

FRIENDLY_FEATURE_NAMES = {
    "riskassesment_940T": "Risk assessment score",
    "numberofqueries_146L": "Recent credit queries",
    "mainoccupationinc_384A": "Applicant income",
    "income_total": "Total household income",
    "empl_employedtotal_800L_mean": "Average employment history",
    "days_employed_700P": "Employment duration",
    "credamount_770A": "Requested credit",
    "annuity_780A": "Loan annuity",
    "credit_to_income": "Credit-to-income ratio",
    "annuity_to_income": "Annuity-to-income ratio",
    "credit_to_annuity": "Credit-to-annuity ratio",
    "employment_share_of_life": "Employment share of life",
    "personal_income_share": "Applicant share of household income",
    "average_overdue_ratio": "Average overdue-to-credit ratio",
    "overdue_to_credit": "Maximum overdue-to-credit ratio",
    "amount_4527230A_mean": "Average tax-registry amount",
    "amount_4527230A_median": "Median tax-registry amount",
    "amount_4527230A_max": "Maximum tax-registry amount",
    "credamount_590A_mean": "Average prior application amount",
    "credamount_590A_median": "Median prior application amount",
    "credamount_590A_max": "Maximum prior application amount",
    "pmts_dpdvalue_108P_mean": "Average days past due",
    "pmts_dpdvalue_108P_median": "Median days past due",
    "pmts_dpdvalue_108P_max": "Maximum days past due",
    "overdueamountmax_950A_mean": "Average maximum overdue amount",
    "payment_rowcount_sum": "Recorded bureau payments",
    "dpd_bucket": "Days-past-due bucket",
    "months_since_last_application": "Months since last application",
    "n_unknown": "Number of unknown values",
    "external_coverage": "External data-source coverage",
    "total_external_records": "Total external records",
}


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to finite log-odds for sigmoid calibration."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def model_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, pd.Series]:
    """Use the shared splitter helper and remove the fairness segment from model inputs."""
    features, target, segment = get_features_target_segment(frame)
    features = features.drop(
        columns=[column for column in EXCLUDED_MODEL_FEATURES if column in features.columns]
    )
    return features, target, segment


def build_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Recreate Task 3 preprocessing while learning every parameter from train only."""
    categorical_columns = features.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "float32",
                FunctionTransformer(
                    np.asarray,
                    kw_args={"dtype": np.float32},
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "unknown_to_median",
                SimpleImputer(
                    missing_values=-1,
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "missing_indicator",
                MissingIndicator(
                    missing_values=-1,
                    features="missing-only",
                    error_on_new=False,
                ),
                numeric_columns,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                ),
                categorical_columns,
            ),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
        n_jobs=-1,
    )
    return preprocessor, numeric_columns, categorical_columns


def transform_float32(preprocessor: ColumnTransformer, features: pd.DataFrame, fit: bool = False) -> np.ndarray:
    """Fit or apply preprocessing and keep the model matrix memory-efficient."""
    transformed = preprocessor.fit_transform(features) if fit else preprocessor.transform(features)
    return np.asarray(transformed, dtype=np.float32)


def fit_logistic_candidates(
    train_matrix: np.ndarray,
    train_target: pd.Series,
    selection_matrix: np.ndarray,
    selection_target: pd.Series,
) -> tuple[LogisticRegression, pd.DataFrame]:
    """Fit candidate regularization strengths and select by time-forward LogLoss."""
    models: dict[float, LogisticRegression] = {}
    rows = []

    for regularization in C_VALUES:
        started = time.perf_counter()
        model = LogisticRegression(
            C=regularization,
            penalty="l2",
            solver="saga",
            max_iter=MAX_ITERATIONS,
            tol=1e-3,
            random_state=SEED,
            n_jobs=-1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_matrix, train_target)

        probabilities = model.predict_proba(selection_matrix)[:, 1]
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        rows.append(
            {
                "C": regularization,
                "logloss": log_loss(selection_target, probabilities, labels=[0, 1]),
                "roc_auc": roc_auc_score(selection_target, probabilities),
                "pr_auc": average_precision_score(selection_target, probabilities),
                "brier": brier_score_loss(selection_target, probabilities),
                "iterations": int(model.n_iter_[0]),
                "converged": converged,
                "fit_seconds": time.perf_counter() - started,
            }
        )
        models[regularization] = model
        print(
            f"C={regularization:g}: LogLoss={rows[-1]['logloss']:.6f}, "
            f"AUC={rows[-1]['roc_auc']:.4f}, iterations={rows[-1]['iterations']}"
        )

    results = pd.DataFrame(rows).sort_values(["logloss", "C"]).reset_index(drop=True)
    minimum_loss = results["logloss"].min()
    near_best = results[results["logloss"] <= minimum_loss + 1e-4]
    selected_c = float(near_best.sort_values("C").iloc[0]["C"])
    results["selected"] = results["C"].eq(selected_c)
    return models[selected_c], results


def fit_calibrator(method: str, probabilities: np.ndarray, target: pd.Series):
    """Fit raw, sigmoid, or isotonic probability calibration."""
    if method == "raw":
        return None
    if method == "sigmoid":
        calibrator = LogisticRegression(C=1_000_000, solver="lbfgs", max_iter=1_000)
        calibrator.fit(probability_logit(probabilities).reshape(-1, 1), target)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(probabilities, target)
        return calibrator
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator, probabilities: np.ndarray) -> np.ndarray:
    """Apply one of the supported probability-calibration methods."""
    if method == "raw":
        calibrated = probabilities
    elif method == "sigmoid":
        calibrated = calibrator.predict_proba(
            probability_logit(probabilities).reshape(-1, 1)
        )[:, 1]
    elif method == "isotonic":
        calibrated = calibrator.predict(probabilities)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return np.clip(np.asarray(calibrated, dtype=float), 1e-6, 1 - 1e-6)


def select_calibration(
    model: LogisticRegression,
    calibration_matrix: np.ndarray,
    calibration_target: pd.Series,
    policy_matrix: np.ndarray,
    policy_target: pd.Series,
) -> tuple[str, object, pd.DataFrame, np.ndarray]:
    """Fit calibration on weeks 64-69 and compare it on later weeks 70-75."""
    calibration_probabilities = model.predict_proba(calibration_matrix)[:, 1]
    policy_probabilities = model.predict_proba(policy_matrix)[:, 1]
    calibrators = {}
    rows = []

    for method in ("raw", "sigmoid", "isotonic"):
        calibrator = fit_calibrator(method, calibration_probabilities, calibration_target)
        evaluated = apply_calibrator(method, calibrator, policy_probabilities)
        rows.append(
            {
                "method": method,
                "logloss": log_loss(policy_target, evaluated, labels=[0, 1]),
                "brier": brier_score_loss(policy_target, evaluated),
                "roc_auc": roc_auc_score(policy_target, evaluated),
                "mean_predicted_default": evaluated.mean(),
                "actual_default_rate": policy_target.mean(),
            }
        )
        calibrators[method] = calibrator

    results = pd.DataFrame(rows).sort_values(["logloss", "brier"]).reset_index(drop=True)
    minimum_loss = results["logloss"].min()
    near_best = results[results["logloss"] <= minimum_loss + 1e-4].copy()
    simplicity_rank = {"raw": 0, "sigmoid": 1, "isotonic": 2}
    near_best["simplicity_rank"] = near_best["method"].map(simplicity_rank)
    selected_method = str(near_best.sort_values("simplicity_rank").iloc[0]["method"])
    results["selected"] = results["method"].eq(selected_method)
    calibrated_policy_probabilities = apply_calibrator(
        selected_method,
        calibrators[selected_method],
        policy_probabilities,
    )
    return (
        selected_method,
        calibrators[selected_method],
        results,
        calibrated_policy_probabilities,
    )


def segment_masks(segment: pd.Series) -> dict[str, np.ndarray]:
    """Return overall and fairness-segment masks in a consistent order."""
    values = np.asarray(segment)
    return {
        "all": np.ones(len(values), dtype=bool),
        "established": values == 0,
        "thin_file": values == 1,
    }


def metric_rows(
    target: pd.Series,
    probabilities: np.ndarray,
    segment: pd.Series,
    split_name: str,
) -> list[dict]:
    """Calculate model-quality and calibration metrics overall and by segment."""
    target_values = np.asarray(target)
    rows = []
    for segment_name, mask in segment_masks(segment).items():
        segment_target = target_values[mask]
        segment_probabilities = probabilities[mask]
        rows.append(
            {
                "split": split_name,
                "segment": segment_name,
                "n": int(mask.sum()),
                "default_rate": segment_target.mean(),
                "mean_predicted_default": segment_probabilities.mean(),
                "calibration_gap": segment_probabilities.mean() - segment_target.mean(),
                "roc_auc": roc_auc_score(segment_target, segment_probabilities),
                "pr_auc": average_precision_score(segment_target, segment_probabilities),
                "logloss": log_loss(segment_target, segment_probabilities, labels=[0, 1]),
                "brier": brier_score_loss(segment_target, segment_probabilities),
            }
        )
    return rows


def evaluate_policy(
    target: pd.Series,
    probabilities: np.ndarray,
    segment: pd.Series,
    threshold: float,
) -> dict:
    """Evaluate an approve-below-threshold policy using normalized profit units."""
    target_values = np.asarray(target)
    segment_values = np.asarray(segment)
    approved = probabilities < threshold
    approved_count = int(approved.sum())
    repaid_approved = approved & (target_values == 0)
    defaulted_approved = approved & (target_values == 1)
    total_profit = (
        repaid_approved.sum() * REPAID_APPROVAL_VALUE
        + defaulted_approved.sum() * DEFAULT_APPROVAL_COST
    )

    established_repayers = (target_values == 0) & (segment_values == 0)
    thin_file_repayers = (target_values == 0) & (segment_values == 1)
    established_repayer_approval = approved[established_repayers].mean()
    thin_file_repayer_approval = approved[thin_file_repayers].mean()

    return {
        "threshold": threshold,
        "approved_count": approved_count,
        "approval_rate": approved.mean(),
        "approved_default_rate": target_values[approved].mean() if approved_count else np.nan,
        "normalized_total_profit": float(total_profit),
        "profit_per_applicant": float(total_profit / len(target_values)),
        "established_repayer_approval": established_repayer_approval,
        "thin_file_repayer_approval": thin_file_repayer_approval,
        "inclusion_gap": established_repayer_approval - thin_file_repayer_approval,
    }


def select_threshold(
    target: pd.Series,
    probabilities: np.ndarray,
    segment: pd.Series,
) -> tuple[float, pd.DataFrame]:
    """Choose the most profitable tune policy under the 20% default-rate ceiling."""
    threshold_grid = np.unique(
        np.concatenate([np.linspace(0.02, 0.50, 97), np.array([DEFAULT_RATE_CEILING])])
    )
    results = pd.DataFrame(
        [evaluate_policy(target, probabilities, segment, threshold) for threshold in threshold_grid]
    )
    feasible = results[
        (results["approved_count"] > 0)
        & (results["approved_default_rate"] <= DEFAULT_RATE_CEILING)
    ]
    if feasible.empty:
        raise RuntimeError("No policy threshold satisfies the default-rate ceiling")
    selected = feasible.sort_values(
        ["normalized_total_profit", "inclusion_gap"],
        ascending=[False, True],
    ).iloc[0]
    selected_threshold = float(selected["threshold"])
    results["selected"] = np.isclose(results["threshold"], selected_threshold)
    results["break_even_reference"] = np.isclose(
        results["threshold"], DEFAULT_RATE_CEILING
    )
    return selected_threshold, results


def friendly_feature_name(raw_name: str) -> str:
    """Convert common masked dataset names into readable report labels."""
    if raw_name.startswith("missingindicator_"):
        base = raw_name.removeprefix("missingindicator_")
        return f"{FRIENDLY_FEATURE_NAMES.get(base, base.replace('_', ' ').title())} missing"
    if raw_name.startswith("education_927M_"):
        return f"Education category {raw_name.removeprefix('education_927M_')}"
    if raw_name.startswith("maritalstatus_703M_"):
        return f"Marital-status category {raw_name.removeprefix('maritalstatus_703M_')}"
    if raw_name.startswith("description_5085714M_"):
        return f"Credit-file description {raw_name.removeprefix('description_5085714M_')}"
    if raw_name.startswith("classificationofcontr_13M_"):
        return f"Bureau contract category {raw_name.removeprefix('classificationofcontr_13M_')} count"
    return FRIENDLY_FEATURE_NAMES.get(raw_name, raw_name.replace("_", " ").title())


def coefficient_table(model: LogisticRegression, feature_names: np.ndarray) -> pd.DataFrame:
    """Create a sorted, readable coefficient table for model interpretation."""
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["absolute_coefficient"] = table["coefficient"].abs()
    table["direction"] = np.where(
        table["coefficient"] > 0,
        "higher predicted default risk",
        "lower predicted default risk",
    )
    table["display_name"] = table["feature"].map(friendly_feature_name)
    return table.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Save a report figure with consistent resolution and close it."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(figure)


def portable_path(path: Path) -> str:
    """Store repository-relative paths in committed run metadata when possible."""
    try:
        return str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)


def plot_roc(target: pd.Series, probabilities: np.ndarray, segment: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    target_values = np.asarray(target)
    for segment_name, mask in segment_masks(segment).items():
        false_positive, true_positive, _ = roc_curve(target_values[mask], probabilities[mask])
        auc = roc_auc_score(target_values[mask], probabilities[mask])
        axis.plot(
            false_positive,
            true_positive,
            linewidth=2,
            label=f"{SEGMENT_LABELS[segment_name]} (AUC {auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random ranking")
    axis.set(
        title="Holdout ROC curves",
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    save_figure(figure, "01_holdout_roc.png")


def plot_precision_recall(target: pd.Series, probabilities: np.ndarray, segment: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    target_values = np.asarray(target)
    for segment_name, mask in segment_masks(segment).items():
        precision, recall, _ = precision_recall_curve(target_values[mask], probabilities[mask])
        pr_auc = average_precision_score(target_values[mask], probabilities[mask])
        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{SEGMENT_LABELS[segment_name]} (AP {pr_auc:.3f})",
        )
    axis.set(
        title="Holdout precision-recall curves",
        xlabel="Recall of defaults",
        ylabel="Precision of default predictions",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    save_figure(figure, "02_holdout_precision_recall.png")


def plot_calibration(target: pd.Series, probabilities: np.ndarray, segment: pd.Series) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    target_values = np.asarray(target)
    for segment_name, mask in segment_masks(segment).items():
        observed, predicted = calibration_curve(
            target_values[mask],
            probabilities[mask],
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(
            predicted,
            observed,
            marker="o",
            linewidth=2,
            label=SEGMENT_LABELS[segment_name],
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set(
        title="Holdout probability calibration",
        xlabel="Mean predicted default probability",
        ylabel="Observed default rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(alpha=0.2)
    save_figure(figure, "03_holdout_calibration.png")


def plot_probability_distributions(
    target: pd.Series,
    probabilities: np.ndarray,
    segment: pd.Series,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    target_values = np.asarray(target)
    segment_values = np.asarray(segment)
    for axis, (segment_value, title) in zip(
        axes,
        ((0, "Established applicants"), (1, "Thin-file applicants")),
    ):
        in_segment = segment_values == segment_value
        axis.hist(
            probabilities[in_segment & (target_values == 0)],
            bins=40,
            range=(0, 1),
            density=True,
            alpha=0.65,
            label="Actually repaid",
        )
        axis.hist(
            probabilities[in_segment & (target_values == 1)],
            bins=40,
            range=(0, 1),
            density=True,
            alpha=0.65,
            label="Actually defaulted",
        )
        axis.set(title=title, xlabel="Predicted default probability", ylabel="Density")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Holdout score distributions")
    save_figure(figure, "04_holdout_score_distributions.png")


def plot_threshold_tradeoffs(results: pd.DataFrame, selected_threshold: float) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    series = (
        ("approval_rate", "Approval rate", "Rate"),
        ("approved_default_rate", "Default rate among approved", "Rate"),
        ("profit_per_applicant", "Normalized profit per applicant", "Profit units"),
        ("inclusion_gap", "Thin-file inclusion gap", "Established minus thin-file"),
    )
    for axis, (column, title, ylabel) in zip(axes.flat, series):
        axis.plot(results["threshold"], results[column], linewidth=2)
        axis.axvline(selected_threshold, color="tab:green", linestyle="--", label="Selected")
        axis.axvline(DEFAULT_RATE_CEILING, color="tab:red", linestyle=":", label="20% reference")
        axis.set(title=title, xlabel="Default-probability threshold", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    figure.suptitle("Policy-validation threshold trade-offs (weeks 70-75)")
    save_figure(figure, "05_threshold_tradeoffs.png")


def plot_coefficients(coefficients: pd.DataFrame, top_n: int = 15) -> None:
    risk = coefficients.nlargest(top_n, "coefficient")
    protective = coefficients.nsmallest(top_n, "coefficient")
    selected = pd.concat([protective, risk]).sort_values("coefficient")
    colors = np.where(selected["coefficient"] > 0, "#d95f02", "#1b9e77")
    figure, axis = plt.subplots(figsize=(11, 11))
    axis.barh(selected["display_name"], selected["coefficient"], color=colors)
    axis.axvline(0, color="black", linewidth=1)
    axis.set(
        title="Strongest standardized Logistic Regression coefficients",
        xlabel="Coefficient (right = more predicted risk, left = less predicted risk)",
    )
    axis.grid(axis="x", alpha=0.2)
    save_figure(figure, "06_top_coefficients.png")


def plot_subgroup_metrics(metrics: pd.DataFrame) -> None:
    holdout = metrics[metrics["split"] == "holdout"].copy()
    holdout["label"] = holdout["segment"].map(SEGMENT_LABELS)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (metric, title) in zip(
        axes.flat,
        (
            ("roc_auc", "ROC-AUC (higher is better)"),
            ("pr_auc", "PR-AUC (higher is better)"),
            ("logloss", "LogLoss (lower is better)"),
            ("brier", "Brier score (lower is better)"),
        ),
    ):
        bars = axis.bar(holdout["label"], holdout[metric], color=["#4c78a8", "#59a14f", "#f28e2b"])
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle("Holdout performance by applicant segment")
    save_figure(figure, "07_holdout_subgroup_metrics.png")


def markdown_metric_table(metrics: pd.DataFrame) -> str:
    """Format holdout metrics without requiring an additional Markdown dependency."""
    lines = [
        "| Segment | Applicants | Default rate | Mean predicted PD | ROC-AUC | PR-AUC | LogLoss | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {SEGMENT_LABELS[row.segment]} | {row.n:,} | {row.default_rate:.1%} | "
            f"{row.mean_predicted_default:.1%} | {row.roc_auc:.3f} | {row.pr_auc:.3f} | "
            f"{row.logloss:.3f} | {row.brier:.3f} |"
        )
    return "\n".join(lines)


def markdown_feature_table(features: pd.DataFrame) -> str:
    lines = [
        "| Feature | Coefficient | Odds ratio | Interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in features.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.coefficient:.3f} | {row.odds_ratio:.2f} | {row.direction} |"
        )
    return "\n".join(lines)


def write_summary(
    hyperparameters: pd.DataFrame,
    calibration_results: pd.DataFrame,
    metrics: pd.DataFrame,
    threshold_results: pd.DataFrame,
    holdout_policy: pd.DataFrame,
    coefficients: pd.DataFrame,
    selected_threshold: float,
    selected_method: str,
    split_summary: pd.DataFrame,
    output_features: int,
) -> None:
    """Write a concise, human-readable explanation of the model and its results."""
    selected_hyperparameters = hyperparameters[hyperparameters["selected"]].iloc[0]
    policy_validation = threshold_results[threshold_results["selected"]].iloc[0]
    holdout_selected = holdout_policy[holdout_policy["policy"] == "selected_threshold"].iloc[0]
    holdout_reference = holdout_policy[holdout_policy["policy"] == "20_percent_reference"].iloc[0]
    holdout_metrics = metrics[metrics["split"] == "holdout"]
    overall_holdout = holdout_metrics[holdout_metrics["segment"] == "all"].iloc[0]
    established_holdout = holdout_metrics[holdout_metrics["segment"] == "established"].iloc[0]
    thin_holdout = holdout_metrics[holdout_metrics["segment"] == "thin_file"].iloc[0]

    top_risk = coefficients.nlargest(5, "coefficient")
    top_protective = coefficients.nsmallest(5, "coefficient")

    summary = f"""# Logistic Regression Baseline

## What We Built

This model estimates each applicant's probability of default. It is the team's transparent baseline:
Random Forest and LightGBM should be compared against the same rows, time periods, and metrics before one
of them is selected for the second-look stage.

The model uses the completed aggregation, cleaning, and feature-engineering work from Tasks 1-4. The
preprocessing approach from Task 3 is fitted again inside this training workflow using weeks 0-63 only.
That keeps information from later weeks out of the medians, scaling values, and encoded category lists.
Task 5's EDA guides the subgroup reporting and the interpretation of thin-file performance.

## Final Application Pipeline

```mermaid
flowchart TD
    A[Applicant submits an application] --> B[Stage 1: Logistic Regression]
    B --> C[Estimated probability of default]
    C --> D{{Below the Stage 1 risk threshold?}}
    D -->|Yes| E[Approve the full requested loan]
    D -->|No| F[Route to the second-look model]
    F --> G[Best validated challenger: Random Forest OR LightGBM]
    G --> H[Second-look default-risk estimate]
    H --> I[Business policy layer]
    I --> J[Approve, offer a smaller amount, or deny]
```

Random Forest and LightGBM are alternatives for Stage 2; applicants do not pass through both models in
sequence. The current data supplies a default outcome, not a correct loan percentage. A smaller loan
offer therefore belongs in the later business-policy layer rather than being presented as a learned
regression target.

## Data and Time Split

| Split | Weeks | Applicants | Purpose |
| --- | --- | ---: | --- |
| Train | 0-63 | {int(split_summary.loc['train', 'rows']):,} | Fit preprocessing and Logistic Regression |
| Calibration | 64-69 | {int(split_summary.loc['calibration', 'rows']):,} | Choose regularization and fit calibration |
| Policy validation | 70-75 | {int(split_summary.loc['policy_validation', 'rows']):,} | Compare calibration and choose the provisional threshold |
| Holdout | 76-91 | {int(split_summary.loc['holdout', 'rows']):,} | Final labeled evaluation, scored after all choices were fixed |
| Score | Unlabeled test CSV | {int(split_summary.loc['score', 'rows']):,} | Produce predictions for the future application/demo pipeline |

- `target = 1` means the applicant defaulted; `target = 0` means the applicant repaid.
- `is_thin_file` is not a model input. It is retained only to compare outcomes for thin-file and
  established applicants.
- `external_coverage` is also excluded because Task 5 showed that its apparent relationship with default
  was caused by the changing mix of thin-file and established applicants, not a useful within-group
  relationship.
- Numeric `-1` values are treated as unknown, replaced with train-period medians, and accompanied by
  missing-value indicators. Numeric features are standardized and categorical features are one-hot
  encoded. The final model matrix contains {output_features:,} columns.
- No class weighting is used because reliable probabilities matter more than forcing equal class counts.

## Model Selection

Four L2 regularization strengths were compared on weeks 64-69. The selected value was
`C = {selected_hyperparameters['C']:g}` with LogLoss {selected_hyperparameters['logloss']:.4f} and
ROC-AUC {selected_hyperparameters['roc_auc']:.4f}. Smaller `C` values apply stronger regularization.

Raw, sigmoid, and isotonic probabilities were then compared on the later weeks 70-75. The selected
method was **{selected_method}**. Its policy-validation LogLoss was
{calibration_results.loc[calibration_results['selected'], 'logloss'].iloc[0]:.4f}, and its mean predicted
default rate was {calibration_results.loc[calibration_results['selected'], 'mean_predicted_default'].iloc[0]:.1%}
versus an actual rate of {calibration_results.loc[calibration_results['selected'], 'actual_default_rate'].iloc[0]:.1%}.

## Final Holdout Results

{markdown_metric_table(holdout_metrics)}

The overall holdout ROC-AUC of {overall_holdout['roc_auc']:.3f} measures ranking: how often a borrower
who defaulted receives a higher risk score than a borrower who repaid. LogLoss and Brier score judge the
quality of the probabilities themselves, so confident mistakes receive a larger penalty.

Performance is stronger for established applicants (AUC {established_holdout['roc_auc']:.3f}) than for
thin-file applicants (AUC {thin_holdout['roc_auc']:.3f}). That difference is expected from the EDA:
established applicants have bureau risk and payment-history fields, while thin-file applicants rely more
on education, employment, income, affordability, tax-registry, and prior-application information.

Overall calibration is close, but it is not equally close for both groups. The model predicts an average
default probability of {thin_holdout['mean_predicted_default']:.1%} for thin-file applicants versus their
actual {thin_holdout['default_rate']:.1%} default rate. For established applicants, it predicts
{established_holdout['mean_predicted_default']:.1%} versus an actual {established_holdout['default_rate']:.1%}.
The final two-stage comparison should therefore examine subgroup calibration, not only the overall score.

## Provisional Stage 1 Policy

The selected probability threshold is **{selected_threshold:.1%}**. On policy-validation weeks, it sent
{policy_validation['approval_rate']:.1%} of applicants directly to a full-loan approval, produced an observed default rate of
{policy_validation['approved_default_rate']:.1%} among approvals, and had an inclusion gap of
{policy_validation['inclusion_gap']:.1%}. A positive inclusion gap means established applicants who
would repay reached an immediate full-loan approval more often than thin-file applicants who would repay.
The applicants not approved at Stage 1 are routed to the second-look model rather than treated as final
denials. Reducing this large routing gap is one of the main jobs of Stage 2.

On the untouched holdout period, that same threshold produced:

| Policy | Immediate full-loan rate | Default rate among approved | Profit per applicant | Stage 1 inclusion gap |
| --- | ---: | ---: | ---: | ---: |
| Selected threshold ({selected_threshold:.1%}) | {holdout_selected['approval_rate']:.1%} | {holdout_selected['approved_default_rate']:.1%} | {holdout_selected['profit_per_applicant']:.3f} | {holdout_selected['inclusion_gap']:.1%} |
| 20% PD reference | {holdout_reference['approval_rate']:.1%} | {holdout_reference['approved_default_rate']:.1%} | {holdout_reference['profit_per_applicant']:.3f} | {holdout_reference['inclusion_gap']:.1%} |

Profit is reported in normalized units: `+1` for an approved applicant who repays, `-4` for an approved
applicant who defaults, and `0` for an applicant who is not approved at Stage 1. This creates the 20%
break-even reference used in the EDA. These are comparison units, not dollar estimates. The team can
replace them when the final business cost assumptions and Inclusive Profit Score penalty are agreed.

The threshold is provisional because Stage 1 applicants above it will ultimately receive a second look,
not an automatic final denial. The complete two-stage policy should be tuned only after the Random Forest
and LightGBM results are available.

## Strongest Coefficients

Positive coefficients move predictions toward default; negative coefficients move predictions toward
repayment. Numeric coefficients use standardized inputs, while categorical coefficients compare an
encoded category with the model intercept and other categories. Masked category codes are preserved
because the source data does not provide their real-world names.

### Higher Predicted Risk

{markdown_feature_table(top_risk)}

### Lower Predicted Risk

{markdown_feature_table(top_protective)}

Coefficients describe the model's fitted associations, not proof that changing a feature would cause a
different repayment outcome. Correlated fields can also divide the same signal across several
coefficients.

## Files Produced

- `models/artifacts/logistic_pipeline.joblib`: preprocessing, Logistic Regression, calibration, and the
  provisional Stage 1 threshold.
- `data/predictions/logistic_score_predictions.csv`: local predictions for the unlabeled scoring set.
- `reports/logistic/model_metrics.csv`: model metrics overall and by applicant segment.
- `reports/logistic/hyperparameter_results.csv`: regularization comparison.
- `reports/logistic/calibration_results.csv`: raw versus calibrated probabilities.
- `reports/logistic/threshold_results.csv`: policy trade-offs across thresholds.
- `reports/logistic/holdout_policy.csv`: selected and 20% reference policies on holdout.
- `reports/logistic/coefficients.csv`: every fitted coefficient and odds ratio.
- `reports/logistic/figures/`: ROC, precision-recall, calibration, score distribution, policy, coefficient,
  and subgroup figures.

## Reproducing the Analysis

From the repository root, after the feature datasets have been generated:

```bash
python models/train_logistic.py
```

The random seed, time boundaries, preprocessing rules, candidate regularization values, calibration
comparison, and policy assumptions are all defined in the script so another teammate can reproduce the
same results.

To reuse the saved model without retraining it:

```bash
python models/predict_logistic.py --input data/features/test_features.csv
```
"""
    (REPORT_DIR / "LOGISTIC_REGRESSION_SUMMARY.md").write_text(summary, encoding="utf-8")


def run(train_path: Path, test_path: Path) -> None:
    """Execute the complete Logistic Regression workflow."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {train_path}...")
    train_data = pd.read_csv(train_path, low_memory=False)
    print(f"Loading {test_path}...")
    test_data = pd.read_csv(test_path, low_memory=False)

    splits = make_splits(train_data, test_data, verbose=False)
    tune = splits["tune"]
    calibration_frame = tune[tune["WEEK_NUM"] <= CALIBRATION_END_WEEK].copy()
    policy_frame = tune[tune["WEEK_NUM"] > CALIBRATION_END_WEEK].copy()
    if calibration_frame.empty or policy_frame.empty:
        raise ValueError("The tune split must contain both calibration and policy-validation weeks")

    split_summary = pd.DataFrame(
        [
            {"split": "train", "rows": len(splits["train"]), "weeks": "0-63"},
            {
                "split": "calibration",
                "rows": len(calibration_frame),
                "weeks": "64-69",
            },
            {
                "split": "policy_validation",
                "rows": len(policy_frame),
                "weeks": "70-75",
            },
            {"split": "holdout", "rows": len(splits["holdout"]), "weeks": "76-91"},
            {"split": "score", "rows": len(splits["score"]), "weeks": "unlabeled"},
        ]
    ).set_index("split")
    split_summary.to_csv(REPORT_DIR / "split_summary.csv")

    score_case_ids = splits["score"]["case_id"].copy()
    train_features, train_target, _ = model_inputs(splits["train"])
    calibration_features, calibration_target, calibration_segment = model_inputs(calibration_frame)
    policy_features, policy_target, policy_segment = model_inputs(policy_frame)
    holdout_features, holdout_target, holdout_segment = model_inputs(splits["holdout"])
    score_features, _, score_segment = model_inputs(splits["score"])

    del train_data, test_data, tune, calibration_frame, policy_frame, splits

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(train_features)
    print(
        f"Fitting preprocessing on {len(train_features):,} applicants: "
        f"{len(numeric_columns)} numeric and {len(categorical_columns)} categorical inputs"
    )
    train_matrix = transform_float32(preprocessor, train_features, fit=True)
    calibration_matrix = transform_float32(preprocessor, calibration_features)
    policy_matrix = transform_float32(preprocessor, policy_features)
    feature_names = preprocessor.get_feature_names_out()
    print(f"Transformed feature count: {len(feature_names)}")
    del train_features, calibration_features, policy_features

    model, hyperparameter_results = fit_logistic_candidates(
        train_matrix,
        train_target,
        calibration_matrix,
        calibration_target,
    )
    hyperparameter_results.to_csv(REPORT_DIR / "hyperparameter_results.csv", index=False)

    (
        calibration_method,
        calibrator,
        calibration_results,
        policy_probabilities,
    ) = select_calibration(
        model,
        calibration_matrix,
        calibration_target,
        policy_matrix,
        policy_target,
    )
    calibration_results.to_csv(REPORT_DIR / "calibration_results.csv", index=False)
    print(f"Selected calibration: {calibration_method}")

    selected_threshold, threshold_results = select_threshold(
        policy_target,
        policy_probabilities,
        policy_segment,
    )
    threshold_results.to_csv(REPORT_DIR / "threshold_results.csv", index=False)
    print(f"Selected Stage 1 threshold: {selected_threshold:.1%}")

    policy_metric_rows = metric_rows(
        policy_target,
        policy_probabilities,
        policy_segment,
        "policy_validation",
    )
    del train_matrix, calibration_matrix, policy_matrix

    print("All model choices are fixed. Scoring the holdout period once...")
    holdout_matrix = transform_float32(preprocessor, holdout_features)
    holdout_raw_probabilities = model.predict_proba(holdout_matrix)[:, 1]
    holdout_probabilities = apply_calibrator(
        calibration_method,
        calibrator,
        holdout_raw_probabilities,
    )
    final_metric_rows = policy_metric_rows + metric_rows(
        holdout_target,
        holdout_probabilities,
        holdout_segment,
        "holdout",
    )
    metrics = pd.DataFrame(final_metric_rows)
    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)

    holdout_policy = pd.DataFrame(
        [
            {
                "policy": "selected_threshold",
                **evaluate_policy(
                    holdout_target,
                    holdout_probabilities,
                    holdout_segment,
                    selected_threshold,
                ),
            },
            {
                "policy": "20_percent_reference",
                **evaluate_policy(
                    holdout_target,
                    holdout_probabilities,
                    holdout_segment,
                    DEFAULT_RATE_CEILING,
                ),
            },
        ]
    )
    holdout_policy.to_csv(REPORT_DIR / "holdout_policy.csv", index=False)

    coefficients = coefficient_table(model, feature_names)
    coefficients.to_csv(REPORT_DIR / "coefficients.csv", index=False)

    plot_roc(holdout_target, holdout_probabilities, holdout_segment)
    plot_precision_recall(holdout_target, holdout_probabilities, holdout_segment)
    plot_calibration(holdout_target, holdout_probabilities, holdout_segment)
    plot_probability_distributions(holdout_target, holdout_probabilities, holdout_segment)
    plot_threshold_tradeoffs(threshold_results, selected_threshold)
    plot_coefficients(coefficients)
    plot_subgroup_metrics(metrics)
    del holdout_matrix, holdout_features

    print("Scoring the unlabeled test applications...")
    score_matrix = transform_float32(preprocessor, score_features)
    score_raw_probabilities = model.predict_proba(score_matrix)[:, 1]
    score_probabilities = apply_calibrator(
        calibration_method,
        calibrator,
        score_raw_probabilities,
    )
    score_predictions = pd.DataFrame(
        {
            "case_id": score_case_ids.to_numpy(),
            "predicted_default_probability": score_probabilities,
            "stage1_route": np.where(
                score_probabilities < selected_threshold,
                "approve_full_loan",
                "send_to_second_look",
            ),
            "is_thin_file": score_segment.to_numpy(),
        }
    )
    score_predictions.to_csv(PREDICTION_PATH, index=False, float_format="%.8f")

    artifact = {
        "model_name": "logistic_regression_baseline",
        "target_definition": "1=defaulted, 0=repaid",
        "preprocessor": preprocessor,
        "model": model,
        "calibration_method": calibration_method,
        "calibrator": calibrator,
        "stage1_threshold": selected_threshold,
        "excluded_model_features": EXCLUDED_MODEL_FEATURES,
        "feature_names": feature_names,
        "train_weeks": "0-63",
        "calibration_weeks": "64-69",
        "policy_validation_weeks": "70-75",
        "holdout_weeks": "76-91",
        "random_seed": SEED,
        "policy_assumptions": {
            "repaid_approval_value": REPAID_APPROVAL_VALUE,
            "default_approval_cost": DEFAULT_APPROVAL_COST,
            "default_rate_ceiling": DEFAULT_RATE_CEILING,
        },
    }
    joblib.dump(artifact, ARTIFACT_PATH, compress=3)

    run_config = {
        "train_path": portable_path(train_path),
        "test_path": portable_path(test_path),
        "rows": {name: int(row["rows"]) for name, row in split_summary.iterrows()},
        "input_features": int(len(numeric_columns) + len(categorical_columns)),
        "output_features": int(len(feature_names)),
        "selected_C": float(hyperparameter_results.loc[hyperparameter_results["selected"], "C"].iloc[0]),
        "calibration_method": calibration_method,
        "stage1_threshold": selected_threshold,
    }
    (REPORT_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    write_summary(
        hyperparameter_results,
        calibration_results,
        metrics,
        threshold_results,
        holdout_policy,
        coefficients,
        selected_threshold,
        calibration_method,
        split_summary,
        len(feature_names),
    )

    print("\nLogistic Regression workflow complete.")
    print(f"Summary: {REPORT_DIR / 'LOGISTIC_REGRESSION_SUMMARY.md'}")
    print(f"Model artifact: {ARTIFACT_PATH}")
    print(f"Local score predictions: {PREDICTION_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    arguments = parser.parse_args()
    run(arguments.train, arguments.test)
