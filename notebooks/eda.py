"""
Task 5: Exploratory data analysis.

Run this after features.py. It reads the complete training feature dataset and writes eight figures
to reports/eda/figures. Figures 1-7 use the full 1,000,000-row training set; figure 8 ranks thin-file
signal strength across all feature types (including education encoded as an ordinal) on the full set of
thin-file applicants and prints a validation check for the education signal. The findings from those
figures are documented in reports/eda/EDA_SUMMARY.md.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "data" / "features" / "train_features.csv"
OUTPUT_DIR = ROOT_DIR / "reports" / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET = "target"
THIN_FILE = "is_thin_file"
EDUCATION = "education_927M"
MISSING_VALUE = -1
SAMPLE_SIZE = 200_000
RANDOM_STATE = 42
BREAK_EVEN_RATE = 0.20
# Figure 8: how many distinct numeric feature families to show alongside the education ordinal.
THIN_FILE_NUMERIC_COUNT = 14
# Validation check for the education signal (printed, not plotted).
TIME_SPLIT_WEEK = 75          # train: weeks 0-75, validation: weeks 76-91
PERMUTATION_SHUFFLES = 1_000  # label shuffles for the permutation test

COLORS = {"overall": "#345995", "established": "#2A9D8F", "thin": "#E76F51"}
CATEGORICAL_FEATURES = ["education_927M", "maritalstatus_703M", "description_5085714M"]
AVAILABILITY_FEATURES = [
    "riskassesment_940T",
    "pmts_dpdvalue_108P_mean",
    "amount_4527230A_mean",
    "credamount_590A_mean",
    "mainoccupationinc_384A",
    "days_employed_700P",
    "credit_to_income",
    "education_927M",
]
NUMERIC_DISTRIBUTIONS = [
    "riskassesment_940T",
    "mainoccupationinc_384A",
    "credamount_770A",
    "credit_to_income",
    "age_in_years",
    "pmts_dpdvalue_108P_mean",
]
GROUPED_RATE_FEATURES = {
    "Overall": [
        "riskassesment_940T",
        "pmts_dpdvalue_108P_mean",
        "credit_to_income",
        "employment_share_of_life",
    ],
    "Thin-file": [
        "amount_4527230A_mean",
        "days_employed_700P",
        "income_total",
        "credit_to_income",
    ],
}

DISPLAY_NAMES = {
    "education_927M": "Education category",
    "dpd_bucket": "Days-past-due risk bucket",
    "riskassesment_940T": "Risk assessment score",
    "mainoccupationinc_384A": "Applicant income",
    "credamount_770A": "Requested credit",
    "credit_to_income": "Credit-to-income ratio",
    "age_in_years": "Age (years)",
    "pmts_dpdvalue_108P_mean": "Average days past due",
    "pmts_dpdvalue_108P_median": "Median days past due",
    "pmts_dpdvalue_108P_max": "Maximum days past due",
    "credamount_770A_mean": "Average past bureau credit",
    "credamount_770A_median": "Median past bureau credit",
    "credamount_770A_max": "Maximum past bureau credit",
    "overdueamountmax_950A_mean": "Average maximum overdue amount",
    "overdueamountmax_950A_median": "Median maximum overdue amount",
    "numberofqueries_146L": "Credit queries",
    "amount_4527230A_mean": "Average tax-registry amount",
    "amount_4527230A_median": "Median tax-registry amount",
    "amount_4527230A_max": "Maximum tax-registry amount",
    "credamount_590A_mean": "Average prior application amount",
    "days_employed_700P": "Employment duration",
    "empl_employedtotal_800L_mean": "Average employment history",
    "empl_employedtotal_800L_median": "Median employment history",
    "empl_employedtotal_800L_max": "Maximum employment history",
    "mainoccupationinc_384A_mean": "Average household occupation income",
    "mainoccupationinc_384A_median": "Median household occupation income",
    "employment_share_of_life": "Employment share of life",
    "requested_vs_past_credit": "Requested-to-past-credit ratio",
    "requested_vs_prior_application": "Requested-to-prior-app ratio",
    "annuity_780A": "Loan annuity",
    "annuity_to_income": "Annuity-to-income ratio",
    "credit_to_annuity": "Credit-to-annuity ratio",
    "income_vs_tax": "Income-to-tax ratio",
    "months_since_last_application": "Months since last application",
    "income_total": "Total household income",
    "n_unknown": "Number of unknown values",
    "is_thin_file": "Thin-file indicator",
    "external_coverage": "Available data sources",
    "classificationofcontr_13M_54ddc605": "Bureau contract class count (54ddc605)",
    "classificationofcontr_13M_fe64f125": "Bureau contract class count (fe64f125)",
    "classificationofcontr_13M_6f5666c7": "Bureau contract class count (6f5666c7)",
    "classificationofcontr_13M_d8bf6dd7": "Bureau contract class count (d8bf6dd7)",
    "classificationofcontr_13M_01851564": "Bureau contract class count (01851564)",
    "classificationofcontr_13M_ba38dde4": "Bureau contract class count (ba38dde4)",
}


def clean_numeric(values: pd.Series) -> pd.Series:
    """Replace the pipeline's -1 unknown marker with NaN for analysis."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.mask(numeric.eq(MISSING_VALUE))


def display_name(feature: str) -> str:
    """Return a readable feature name while preserving the original name in calculations."""
    return DISPLAY_NAMES.get(feature, feature.replace("_", " "))


def signal_family(feature: str) -> str:
    """Group mean, median, and maximum versions so charts show distinct ideas."""
    prefix_groups = {
        "pmts_dpdvalue_108P": "days_past_due",
        "credamount_770A_": "past_bureau_credit",
        "overdueamountmax_950A": "overdue_amount",
        "amount_4527230A": "tax_registry_amount",
        "empl_employedtotal_800L": "employment_history",
        "mainoccupationinc_384A": "occupation_income",
        "credamount_590A": "prior_application_amount",
        "pmts_overdue_1140A": "payment_overdue",
        "classificationofcontr_13M": "contract_classification",
    }
    for prefix, family in prefix_groups.items():
        if feature.startswith(prefix):
            return family
    return feature


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric/boolean feature columns, excluding ids and one-hot indicator families."""
    excluded = {"case_id", "MONTH", "WEEK_NUM", TARGET}
    return [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded
        and not column.startswith(
            ("status_219L_", "housingtype_772M_", "name_4527232M_")
        )
    ]


def score_features(frame: pd.DataFrame, features: list[str], segment: str) -> list[dict]:
    """Univariate separation (2 x |AUC - 0.5|) for each feature within one applicant group."""
    rows = []
    for feature in features:
        values = clean_numeric(frame[feature])
        valid = values.notna()
        auc = np.nan
        if valid.sum() >= 500 and values.loc[valid].nunique() > 1 and frame.loc[
            valid, TARGET
        ].nunique() > 1:
            auc = roc_auc_score(frame.loc[valid, TARGET], values.loc[valid])
        rows.append(
            {
                "segment": segment,
                "feature": feature,
                "display_name": display_name(feature),
                "auc": auc,
                "signal_strength": 2 * abs(auc - 0.5) if pd.notna(auc) else np.nan,
            }
        )
    return rows


def numeric_signal(data: pd.DataFrame) -> pd.DataFrame:
    """Rank numeric features using univariate AUC overall and within each segment."""
    sample = data.sample(min(SAMPLE_SIZE, len(data)), random_state=RANDOM_STATE)
    features = numeric_feature_columns(sample)
    groups = {
        "Overall": sample,
        "Established": sample[sample[THIN_FILE].eq(0)],
        "Thin-file": sample[sample[THIN_FILE].eq(1)],
    }
    rows = []
    for segment, group in groups.items():
        rows.extend(score_features(group, features, segment))
    return pd.DataFrame(rows)


def diverse_signal(signal: pd.DataFrame, segment: str, count: int) -> pd.DataFrame:
    """Select the strongest distinct feature families for a readable ranking."""
    ranked = signal[signal["segment"] == segment].dropna(subset=["signal_strength"]).copy()
    ranked["family"] = ranked["feature"].map(signal_family)
    return (
        ranked.sort_values("signal_strength", ascending=False)
        .drop_duplicates("family")
        .head(count)
    )


def education_ordinal_values(
    frame: pd.DataFrame, order: list[str] | None = None
) -> tuple[pd.Series, list[str]]:
    """Encode anonymized education as an ordinal (0..k-1) ordered by default rate.

    Figure 6 covers numeric features only; education is categorical. Ordering the categories by
    default rate turns them into a single ordinal feature that can be scored on the same scale.
    An explicit `order` can be supplied so an order learned on one split is applied cold to another.
    """
    if order is None:
        order = (
            frame.groupby(EDUCATION, observed=True)[TARGET]
            .mean()
            .sort_values()
            .index.tolist()
        )
    mapping = {category: rank for rank, category in enumerate(order)}
    return frame[EDUCATION].map(mapping).astype(float), order


def thin_file_all_signal(data: pd.DataFrame) -> pd.DataFrame:
    """Separation for every numeric feature plus the education ordinal, on all thin-file applicants.

    Unlike figure 6 (which samples), this uses the full thin-file population because education is the
    headline thin-file finding and we want the ranking computed on every available applicant.
    """
    thin = data[data[THIN_FILE].eq(1)]
    features = numeric_feature_columns(thin)
    signal = pd.DataFrame(score_features(thin, features, "Thin-file"))
    signal["kind"] = "numeric"

    values, _ = education_ordinal_values(thin)
    valid = values.notna()
    education_auc = roc_auc_score(thin.loc[valid, TARGET], values.loc[valid])
    education_row = {
        "segment": "Thin-file",
        "feature": EDUCATION,
        "display_name": "Education group (A->E)",
        "auc": education_auc,
        "signal_strength": 2 * abs(education_auc - 0.5),
        "kind": "education",
    }
    return pd.concat([signal, pd.DataFrame([education_row])], ignore_index=True)


def thin_file_signal_ranking(all_signal: pd.DataFrame) -> pd.DataFrame:
    """Keep the education ordinal plus the strongest distinct numeric families for the chart."""
    numeric = all_signal[all_signal["kind"] == "numeric"].dropna(subset=["signal_strength"]).copy()
    numeric["family"] = numeric["feature"].map(signal_family)
    numeric = (
        numeric.sort_values("signal_strength", ascending=False)
        .drop_duplicates("family")
        .head(THIN_FILE_NUMERIC_COUNT)
    )
    education = all_signal[all_signal["kind"] == "education"]
    return pd.concat([education, numeric], ignore_index=True)


def validate_education_signal(data: pd.DataFrame) -> dict:
    """Confirm education's signal is real, not an artifact of defining the ordinal on the same data.

    Runs two checks and returns their results for printing:
      1. Time-based holdout: derive the ordinal order from train-thin only (weeks 0-TIME_SPLIT_WEEK),
         then apply it cold to the later validation weeks. A stable separation means no leakage.
      2. Permutation test: shuffle the education labels PERMUTATION_SHUFFLES times and recompute AUC.
         A real signal sits far above anything the shuffles produce.
    """
    thin = data[data[THIN_FILE].eq(1)]
    full_values, _ = education_ordinal_values(thin)
    observed_auc = roc_auc_score(thin[TARGET], full_values)

    train = thin[thin["WEEK_NUM"] <= TIME_SPLIT_WEEK]
    validation = thin[thin["WEEK_NUM"] > TIME_SPLIT_WEEK]
    train_values, order = education_ordinal_values(train)
    validation_values, _ = education_ordinal_values(validation, order)
    train_sep = 2 * abs(roc_auc_score(train[TARGET], train_values) - 0.5)
    validation_sep = 2 * abs(roc_auc_score(validation[TARGET], validation_values) - 0.5)

    rng = np.random.default_rng(RANDOM_STATE)
    targets = thin[TARGET].to_numpy()
    encoded = full_values.to_numpy()
    null_aucs = np.array(
        [roc_auc_score(targets, rng.permutation(encoded)) for _ in range(PERMUTATION_SHUFFLES)]
    )
    exceed = int((null_aucs >= observed_auc).sum())
    return {
        "observed_auc": observed_auc,
        "observed_separation": 2 * abs(observed_auc - 0.5),
        "train_separation": train_sep,
        "validation_separation": validation_sep,
        "null_max_auc": float(null_aucs.max()),
        "p_value": (exceed + 1) / (PERMUTATION_SHUFFLES + 1),
    }


def category_aliases(data: pd.DataFrame) -> pd.DataFrame:
    """Assign neutral aliases without pretending to know anonymized category meanings."""
    rows = []
    for feature in CATEGORICAL_FEATURES:
        overall = (
            data.groupby(feature, dropna=False, observed=True)[TARGET]
            .agg(applicants="size", default_rate="mean")
            .reset_index()
            .rename(columns={feature: "category"})
        )
        overall["category"] = overall["category"].astype(str)

        if feature == "education_927M":
            overall = overall.sort_values("default_rate")
            labels = [f"Education group {letter}" for letter in "ABCDE"]
        elif feature == "description_5085714M":
            missing = overall[overall["category"] == "MISSING"]
            known = overall[overall["category"] != "MISSING"].sort_values("default_rate")
            overall = pd.concat([missing, known], ignore_index=True)
            labels = ["No bureau description"] + [
                f"Bureau description group {letter}" for letter in "ABCDEF"
            ]
        else:
            overall = overall.sort_values("category")
            labels = [f"Marital-status group {letter}" for letter in "ABCD"]

        for label, row in zip(labels, overall.itertuples()):
            rows.append({"feature": feature, "category": row.category, "display_label": label})
    return pd.DataFrame(rows)


def categorical_rates(data: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    """Calculate category sizes and default rates overall and by segment."""
    groups = {
        "Overall": data,
        "Established": data[data[THIN_FILE].eq(0)],
        "Thin-file": data[data[THIN_FILE].eq(1)],
    }
    rate_tables = []
    for feature in CATEGORICAL_FEATURES:
        for segment, group in groups.items():
            rates = (
                group.groupby(feature, dropna=False, observed=True)[TARGET]
                .agg(applicants="size", defaults="sum", default_rate="mean")
                .reset_index()
                .rename(columns={feature: "category"})
            )
            rates.insert(0, "segment", segment)
            rates.insert(0, "feature", feature)
            rate_tables.append(rates)

    rates = pd.concat(rate_tables, ignore_index=True)
    rates["category"] = rates["category"].astype(str)
    return rates.merge(aliases, on=["feature", "category"], how="left")


def feature_availability(data: pd.DataFrame) -> pd.DataFrame:
    """Compare which useful inputs are present for established and thin-file applicants."""
    rows = []
    for segment, group in data.groupby("segment", observed=True):
        for feature in AVAILABILITY_FEATURES:
            values = group[feature]
            if is_numeric_dtype(values):
                available = values.notna() & values.ne(MISSING_VALUE)
            else:
                available = values.notna() & values.astype("string").str.upper().ne("MISSING")
            rows.append(
                {
                    "segment": segment,
                    "feature": feature,
                    "display_name": display_name(feature),
                    "availability_rate": available.mean(),
                }
            )
    return pd.DataFrame(rows)


def feature_group_default_rates(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate default rates across ten approximately equal-sized value groups."""
    rows = []
    segments = {
        "Overall": data,
        "Thin-file": data[data[THIN_FILE].eq(1)],
    }
    for segment, features in GROUPED_RATE_FEATURES.items():
        segment_data = segments[segment]
        for feature in features:
            values = clean_numeric(segment_data[feature])
            valid = values.notna()
            grouped = pd.DataFrame(
                {"value": values[valid], TARGET: segment_data.loc[valid, TARGET]}
            )
            grouped["value_group"] = pd.qcut(
                grouped["value"], 10, labels=False, duplicates="drop"
            )
            summary = (
                grouped.groupby("value_group", observed=True)[TARGET]
                .agg(applicants="size", default_rate="mean")
                .reset_index()
            )
            summary.insert(0, "feature", feature)
            summary.insert(0, "segment", segment)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def coverage_by_segment(data: pd.DataFrame) -> pd.DataFrame:
    """Default rate by data-source coverage, overall and within each segment (for figure 5b)."""
    groups = {
        "Overall": data,
        "Thin-file": data[data[THIN_FILE].eq(1)],
        "Established": data[data[THIN_FILE].eq(0)],
    }
    tables = []
    for segment, group in groups.items():
        rates = (
            group.groupby("external_coverage", observed=True)[TARGET]
            .agg(applicants="size", default_rate="mean")
            .reset_index()
        )
        rates.insert(0, "segment", segment)
        tables.append(rates)
    return pd.concat(tables, ignore_index=True)


def save_figure(figure: plt.Figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_target_and_segments(data: pd.DataFrame, segments: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    segment_rates = segments.set_index("segment")["default_rate"]
    labels = ["Overall", "Established", "Thin-file"]
    rates = [data[TARGET].mean(), segment_rates["Established"], segment_rates["Thin-file"]]
    bars = axes[0].bar(labels, rates, color=list(COLORS.values()))
    axes[0].axhline(BREAK_EVEN_RATE, color="gray", linestyle="--", label="20% simple break-even")
    axes[0].set(title="Default rate by applicant segment", ylabel="Default rate", ylim=(0, 0.34))
    axes[0].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].legend(frameon=False)
    axes[0].bar_label(bars, labels=[f"{value:.1%}" for value in rates], padding=3)

    shares = segments.set_index("segment")["applicant_share"]
    bars = axes[1].bar(
        ["Established", "Thin-file"],
        shares,
        color=[COLORS["established"], COLORS["thin"]],
    )
    axes[1].set(title="Applicant mix", ylabel="Share of applications", ylim=(0, 0.8))
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].bar_label(bars, labels=[f"{value:.1%}" for value in shares], padding=3)
    save_figure(figure, "01_target_and_segments.png")


def plot_weekly_rates(weekly: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11, 5))
    for column, label, color in [
        ("overall_default_rate", "Overall", COLORS["overall"]),
        ("established_default_rate", "Established", COLORS["established"]),
        ("thin_file_default_rate", "Thin-file", COLORS["thin"]),
    ]:
        axis.plot(
            weekly["WEEK_NUM"],
            weekly[column].rolling(4, min_periods=1).mean(),
            label=f"{label} (4-week average)",
            color=color,
            linewidth=2,
        )
    axis.axhline(BREAK_EVEN_RATE, color="gray", linestyle="--", label="20% simple break-even")
    axis.set(title="Default rate changes across training weeks", xlabel="Training week",
             ylabel="Default rate")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(frameon=False, ncol=2)
    save_figure(figure, "02_default_rate_over_time.png")


def plot_feature_availability(availability: pd.DataFrame) -> None:
    pivot = availability.pivot(
        index="display_name", columns="segment", values="availability_rate"
    ).reindex([display_name(feature) for feature in AVAILABILITY_FEATURES])
    positions = np.arange(len(pivot))
    figure, axis = plt.subplots(figsize=(11, 6))
    established = axis.barh(positions - 0.18, pivot["Established"], height=0.34,
                            color=COLORS["established"], label="Established")
    thin = axis.barh(positions + 0.18, pivot["Thin-file"], height=0.34,
                    color=COLORS["thin"], label="Thin-file")
    axis.set_yticks(positions, pivot.index)
    axis.invert_yaxis()
    axis.set(title="Useful feature availability by applicant segment",
             xlabel="Applicants with a usable value", xlim=(0, 1.08))
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    axis.bar_label(established, labels=[f"{value:.0%}" for value in pivot["Established"]],
                   padding=3, fontsize=8)
    axis.bar_label(thin, labels=[f"{value:.0%}" for value in pivot["Thin-file"]],
                   padding=3, fontsize=8)
    save_figure(figure, "03_feature_availability.png")


def plot_numeric_distributions(data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for axis, feature in zip(axes.flat, NUMERIC_DISTRIBUTIONS):
        values = clean_numeric(data[feature]).dropna()
        lower, upper = values.quantile([0.01, 0.99])
        display_values = values[values.between(lower, upper)]
        median = values.median()
        axis.hist(display_values, bins=45, color=COLORS["overall"], alpha=0.85)
        axis.axvline(median, color=COLORS["thin"], linestyle="--")
        axis.text(
            0.98,
            0.92,
            f"Median: {median:,.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        axis.set(title=display_name(feature), ylabel="Applicants")
    figure.suptitle("Key numeric distributions (display limited to p01-p99)", y=1.02)
    save_figure(figure, "04_numeric_distributions.png")


def plot_education_and_coverage(rates: pd.DataFrame, coverage: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    education = rates[
        (rates["feature"] == "education_927M")
        & rates["segment"].isin(["Overall", "Thin-file"])
    ]
    overall = education[education["segment"] == "Overall"].set_index("display_label")[
        "default_rate"
    ].sort_values()
    thin = education[education["segment"] == "Thin-file"].set_index("display_label")[
        "default_rate"
    ]
    positions = np.arange(len(overall))
    overall_bars = axes[0].barh(positions - 0.18, overall, height=0.34,
                                label="Overall", color=COLORS["overall"])
    thin_bars = axes[0].barh(positions + 0.18, thin.reindex(overall.index), height=0.34,
                             label="Thin-file", color=COLORS["thin"])
    axes[0].set_yticks(positions, overall.index)
    axes[0].set(title="Default rate by anonymized education group", xlabel="Default rate")
    axes[0].xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[0].legend(frameon=False)
    axes[0].bar_label(overall_bars, labels=[f"{value:.1%}" for value in overall],
                      padding=3, fontsize=8)
    axes[0].bar_label(thin_bars, labels=[f"{value:.1%}" for value in thin.reindex(overall.index)],
                      padding=3, fontsize=8)

    coverage_bars = axes[1].bar(
        coverage["external_coverage"].astype(str),
        coverage["default_rate"],
        color=COLORS["overall"],
    )
    axes[1].set(title="Default rate falls with more available data sources",
                xlabel="Available sources (bureau, prior applications, tax)",
                ylabel="Default rate", ylim=(0, 0.30))
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axes[1].bar_label(
        coverage_bars,
        labels=[f"{row.default_rate:.1%}\nn={row.applicants:,}" for row in coverage.itertuples()],
        padding=3,
        fontsize=8,
    )
    save_figure(figure, "05_education_and_coverage.png")


def plot_coverage_simpsons(coverage_segments: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11, 6.5))
    styles = {
        "Overall": {"color": "gray", "linestyle": "--", "label": "Overall (misleading aggregate)"},
        "Thin-file": {"color": COLORS["thin"], "linestyle": "-", "label": "Thin-file applicants"},
        "Established": {"color": COLORS["established"], "linestyle": "-",
                        "label": "Established applicants"},
    }
    for segment, style in styles.items():
        line = coverage_segments[coverage_segments["segment"] == segment].sort_values(
            "external_coverage"
        )
        axis.plot(line["external_coverage"], line["default_rate"], marker="o", linewidth=2.2,
                  markersize=8, **style)
        if segment != "Overall":
            last = line.iloc[-1]
            axis.annotate(
                f"  {last['default_rate']:.1%} {segment.split('-')[0].lower()}",
                xy=(last["external_coverage"], last["default_rate"]),
                va="center", fontsize=11, fontweight="bold", color=style["color"],
            )
    axis.set(
        title="Simpson's paradox: coverage looks informative overall, but isn't\n"
        "Within each segment the default rate is flat as coverage increases",
        xlabel="Available data sources  (0 = none  .  1 = one of bureau / prior apps / tax  "
        ".  3 = all three)",
        ylabel="Default rate",
        xticks=[0, 1, 2, 3],
        xlim=(-0.3, 3.6),
    )
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(frameon=False, loc="upper right")
    save_figure(figure, "05b_coverage_simpsons_paradox.png")


def plot_numeric_signal(signal: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for axis, segment, color in [
        (axes[0], "Overall", COLORS["overall"]),
        (axes[1], "Thin-file", COLORS["thin"]),
    ]:
        strongest = diverse_signal(signal, segment, count=8).sort_values("signal_strength").copy()
        strongest["chart_label"] = strongest["display_name"] + np.where(
            strongest["auc"] >= 0.5, " (higher = more risk)", " (higher = less risk)"
        )
        bars = axis.barh(strongest["chart_label"], strongest["signal_strength"], color=color)
        axis.set(title=f"Strongest numeric signals: {segment}",
                 xlabel="Single-feature separation (0 = none, 1 = perfect)", xlim=(0, 0.68))
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in strongest["signal_strength"]],
                       padding=3, fontsize=8)
    save_figure(figure, "06_numeric_feature_signal.png")


def plot_feature_group_rates(grouped_rates: pd.DataFrame, reference_rates: dict[str, float]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=True)
    for row, (segment, features) in enumerate(GROUPED_RATE_FEATURES.items()):
        color = COLORS["overall"] if segment == "Overall" else COLORS["thin"]
        for column, feature in enumerate(features):
            axis = axes[row, column]
            grouped = grouped_rates[
                (grouped_rates["segment"] == segment)
                & (grouped_rates["feature"] == feature)
            ]
            axis.plot(
                grouped["value_group"] + 1,
                grouped["default_rate"],
                marker="o",
                color=color,
                linewidth=2,
            )
            axis.axhline(reference_rates[segment], color="gray", linestyle="--", linewidth=1)
            axis.set(
                title=f"{segment}: {display_name(feature)}",
                xlabel="Value group\n(lowest 10% to highest 10%)",
                xticks=range(1, len(grouped) + 1),
                ylim=(0, 0.60),
            )
            if column == 0:
                axis.set_ylabel("Default rate")
            axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    figure.suptitle("Default rates from lowest to highest feature values", y=1.03)
    figure.text(
        0.5,
        0.005,
        "Each feature is divided into 10 approximately equal-sized applicant groups; "
        "the dashed line is the segment average.",
        ha="center",
        fontsize=10,
    )
    save_figure(figure, "07_binned_default_rates.png")


def plot_thin_file_signal(ranking: pd.DataFrame) -> None:
    ranking = ranking.sort_values("signal_strength").reset_index(drop=True)
    labels = ranking["display_name"] + np.where(
        ranking["auc"] >= 0.5, "\n(higher = more risk)", "\n(higher = less risk)"
    )
    colors = np.where(
        ranking["kind"].eq("education"), COLORS["established"], COLORS["overall"]
    )
    figure, axis = plt.subplots(figsize=(12, 8))
    bars = axis.barh(labels, ranking["signal_strength"], color=list(colors))
    axis.set(
        title="Thin-file applicants: signal strength - all feature types\n"
        "Education stands out as the strongest thin-file predictor",
        xlabel="Single-feature separation score\n(0 = no signal, 1 = perfect)",
        xlim=(0, max(0.5, ranking["signal_strength"].max() * 1.15)),
    )
    axis.bar_label(
        bars, labels=[f"{value:.2f}" for value in ranking["signal_strength"]],
        padding=3, fontsize=9,
    )

    education = ranking[ranking["kind"].eq("education")].iloc[0]
    best_numeric = ranking.loc[ranking["kind"].eq("numeric"), "signal_strength"].max()
    education_position = int(ranking.index[ranking["kind"].eq("education")][0])
    ratio = education["signal_strength"] / best_numeric if best_numeric else float("nan")
    axis.annotate(
        f"Education: {education['signal_strength']:.2f}\n"
        f"({ratio:.1f}x stronger than\nnext-best feature: {best_numeric:.2f})",
        xy=(education["signal_strength"], education_position),
        xytext=(education["signal_strength"] - 0.16, education_position - 3.4),
        fontsize=10,
        fontweight="bold",
        color=COLORS["established"],
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": COLORS["established"]},
        arrowprops={"arrowstyle": "->", "color": COLORS["established"]},
    )
    axis.legend(
        handles=[
            Patch(color=COLORS["established"], label="Categorical: education (ordinal)"),
            Patch(color=COLORS["overall"], label="Numeric features"),
        ],
        frameon=False,
        loc="lower right",
    )
    save_figure(figure, "08_thin_file_signal_complete.png")


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run notebooks/features.py first.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for legacy_name in ["03_unknown_rates.png", "05_categorical_default_rates.png"]:
        legacy_path = FIGURES_DIR / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    print(f"Loading {INPUT_FILE}...")
    data = pd.read_csv(INPUT_FILE)
    data["segment"] = np.where(data[THIN_FILE].eq(1), "Thin-file", "Established")
    print(f"Loaded {len(data):,} applications and {data.shape[1] - 1} source columns.")

    segments = data.groupby("segment", observed=True)[TARGET].agg(
        applicants="size", defaults="sum", default_rate="mean"
    ).reset_index()
    segments["applicant_share"] = segments["applicants"] / len(data)

    weekly = data.groupby("WEEK_NUM", observed=True)[TARGET].agg(
        applicants="size", overall_default_rate="mean"
    )
    segment_weekly = data.pivot_table(
        index="WEEK_NUM", columns="segment", values=TARGET, aggfunc="mean", observed=True
    ).rename(
        columns={"Established": "established_default_rate", "Thin-file": "thin_file_default_rate"}
    )
    weekly = weekly.join(segment_weekly).reset_index()

    aliases = category_aliases(data)
    rates = categorical_rates(data, aliases)
    availability = feature_availability(data)
    signal = numeric_signal(data)
    coverage = data.groupby("external_coverage", observed=True)[TARGET].agg(
        applicants="size", default_rate="mean"
    ).reset_index()
    coverage_segments = coverage_by_segment(data)
    grouped_rates = feature_group_default_rates(data)
    reference_rates = {
        "Overall": data[TARGET].mean(),
        "Thin-file": segments.set_index("segment").loc["Thin-file", "default_rate"],
    }
    thin_file_signal = thin_file_all_signal(data)
    thin_file_ranking = thin_file_signal_ranking(thin_file_signal)

    print("Creating figures...")
    plot_target_and_segments(data, segments)
    plot_weekly_rates(weekly)
    plot_feature_availability(availability)
    plot_numeric_distributions(data)
    plot_education_and_coverage(rates, coverage)
    plot_coverage_simpsons(coverage_segments)
    plot_numeric_signal(signal)
    plot_feature_group_rates(grouped_rates, reference_rates)
    plot_thin_file_signal(thin_file_ranking)

    print("Validating the education signal (time-based holdout + permutation test)...")
    validation = validate_education_signal(data)
    print(
        f"  Education separation: {validation['observed_separation']:.3f} "
        f"(AUC {validation['observed_auc']:.3f}) on all thin-file applicants.\n"
        f"  Time-based holdout: train {validation['train_separation']:.3f} "
        f"-> validation {validation['validation_separation']:.3f} "
        "(order learned on weeks 0-75, applied cold to weeks 76-91).\n"
        f"  Permutation test ({PERMUTATION_SHUFFLES} shuffles): null max AUC "
        f"{validation['null_max_auc']:.3f}, p = {validation['p_value']:.4f}."
    )

    expected = [
        OUTPUT_DIR / "EDA_SUMMARY.md",
        FIGURES_DIR / "05b_coverage_simpsons_paradox.png",
    ] + [
        FIGURES_DIR / f"{number:02d}_{name}.png"
        for number, name in [
            (1, "target_and_segments"),
            (2, "default_rate_over_time"),
            (3, "feature_availability"),
            (4, "numeric_distributions"),
            (5, "education_and_coverage"),
            (6, "numeric_feature_signal"),
            (7, "binned_default_rates"),
            (8, "thin_file_signal_complete"),
        ]
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in expected):
        raise RuntimeError("EDA output verification failed.")
    print(f"-> EDA complete. Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
