"""
Task 5: Exploratory data analysis.

Run this after features.py. It uses the full training dataset for summary statistics and charts,
then writes a short findings note, reusable tables, and figures to reports/eda.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT_DIR / "data" / "features" / "train_features.csv"
OUTPUT_DIR = ROOT_DIR / "reports" / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

TARGET = "target"
THIN_FILE = "is_thin_file"
MISSING_VALUE = -1
SAMPLE_SIZE = 200_000
RANDOM_STATE = 42
BREAK_EVEN_RATE = 0.20

COLORS = {"overall": "#345995", "established": "#2A9D8F", "thin": "#E76F51"}
CATEGORICAL_FEATURES = ["education_927M", "maritalstatus_703M", "description_5085714M"]
NUMERIC_DISTRIBUTIONS = [
    "riskassesment_940T",
    "mainoccupationinc_384A",
    "credamount_770A",
    "credit_to_income",
    "age_in_years",
    "pmts_dpdvalue_108P_mean",
]
BINNED_FEATURES = [
    "riskassesment_940T",
    "pmts_dpdvalue_108P_mean",
    "credit_to_income",
    "employment_share_of_life",
]
DISPLAY_NAMES = {
    "riskassesment_940T": "Risk assessment score",
    "mainoccupationinc_384A": "Applicant income",
    "credamount_770A": "Requested credit",
    "credit_to_income": "Credit-to-income ratio",
    "age_in_years": "Age (years)",
    "pmts_dpdvalue_108P_mean": "Average days past due",
    "employment_share_of_life": "Employment share of life",
    "education_927M": "Education category",
    "description_5085714M": "Credit-file description",
}


def clean_numeric(values: pd.Series) -> pd.Series:
    """Replace the pipeline's -1 unknown marker with NaN for analysis."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.mask(numeric.eq(MISSING_VALUE))


def feature_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Create a compact data dictionary with coverage and numeric summary statistics."""
    rows = []
    for column in data.columns:
        values = data[column]
        if is_numeric_dtype(values):
            missing = values.isna() | values.eq(MISSING_VALUE)
            valid = clean_numeric(values)
        else:
            missing = values.isna() | values.astype("string").str.upper().eq("MISSING")
            valid = None

        if column == "case_id":
            role = "identifier"
        elif column == TARGET:
            role = "target"
        elif column in {"date_decision", "MONTH", "WEEK_NUM"}:
            role = "time"
        elif not is_numeric_dtype(values):
            role = "categorical"
        elif values.nunique(dropna=False) <= 2:
            role = "binary"
        else:
            role = "numeric"

        row = {
            "feature": column,
            "role": role,
            "dtype": str(values.dtype),
            "unique_values": values.nunique(dropna=False),
            "unknown_count": int(missing.sum()),
            "unknown_rate": missing.mean(),
        }
        if valid is not None:
            row.update(
                {
                    "mean": valid.mean(),
                    "median": valid.median(),
                    "standard_deviation": valid.std(),
                    "minimum": valid.min(),
                    "p01": valid.quantile(0.01),
                    "p99": valid.quantile(0.99),
                    "maximum": valid.max(),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_signal(data: pd.DataFrame) -> pd.DataFrame:
    """Rank numeric features using univariate AUC overall and within each segment."""
    sample = data.sample(min(SAMPLE_SIZE, len(data)), random_state=RANDOM_STATE)
    excluded = {"case_id", "MONTH", "WEEK_NUM", TARGET}
    features = [
        column
        for column in sample.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded
    ]
    groups = {
        "Overall": sample,
        "Established": sample[sample[THIN_FILE].eq(0)],
        "Thin-file": sample[sample[THIN_FILE].eq(1)],
    }
    rows = []

    for segment, group in groups.items():
        for feature in features:
            values = clean_numeric(group[feature])
            valid = values.notna()
            auc = np.nan
            if valid.sum() >= 500 and values.loc[valid].nunique() > 1:
                auc = roc_auc_score(group.loc[valid, TARGET], values.loc[valid])
            rows.append(
                {
                    "segment": segment,
                    "feature": feature,
                    "valid_rows": int(valid.sum()),
                    "auc": auc,
                    "signal_strength": 2 * abs(auc - 0.5) if pd.notna(auc) else np.nan,
                    "direction": (
                        "higher values -> more defaults"
                        if pd.notna(auc) and auc >= 0.5
                        else "higher values -> fewer defaults"
                        if pd.notna(auc)
                        else "not available"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["segment", "signal_strength"], ascending=[True, False]
    )


def categorical_summaries(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate category default rates and Cramer's V target association."""
    groups = {
        "Overall": data,
        "Established": data[data[THIN_FILE].eq(0)],
        "Thin-file": data[data[THIN_FILE].eq(1)],
    }
    rate_tables = []
    signal_rows = []

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

            contingency = pd.crosstab(group[feature].fillna("MISSING"), group[TARGET])
            if contingency.shape[0] < 2:
                association = np.nan
            else:
                observed = contingency.to_numpy(dtype=float)
                expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
                chi_squared = np.sum((observed - expected) ** 2 / expected)
                association = np.sqrt(chi_squared / observed.sum())
            signal_rows.append(
                {
                    "segment": segment,
                    "feature": feature,
                    "categories": group[feature].nunique(dropna=False),
                    "cramers_v": association,
                }
            )

    return pd.concat(rate_tables, ignore_index=True), pd.DataFrame(signal_rows)


def high_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature pairs above 0.85 absolute correlation."""
    sample = data.sample(min(SAMPLE_SIZE, len(data)), random_state=RANDOM_STATE)
    excluded = {"case_id", "MONTH", "WEEK_NUM", TARGET}
    features = [
        column
        for column in sample.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded
    ]
    values = sample[features].astype(float).replace(MISSING_VALUE, np.nan)
    correlations = values.corr(min_periods=1_000)
    upper_triangle = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    pairs = upper_triangle.stack().rename("correlation")
    pairs = pairs[pairs.abs() >= 0.85].sort_values(key=lambda series: series.abs(), ascending=False)
    result = pairs.reset_index().rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
    result["absolute_correlation"] = result["correlation"].abs()
    return result


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
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                     f"{bar.get_height():.1%}", ha="center")

    shares = segments.set_index("segment")["applicant_share"]
    bars = axes[1].bar(["Established", "Thin-file"], shares,
                       color=[COLORS["established"], COLORS["thin"]])
    axes[1].set(title="Applicant mix", ylabel="Share of applications", ylim=(0, 0.8))
    axes[1].yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                     f"{bar.get_height():.1%}", ha="center")
    save_figure(figure, "01_target_and_segments.png")


def plot_weekly_rates(weekly: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(11, 5))
    for column, label, color in [
        ("overall_default_rate", "Overall", COLORS["overall"]),
        ("established_default_rate", "Established", COLORS["established"]),
        ("thin_file_default_rate", "Thin-file", COLORS["thin"]),
    ]:
        axis.plot(weekly["WEEK_NUM"], weekly[column].rolling(4, min_periods=1).mean(),
                  label=f"{label} (4-week average)", color=color, linewidth=2)
    axis.axhline(BREAK_EVEN_RATE, color="gray", linestyle="--", label="20% simple break-even")
    axis.set(title="Default rate changes across training weeks", xlabel="Training week",
             ylabel="Default rate")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(frameon=False, ncol=2)
    save_figure(figure, "02_default_rate_over_time.png")


def plot_unknown_rates(profile: pd.DataFrame) -> None:
    unknown = profile[(profile["unknown_rate"] > 0) & ~profile["feature"].isin(["case_id", TARGET])]
    unknown = unknown.nlargest(15, "unknown_rate").sort_values("unknown_rate")
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(unknown["feature"], unknown["unknown_rate"], color=COLORS["overall"])
    axis.set(title="Features with the highest unknown rates", xlabel="Share recorded as unknown")
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar in bars:
        axis.text(bar.get_width() + 0.006, bar.get_y() + bar.get_height() / 2,
                  f"{bar.get_width():.1%}", va="center", fontsize=8)
    save_figure(figure, "03_unknown_rates.png")


def plot_numeric_distributions(data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for axis, feature in zip(axes.flat, NUMERIC_DISTRIBUTIONS):
        values = clean_numeric(data[feature]).dropna()
        lower, upper = values.quantile([0.01, 0.99])
        display_values = values[values.between(lower, upper)]
        axis.hist(display_values, bins=45, color=COLORS["overall"], alpha=0.85)
        axis.axvline(values.median(), color=COLORS["thin"], linestyle="--")
        axis.set(title=DISPLAY_NAMES[feature], ylabel="Applicants")
    figure.suptitle("Key numeric distributions (display limited to p01-p99)", y=1.02)
    save_figure(figure, "04_numeric_distributions.png")


def plot_categorical_rates(rates: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, feature in zip(axes, ["education_927M", "description_5085714M"]):
        subset = rates[(rates["feature"] == feature) & rates["segment"].isin(["Overall", "Thin-file"])]
        overall = subset[subset["segment"] == "Overall"].set_index("category")["default_rate"].sort_values()
        thin = subset[subset["segment"] == "Thin-file"].set_index("category")["default_rate"]
        positions = np.arange(len(overall))
        axis.barh(positions - 0.18, overall, height=0.34, label="Overall", color=COLORS["overall"])
        axis.barh(positions + 0.18, thin.reindex(overall.index), height=0.34,
                  label="Thin-file", color=COLORS["thin"])
        axis.set_yticks(positions, overall.index)
        title = DISPLAY_NAMES[feature]
        if feature == "description_5085714M":
            title += "\n(thin-file applicants only have MISSING)"
        axis.set(title=title, xlabel="Default rate")
        axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        axis.legend(frameon=False)
    save_figure(figure, "05_categorical_default_rates.png")


def plot_numeric_signal(signal: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for axis, segment, color in [
        (axes[0], "Overall", COLORS["overall"]),
        (axes[1], "Thin-file", COLORS["thin"]),
    ]:
        strongest = signal[signal["segment"] == segment].nlargest(10, "signal_strength")
        strongest = strongest.sort_values("signal_strength")
        axis.barh(strongest["feature"], strongest["signal_strength"], color=color)
        axis.set(title=f"Strongest numeric signals: {segment}",
                 xlabel="Univariate AUC separation (0 = none, 1 = perfect)", xlim=(0, 0.64))
    save_figure(figure, "06_numeric_feature_signal.png")


def plot_binned_rates(data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.flat, BINNED_FEATURES):
        values = clean_numeric(data[feature])
        valid = values.notna()
        plot_data = pd.DataFrame({"value": values[valid], TARGET: data.loc[valid, TARGET]})
        plot_data["decile"] = pd.qcut(plot_data["value"], 10, labels=False, duplicates="drop")
        binned = plot_data.groupby("decile", observed=True)[TARGET].mean().reset_index()
        axis.plot(binned["decile"] + 1, binned[TARGET], marker="o", color=COLORS["overall"], linewidth=2)
        axis.axhline(data[TARGET].mean(), color="gray", linestyle="--", linewidth=1)
        axis.set(title=DISPLAY_NAMES[feature], xlabel="Feature decile (low to high)",
                 ylabel="Default rate", xticks=range(1, len(binned) + 1))
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    figure.suptitle("Default rate across numeric feature deciles", y=1.02)
    save_figure(figure, "07_binned_default_rates.png")


def describe_signal(signal: pd.DataFrame, segment: str) -> str:
    strongest = signal[signal["segment"] == segment].nlargest(5, "signal_strength")
    return ", ".join(
        f"`{row.feature}` ({row.auc:.3f} AUC; {row.direction})"
        for row in strongest.itertuples()
    )


def write_summary(data: pd.DataFrame, profile: pd.DataFrame, segments: pd.DataFrame,
                  signal: pd.DataFrame, category_rates: pd.DataFrame,
                  category_signal: pd.DataFrame, correlations: pd.DataFrame) -> None:
    segment_rates = segments.set_index("segment")["default_rate"]
    first_weeks = data[data["WEEK_NUM"] <= data["WEEK_NUM"].min() + 9]
    last_weeks = data[data["WEEK_NUM"] >= data["WEEK_NUM"].max() - 9]
    early_thin = first_weeks.loc[first_weeks[THIN_FILE].eq(1), TARGET].mean()
    late_thin = last_weeks.loc[last_weeks[THIN_FILE].eq(1), TARGET].mean()
    early_established = first_weeks.loc[first_weeks[THIN_FILE].eq(0), TARGET].mean()
    late_established = last_weeks.loc[last_weeks[THIN_FILE].eq(0), TARGET].mean()

    unknown = profile[(profile["unknown_rate"] > 0) & ~profile["feature"].isin(["case_id", TARGET])]
    unknown_text = ", ".join(
        f"`{row.feature}` ({row.unknown_rate:.1%})" for row in unknown.nlargest(5, "unknown_rate").itertuples()
    )
    category = category_signal[category_signal["segment"] == "Overall"].nlargest(1, "cramers_v").iloc[0]
    education = category_rates[(category_rates["feature"] == "education_927M")
                               & (category_rates["segment"] == "Overall")]
    pair_text = "; ".join(
        f"`{row.feature_1}` with `{row.feature_2}` ({row.correlation:.3f})"
        for row in correlations.head(4).itertuples()
    )

    summary = f"""# Exploratory Data Analysis Summary

This analysis uses `data/features/train_features.csv`, the output of Task 4. It does not alter any
aggregation, cleaning, preprocessing, or feature-engineering work from Tasks 1-4.

## Dataset overview

- **{len(data):,} applications**, **{data.shape[1] - 1} columns**, and **{data['WEEK_NUM'].nunique()} weeks**
  from **{data['date_decision'].min()}** to **{data['date_decision'].max()}**.
- Overall default rate: **{data[TARGET].mean():.1%}**. This is close to the challenge's simple 20%
  break-even rate, but decisions still require applicant-level probabilities.
- **{data['case_id'].duplicated().sum():,} duplicate case IDs** and
  **{data.drop(columns='segment').isna().sum().sum():,} true NaN values**.

## Main findings

### Thin-file applicants

- Thin-file applicants are **{data[THIN_FILE].mean():.1%}** of applications.
- Their default rate is **{segment_rates['Thin-file']:.1%}**, compared with
  **{segment_rates['Established']:.1%}** for established applicants.
- This group-level difference is not a denial rule. It shows why the model needs non-bureau signals
  and separate thin-file performance checks.

### Time drift

- Thin-file default falls from **{early_thin:.1%}** in the first 10 weeks to
  **{late_thin:.1%}** in the last 10 weeks.
- Established default stays steadier: **{early_established:.1%}** early versus
  **{late_established:.1%}** late.
- Later training weeks should be held out for validation because a random split could hide this drift.

### Coverage and missingness

- Task 2 removed true NaNs, but numeric unknowns remain encoded as `-1` and categorical unknowns as
  `MISSING`. Highest unknown rates: {unknown_text}.
- Missingness and source coverage should remain available to the model rather than being treated as
  ordinary measured values.

### Early feature signal

- Overall: {describe_signal(signal, 'Overall')}.
- Thin-file: {describe_signal(signal, 'Thin-file')}.
- Strongest reviewed categorical association: `{category['feature']}` (Cramer's V
  **{category['cramers_v']:.3f}**). Education-category default rates range from
  **{education['default_rate'].min():.1%}** to **{education['default_rate'].max():.1%}**.
- These are one-feature relationships, not final model importance or causal effects.

### Related features

- Strongest high-correlation pairs: {pair_text}.
- Correlated features are not automatically wrong, but they may duplicate information in linear models.

## Modeling takeaways

1. Use time-based validation.
2. Report performance separately for thin-file and established applicants.
3. Preserve missing/source-coverage indicators.
4. Test income, employment, affordability, tax, prior-application, and education signals for thin files.
5. Confirm these EDA signals with validation, calibration, profit, and inclusion metrics.

The signal and correlation calculations use a fixed sample of up to {SAMPLE_SIZE:,} rows for speed.
All other results use the full training dataset. Generated tables and figures are in this folder.
"""
    (OUTPUT_DIR / "EDA_SUMMARY.md").write_text(summary, encoding="utf-8")


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run notebooks/features.py first.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_FILE}...")
    data = pd.read_csv(INPUT_FILE)
    data["segment"] = np.where(data[THIN_FILE].eq(1), "Thin-file", "Established")
    print(f"Loaded {len(data):,} applications and {data.shape[1] - 1} source columns.")

    # Build reusable tables before plotting so every figure is based on saved results.
    profile = feature_profile(data.drop(columns="segment"))
    overview = pd.DataFrame(
        {
            "metric": ["rows", "columns", "default_rate", "thin_file_share", "duplicate_case_ids"],
            "value": [len(data), data.shape[1] - 1, data[TARGET].mean(), data[THIN_FILE].mean(),
                      data["case_id"].duplicated().sum()],
        }
    )
    segments = data.groupby("segment", observed=True)[TARGET].agg(
        applicants="size", defaults="sum", default_rate="mean"
    ).reset_index()
    segments["applicant_share"] = segments["applicants"] / len(data)

    weekly = data.groupby("WEEK_NUM", observed=True)[TARGET].agg(
        applicants="size", overall_default_rate="mean"
    )
    segment_weekly = data.pivot_table(index="WEEK_NUM", columns="segment", values=TARGET,
                                      aggfunc="mean", observed=True).rename(
        columns={"Established": "established_default_rate", "Thin-file": "thin_file_default_rate"}
    )
    weekly = weekly.join(segment_weekly).reset_index()
    signal = numeric_signal(data)
    category_rates, category_signal = categorical_summaries(data)
    correlations = high_correlations(data)

    overview.to_csv(TABLES_DIR / "dataset_overview.csv", index=False)
    profile.to_csv(TABLES_DIR / "feature_profile.csv", index=False, float_format="%.6f")
    segments.to_csv(TABLES_DIR / "segment_summary.csv", index=False, float_format="%.6f")
    weekly.to_csv(TABLES_DIR / "weekly_summary.csv", index=False, float_format="%.6f")
    signal.to_csv(TABLES_DIR / "numeric_signal.csv", index=False, float_format="%.6f")
    category_rates.to_csv(TABLES_DIR / "categorical_default_rates.csv", index=False, float_format="%.6f")
    category_signal.to_csv(TABLES_DIR / "categorical_signal.csv", index=False, float_format="%.6f")
    correlations.to_csv(TABLES_DIR / "high_correlations.csv", index=False, float_format="%.6f")

    print("Creating figures...")
    plot_target_and_segments(data, segments)
    plot_weekly_rates(weekly)
    plot_unknown_rates(profile)
    plot_numeric_distributions(data)
    plot_categorical_rates(category_rates)
    plot_numeric_signal(signal)
    plot_binned_rates(data)
    write_summary(data, profile, segments, signal, category_rates, category_signal, correlations)

    expected = [OUTPUT_DIR / "EDA_SUMMARY.md"] + list(FIGURES_DIR.glob("*.png")) + list(TABLES_DIR.glob("*.csv"))
    if len(expected) != 16 or any(path.stat().st_size == 0 for path in expected):
        raise RuntimeError("EDA output verification failed.")
    print(f"-> EDA complete. Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
