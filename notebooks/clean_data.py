import os
import pandas as pd


# -------------------------
# Load train and test datasets
# -------------------------

train = pd.read_csv("data/aggregated/train_aggregated.csv")
test = pd.read_csv("data/aggregated/test_aggregated.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nMissing values before:")
print("Train:", train.isna().sum().sum())
print("Test:", test.isna().sum().sum())


# -------------------------
# Handle missing structural count values
# Missing means there were no corresponding records
# -------------------------

count_cols = [
    "applprev_rowcount",
    "person_rowcount",
    "tax_rowcount",
    "bureau_rowcount",
    "payment_rowcount_sum"
]

for col in count_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)


# -------------------------
# Handle missing category counts
# No corresponding records means zero of that category
# -------------------------

count_prefixes = [
    "status_219L_",
    "name_4527232M_",
    "classificationofcontr_13M_"
]

for col in train.columns:
    if any(col.startswith(prefix) for prefix in count_prefixes):
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)


# -------------------------
# Handle remaining numeric missing values
# -1 represents an unknown/missing value
# -------------------------

numeric_cols = train.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    if col in test.columns:
        train[col] = train[col].fillna(-1)
        test[col] = test[col].fillna(-1)


# -------------------------
# Handle categorical missing values
# -------------------------

train["description_5085714M"] = (
    train["description_5085714M"].fillna("MISSING")
)

test["description_5085714M"] = (
    test["description_5085714M"].fillna("MISSING")
)


# -------------------------
# Verify missing values
# -------------------------

print("\nMissing values after:")
print("Train:", train.isna().sum().sum())
print("Test:", test.isna().sum().sum())


""" # -------------------------
# Check potential outliers
# -------------------------
# Potential outliers were reviewed using the IQR method.
# Extreme values were retained because they were plausible
# and may contain meaningful credit-risk information.
# Code kept and commented out in case we need to revisit.

print("\nPOTENTIAL OUTLIERS:")

# Columns where IQR outlier detection is not appropriate
exclude_cols = [
    "case_id",
    "target",
    "MONTH",
    "WEEK_NUM",
    "numberofqueries_146L",
    "applprev_rowcount",
    "person_rowcount",
    "tax_rowcount",
    "bureau_rowcount",
    "payment_rowcount_sum"
]

# Category/count features
exclude_prefixes = [
    "status_219L_",
    "name_4527232M_",
    "classificationofcontr_13M_",
    "housingtype_772M_"
]

numeric_cols = train.select_dtypes(include=["number"]).columns

for col in numeric_cols:

    if col in exclude_cols:
        continue

    if any(col.startswith(prefix) for prefix in exclude_prefixes):
        continue

    # Exclude -1 because it represents missing data
    values = train.loc[train[col] != -1, col]

    if len(values) == 0:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    potential_outliers = (
        (values < lower_bound) |
        (values > upper_bound)
    ).sum()

    if potential_outliers > 0:
        percent = (potential_outliers / len(values)) * 100

        print(
            f"{col}: "
            f"min={values.min()}, "
            f"max={values.max()}, "
            f"lower={lower_bound:.2f}, "
            f"upper={upper_bound:.2f}, "
            f"outliers={potential_outliers}, "
            f"percent={percent:.2f}%"
        ) """


# -------------------------
# Save cleaned datasets
# -------------------------

os.makedirs("data/cleaned", exist_ok=True)

train.to_csv(
    "data/cleaned/train_cleaned.csv",
    index=False
)

test.to_csv(
    "data/cleaned/test_cleaned.csv",
    index=False
)

print("\nCleaned datasets saved to data/cleaned/")