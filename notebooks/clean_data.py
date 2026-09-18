import pandas as pd

# Load train and test datasets
train = pd.read_csv("data/aggregated/train_aggregated.csv")
test = pd.read_csv("data/aggregated/test_aggregated.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nMissing values before:")
print("Train:", train.isna().sum().sum())
print("Test:", test.isna().sum().sum())


# -------------------------
# Handle numeric missing values
# -------------------------

numeric_cols = train.select_dtypes(include=["number"]).columns

for col in numeric_cols:

    # Make sure the column exists in test
    if col in test.columns:

        # Calculate median using TRAINING data only
        median = train[col].median()

        # Use training median for both
        train[col] = train[col].fillna(median)
        test[col] = test[col].fillna(median)


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
# Verify
# -------------------------

print("\nMissing values after:")
print("Train:", train.isna().sum().sum())
print("Test:", test.isna().sum().sum())


# -------------------------
# Handle outliers
# -------------------------

outlier_cols = [
    "mainoccupationinc_384A",
    "credamount_770A",
    "annuity_780A",
    "days_employed_700P",
    "credamount_590A_mean",
    "credamount_590A_max",
    "credamount_590A_median",
    "mainoccupationinc_384A_mean",
    "mainoccupationinc_384A_max",
    "mainoccupationinc_384A_median",
    "income_total",
    "empl_employedtotal_800L_mean",
    "empl_employedtotal_800L_max",
    "empl_employedtotal_800L_median",
    "amount_4527230A_mean",
    "amount_4527230A_max",
    "amount_4527230A_median",
    "credamount_770A_mean",
    "credamount_770A_max",
    "credamount_770A_median",
    "overdueamountmax_950A_mean",
    "overdueamountmax_950A_max",
    "overdueamountmax_950A_median",
    "pmts_dpdvalue_108P_mean",
    "pmts_dpdvalue_108P_max",
    "pmts_dpdvalue_108P_median",
    "pmts_overdue_1140A_contractmax_max",
    "pmts_overdue_1140A_allpayments_mean",
    "pmts_overdue_1140A_allpayments_median"
]

for col in outlier_cols:

    # Calculate boundaries from training data only
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Cap values outside the boundaries
    train[col] = train[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    test[col] = test[col].clip(
        lower=lower_bound,
        upper=upper_bound
    )

print("\nOutlier handling complete.")

# -------------------------
# Verify outlier handling
# -------------------------

print("\nOUTLIERS AFTER CAPPING:")

for col in outlier_cols:

    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (train[col] < lower_bound) |
        (train[col] > upper_bound)
    ).sum()

    print(f"{col}: {outliers}")
import os

# -------------------------
# Save cleaned datasets
# -------------------------

os.makedirs("data/cleaned", exist_ok=True)

train.to_csv("data/cleaned/train_cleaned.csv", index=False)
test.to_csv("data/cleaned/test_cleaned.csv", index=False)

print("\nCleaned datasets saved to data/cleaned/")