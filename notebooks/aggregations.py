"""
aggregations.py

Functions to squash each depth-1/depth-2 feature table in the dataset 
down to one row per case_id, so it can be joined onto base (training or test).

Outline:
  - applprev_1
  - credit_bureau_a aggregations
  - person_1
  - static_0
  - tax_registry_a_1
"""

import pandas as pd
from aggregation_helpers import mean_median_max_of, onehot_encode_and_sum, concat_and_sort

# applprev_1
def aggregate_applprev_1(df: pd.DataFrame, category_list=None) -> pd.DataFrame:
    """Squash the applprev_1 table."""
    numeric_agg = df.groupby("case_id").agg(
        applprev_rowcount=("num_group1", "count"),
        **mean_median_max_of("credamount_590A"),
    ).reset_index()

    onehot_agg = onehot_encode_and_sum(df, "status_219L", category_list=category_list)

    return numeric_agg.merge(onehot_agg, on="case_id", how="left")

# credit_bureau_a_*
def _aggregate_credit_bureau_a_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Squash the credit_bureau_a_2 table (case_id, num_group1, num_group2)
    into one row per (case_id, num_group1).
    """
    agg = df.groupby(["case_id", "num_group1"]).agg(
        payment_rowcount=("num_group2", "count"),
        **mean_median_max_of("pmts_overdue_1140A")
    ).reset_index()
    return agg


def combine_credit_bureau_a_1(df_0: pd.DataFrame, df_1: pd.DataFrame) -> pd.DataFrame:
    """Combine the two credit_bureau_a_1 tables (depth 1) into one table."""
    return concat_and_sort(df_0, df_1)


def aggregate_credit_bureau_a_1(df_a1: pd.DataFrame, df_a2: pd.DataFrame, category_list=None) -> pd.DataFrame:
    """Squash the credit_bureau_a_1 table."""
    contract_payments = _aggregate_credit_bureau_a_2(df_a2)

    contracts = df_a1.merge(contract_payments, on=["case_id", "num_group1"], how="left")

    # payment-level stats, computed directly across ALL of a person's payments regardless of the contract
    payment_level_agg = df_a2.groupby("case_id").agg(
        pmts_overdue_1140A_allpayments_mean=("pmts_overdue_1140A", "mean"),
        pmts_overdue_1140A_allpayments_median=("pmts_overdue_1140A", "median"),
    ).reset_index()

    # squash contracts (now with per-contract payment stats) to one row per case_id
    numeric_agg = contracts.groupby("case_id").agg(
        bureau_rowcount=("num_group1", "count"),
        **mean_median_max_of("credamount_770A"),
        **mean_median_max_of("overdueamountmax_950A"),
        **mean_median_max_of("pmts_dpdvalue_108P"),
        payment_rowcount_sum=("payment_rowcount", "sum"),
        pmts_overdue_1140A_contractmax_max=("pmts_overdue_1140A_max", "max"), # worst overdue payment on their single worst contract
    ).reset_index()

    numeric_agg = numeric_agg.merge(payment_level_agg, on="case_id", how="left")

    onehot_agg = onehot_encode_and_sum(contracts, "classificationofcontr_13M", category_list=category_list)
    return numeric_agg.merge(onehot_agg, on="case_id", how="left")


# person_1
def aggregate_person_1(df: pd.DataFrame, category_list=None) -> pd.DataFrame:
    """Squash the person_1 table."""
    numeric_agg = df.groupby("case_id").agg(
        person_rowcount=("num_group1", "count"),
        **mean_median_max_of("mainoccupationinc_384A"),
        income_total=("mainoccupationinc_384A", "sum"),
        **mean_median_max_of("empl_employedtotal_800L"),
    ).reset_index()
    numeric_agg["has_second_person"] = numeric_agg["person_rowcount"] > 1

    onehot_agg = onehot_encode_and_sum(df, "housingtype_772M", category_list=category_list)
    return numeric_agg.merge(onehot_agg, on="case_id", how="left")


# static_0
def combine_static_0(df_0: pd.DataFrame, df_1: pd.DataFrame) -> pd.DataFrame:
    """Combine the two static_0 tables (depth 0) into one table."""
    return concat_and_sort(df_0, df_1)


# tax_registry_a_1
def aggregate_tax_registry_a_1(df: pd.DataFrame, category_list=None) -> pd.DataFrame:
    """Squash the tax_registry_a_1 table."""
    numeric_agg = df.groupby("case_id").agg(
        tax_rowcount=("num_group1", "count"),
        **mean_median_max_of("amount_4527230A")
    ).reset_index()

    onehot_agg = onehot_encode_and_sum(df, "name_4527232M", category_list=category_list)
    return numeric_agg.merge(onehot_agg, on="case_id", how="left")
