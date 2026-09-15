"""
data.py

Creates the single aggregated training and testing csv file.
Outputs to data/aggregated directory.
"""

import os
import pandas as pd
from aggregation_helpers import fit_categories
from aggregations import ( aggregate_applprev_1, aggregate_credit_bureau_a_1, aggregate_tax_registry_a_1, 
                          aggregate_person_1, combine_credit_bureau_a_1, combine_credit_bureau_a_1, combine_static_0 ) 

INPUT_DIR = "data"
OUTPUT_DIR = "data/aggregated"

TABLE_NAMES = [
    # case_id | num_group1 | credamount_590A | approvaldate_319D | status_219L
    "applprev_1", 
    # case_id | date_decision | MONTH | WEEK_NUM | target
    "base", 
    # case_id | num_group1 | credamount_770A | overdueamountmax_950A | pmts_dpdvalue_108P | classificationofcontr_13M
    "credit_bureau_a_1_0", "credit_bureau_a_1_1", 
    # # case_id | num_group1 | num_group2 | pmts_month_158T | pmts_overdue_1140A
    "credit_bureau_a_2", 
    # case_id | num_group1 | personindex_1023L | mainoccupationinc_384A | empl_employedtotal_800L | housingtype_772M
    "person_1", 
    # case_id | mainoccupationinc_384A | credamount_770A | annuity_780A | days_employed_700P | education_927M | maritalstatus_703M| birth_259D
    "static_0_0", "static_0_1", 
    # case_id| riskassesment_940T | numberofqueries_146L | description_5085714M
    "static_cb_0", 
    # case_id | num_group1 | amount_4527230A | name_4527232M
    "tax_registry_a_1",
]

def _check_data_files_exist(base_dir: str, output_dir: str) -> None:
    """
    Verify all required csv files exist before running the pipeline.
    Creates output_dir if it doesn't already exist. 
    Raises FileNotFoundError listing every missing input file.
    """
    missing = []
    for prefix in ("train", "test"):
        for name in TABLE_NAMES:
            path = f"{base_dir}/{prefix}/{prefix}_{name}.csv"
            if not os.path.isfile(path):
                missing.append(path)

    if missing:
        raise FileNotFoundError(
            "Missing required data file(s):\n  " + "\n  ".join(missing) +
            f"\n\nExpected files under {base_dir}/train/ and {base_dir}/test/, "
            f"named like '{{split}}_{{table_name}}.csv' (e.g. train_applprev_1.csv)."
        )

    os.makedirs(output_dir, exist_ok=True)

def _load_raw_tables(prefix: str) -> dict[str, pd.DataFrame]:
    """
    Load all raw tables into a dict keyed by table name, e.g. {"applprev_1": df, "base": df, ...}.
    """
    return {
        name: pd.read_csv(f"data/{prefix}/{prefix}_{name}.csv")
        for name in TABLE_NAMES
    }

def _fit_all_categories(train_tables: dict[str, pd.DataFrame]) -> dict[str, list]:
    """
    Fit the fixed category list for every masked column, from TRAINING tables only. 
    Reuse the returned dict for both train and test aggregation so one-hot columns match across both.
    """
    # combine the depth-1 credit bureau tables before fitting the categories
    credit_bureau_a_1 = combine_credit_bureau_a_1(
        train_tables["credit_bureau_a_1_0"], train_tables["credit_bureau_a_1_1"]
    )
    return {
        "status_219L": fit_categories(train_tables["applprev_1"], "status_219L"),
        "name_4527232M": fit_categories(train_tables["tax_registry_a_1"], "name_4527232M"),
        "classificationofcontr_13M": fit_categories(credit_bureau_a_1, "classificationofcontr_13M"),
    }


def _aggregate_all_data(tables: dict[str, pd.DataFrame], category_lists: dict[str, list]) -> pd.DataFrame:
    """Aggregate all the dataframes into one dataframe."""
    master_data = tables["base"].copy()

    # DEPTH 0
    # static_0_0 and static_0_1
    static_agg = combine_static_0(tables["static_0_0"], tables["static_0_1"])
    master_data = master_data.merge(static_agg, on="case_id", how="left")

    # static_cb_0
    master_data = master_data.merge(tables["static_cb_0"], on="case_id", how="left")


    # DEPTH 1
    # applprev_1
    applprev_1_agg = aggregate_applprev_1(tables["applprev_1"], category_list=category_lists["status_219L"])
    master_data = master_data.merge(applprev_1_agg, on="case_id", how="left")

    # person_1
    person_1_agg = aggregate_person_1(tables["person_1"])
    master_data = master_data.merge(person_1_agg, on="case_id", how="left")

    # tax_registry_a
    tax_registry_a_1_agg = aggregate_tax_registry_a_1(tables["tax_registry_a_1"], category_list=category_lists["name_4527232M"])
    master_data = master_data.merge(tax_registry_a_1_agg, on="case_id", how="left")


    # !! DEPTH 1 & 2 data squashing & merging with base data
    # credit bureau
    credit_bureau_a_1 = combine_credit_bureau_a_1(tables["credit_bureau_a_1_0"], tables["credit_bureau_a_1_1"])
    credit_bureau_a_agg = aggregate_credit_bureau_a_1(credit_bureau_a_1, 
                                                      tables["credit_bureau_a_2"], 
                                                      category_list=category_lists["classificationofcontr_13M"])
    master_data = master_data.merge(credit_bureau_a_agg, on="case_id", how="left")

    return master_data


def prepare_datasets(input_dir: str, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, aggregate, and save both train and test datasets. 
    Writes train_aggregated.csv and test_aggregated.csv to output_dir, and returns both dataframes.
    """
    _check_data_files_exist(input_dir, output_dir)

    train_tables = _load_raw_tables("train")
    test_tables = _load_raw_tables("test")

    category_lists = _fit_all_categories(train_tables)

    train_data = _aggregate_all_data(train_tables, category_lists)
    test_data = _aggregate_all_data(test_tables, category_lists)

    train_data.to_csv(f"{output_dir}/train_aggregated.csv", index=False)
    test_data.to_csv(f"{output_dir}/test_aggregated.csv", index=False)

    print(f'-> success! wrote train & test datasets as csv files to {output_dir}')

    return train_data, test_data
    

if __name__ == "__main__":
    prepare_datasets(INPUT_DIR, OUTPUT_DIR)