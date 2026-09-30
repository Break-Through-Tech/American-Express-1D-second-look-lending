# Exploratory Data Analysis Summary

This analysis uses `data/features/train_features.csv`, the output of Task 4. It does not alter any
aggregation, cleaning, preprocessing, or feature-engineering work from Tasks 1-4.

## Dataset overview

- **1,000,000 applications**, **87 columns**, and **92 weeks**
  from **2019-01-01** to **2020-09-29**.
- Overall default rate: **19.2%**. This is close to the challenge's simple 20%
  break-even rate, but decisions still require applicant-level probabilities.
- **0 duplicate case IDs** and
  **0 true NaN values**.

## Main findings

### Thin-file applicants

- Thin-file applicants are **30.0%** of applications.
- Their default rate is **25.6%**, compared with
  **16.4%** for established applicants.
- This group-level difference is not a denial rule. It shows why the model needs non-bureau signals
  and separate thin-file performance checks.

### Time drift

- Thin-file default falls from **32.4%** in the first 10 weeks to
  **19.7%** in the last 10 weeks.
- Established default stays steadier: **16.5%** early versus
  **16.3%** late.
- Later training weeks should be held out for validation because a random split could hide this drift.

### Coverage and missingness

- Task 2 removed true NaNs, but numeric unknowns remain encoded as `-1` and categorical unknowns as
  `MISSING`. Highest unknown rates: `pmts_overdue_1140A_contractmax_max` (36.0%), `pmts_overdue_1140A_allpayments_mean` (36.0%), `pmts_overdue_1140A_allpayments_median` (36.0%), `overdueamountmax_950A_mean` (33.7%), `overdueamountmax_950A_max` (33.7%).
- Missingness and source coverage should remain available to the model rather than being treated as
  ordinary measured values.

### Early feature signal

- Overall: `riskassesment_940T` (0.799 AUC; higher values -> more defaults), `pmts_dpdvalue_108P_mean` (0.714 AUC; higher values -> more defaults), `pmts_dpdvalue_108P_median` (0.710 AUC; higher values -> more defaults), `pmts_dpdvalue_108P_max` (0.676 AUC; higher values -> more defaults), `credamount_770A_mean` (0.674 AUC; higher values -> more defaults).
- Thin-file: `amount_4527230A_mean` (0.411 AUC; higher values -> fewer defaults), `amount_4527230A_median` (0.413 AUC; higher values -> fewer defaults), `amount_4527230A_max` (0.426 AUC; higher values -> fewer defaults), `days_employed_700P` (0.426 AUC; higher values -> fewer defaults), `empl_employedtotal_800L_mean` (0.432 AUC; higher values -> fewer defaults).
- Strongest reviewed categorical association: `education_927M` (Cramer's V
  **0.343**). Education-category default rates range from
  **3.9%** to **46.7%**.
- These are one-feature relationships, not final model importance or causal effects.

### Related features

- Strongest high-correlation pairs: `empl_employedtotal_800L_mean` with `empl_employedtotal_800L_median` (1.000); `mainoccupationinc_384A_mean` with `mainoccupationinc_384A_median` (1.000); `person_rowcount` with `has_second_person` (1.000); `amount_4527230A_mean` with `amount_4527230A_median` (0.968).
- Correlated features are not automatically wrong, but they may duplicate information in linear models.

## Modeling takeaways

1. Use time-based validation.
2. Report performance separately for thin-file and established applicants.
3. Preserve missing/source-coverage indicators.
4. Test income, employment, affordability, tax, prior-application, and education signals for thin files.
5. Confirm these EDA signals with validation, calibration, profit, and inclusion metrics.

The signal and correlation calculations use a fixed sample of up to 200,000 rows for speed.
All other results use the full training dataset. Generated tables and figures are in this folder.
