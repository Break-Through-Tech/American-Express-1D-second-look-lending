# Task 5 EDA Findings

## Key Numbers

- 1,000,000 applications across 92 training weeks.
- 191,752 defaults, giving an overall default rate of 19.2%.
- 299,821 thin-file applicants, representing 30.0% of all applications.
- Thin-file default rate: 25.6%.
- Established-applicant default rate: 16.4%.
- No duplicate case IDs, constant columns, or true NaN values were found. Unknown values are stored as
  `-1` for numeric fields and `MISSING` for categorical fields.

## Figure 1: Applicant Mix and Default Rate

Thin-file applicants have a default rate 9.2 percentage points higher than established applicants.
Because they are almost one-third of the dataset, overall performance could hide poor results for the
population this project is intended to serve.

The dashed 20% line shows the challenge's simple break-even default rate from the example payoff. It is
context for the group averages, not an applicant-level approval rule.

## Figure 2: Risk Changes Over Time

- Thin-file default falls from 32.4% in the first 10 weeks to 19.7% in the last 10 weeks.
- Established default stays nearly unchanged: 16.5% early and 16.3% late.

Most of the improvement in the overall default rate comes from the thin-file segment changing over time.
A random validation split would mix the early high-risk period with the later lower-risk period. A
time-based split will provide a more realistic test of future performance.

## Figure 3: Feature Availability

| Feature | Established | Thin-file |
| --- | ---: | ---: |
| Risk assessment score | 95.0% | 0.0% |
| Average days past due | 100.0% | 0.0% |
| Average tax-registry amount | 75.0% | 75.0% |
| Average prior application amount | 80.0% | 80.0% |
| Applicant income | 95.0% | 95.0% |
| Employment duration | 84.9% | 85.0% |
| Credit-to-income ratio | 93.1% | 93.2% |
| Education category | 100.0% | 100.0% |

Bureau risk and payment-history fields are unavailable for every thin-file applicant. Income,
employment, affordability, tax-registry, prior-application, and education fields remain available at
almost identical rates in both segments. These are the most practical inputs for second-look decisions.

## Figure 4: Numeric Distributions

The red dashed line is the median. The histograms display the 1st through 99th percentile range so a few
extreme values do not compress the useful part of each chart. No observations are removed from the data.

| Feature | Median | 1st percentile | 99th percentile | Skewness |
| --- | ---: | ---: | ---: | ---: |
| Applicant income | 22,021.00 | 7,516.00 | 64,417.56 | 1.59 |
| Requested credit | 14,760.00 | 4,610.00 | 47,330.00 | 1.75 |
| Credit-to-income ratio | 0.67 | 0.14 | 3.26 | 2.76 |
| Age (years) | 46.28 | 22.28 | 70.38 | 0.00 |
| Average days past due | 8.90 | 0.00 | 28.18 | 0.75 |

Income, requested credit, credit-to-income, and days past due are right-skewed. Their medians are more
representative than their means. Age is nearly symmetric. The long upper tail in credit-to-income may
contain useful risk information and should not automatically be removed as an outlier.

## Figure 5: Education and Data-Source Coverage

The education values are anonymized. Groups A through E are neutral aliases ordered from the lowest to
highest observed overall default rate. They do not reveal what the original education categories mean.

| Display Label | Original Code | Applicants | Overall Default | Thin-File Default |
| --- | --- | ---: | ---: | ---: |
| Education group A | `f937ecaa` | 141,515 | 3.9% | 7.3% |
| Education group B | `10795bac` | 218,145 | 8.4% | 14.0% |
| Education group C | `6def22f0` | 279,658 | 15.5% | 22.9% |
| Education group D | `242b264b` | 218,669 | 26.6% | 34.9% |
| Education group E | `2d4a2bc4` | 142,013 | 46.7% | 52.7% |

The ordering remains strong among thin-file applicants, so education may provide useful separation when
bureau information is unavailable. Marital-status default rates are nearly identical across all four
categories, so marital status appears much less useful.

Credit-file description is associated with default overall, but every thin-file applicant has
`MISSING`. It cannot separate applicants within the thin-file segment.

| Available Data Sources | Applicants | Default Rate |
| ---: | ---: | ---: |
| 0 | 15,007 | 25.7% |
| 1 | 140,274 | 23.3% |
| 2 | 424,499 | 20.3% |
| 3 | 420,220 | 16.4% |

Default falls as bureau, prior-application, and tax-registry coverage increases. Data-source coverage is
therefore informative and should remain available to later models.

## Figure 6: Strongest Single-Feature Signals

The separation score is `2 × |AUC - 0.5|`. A score of 0 means the feature has no ability to rank defaults
by itself; a score of 1 means perfect separation. Mean, median, and maximum versions of the same concept
are grouped so the chart does not repeat nearly identical signals.

Overall, the strongest numeric signals are risk assessment, days past due, past bureau credit, overdue
amounts, and credit-query activity. These depend heavily on bureau history.

For thin-file applicants, the strongest numeric signals shift to tax-registry amounts, employment
duration, employment history, household income, prior-application amounts, and credit-to-income. Their
individual separation is weaker, which shows why thin-file predictions are more difficult.

The direction written beside each feature matters. For example, higher days past due is associated with
more defaults, while higher tax-registry amounts and longer employment are associated with fewer defaults.

## Figure 7: Default Rate Across Feature Deciles

Each feature is sorted from low to high and divided into ten similarly sized applicant groups.

| Feature | Lowest-Decile Default | Highest-Decile Default |
| --- | ---: | ---: |
| Risk assessment score | 1.4% | 54.1% |
| Average days past due | 5.3% | 40.5% |
| Credit-to-income ratio | 15.4% | 23.4% |
| Employment share of life | 28.4% | 13.7% |

Risk assessment and days-past-due behavior have strong increasing risk gradients. Credit-to-income has a
smaller but consistent increase. Employment share of life moves in the opposite direction: longer
employment relative to age is associated with lower default.

## Main Takeaways

1. Validate by time because thin-file risk changes substantially across the training period.
2. Evaluate thin-file and established applicants separately instead of relying only on overall metrics.
3. Preserve missingness and source-coverage indicators because bureau absence is central to this task.
4. For thin-file applicants, prioritize education, tax-registry amounts, employment, income,
   affordability, and prior-application information.
5. These findings describe associations, not causes. Confirm them with validation and calibration before
   using them in lending decisions.

All counts, rates, distributions, and time trends use the full training dataset. Single-feature signal
rankings use a fixed 200,000-row sample for faster and reproducible execution.
