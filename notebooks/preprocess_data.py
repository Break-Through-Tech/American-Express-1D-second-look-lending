"""
Task 3: Standardize categorical variables, normalize numerical features. Organize the different categories (one-hot encoding) and scale if needed.

- Separate the columns into distinct lists filtering out the target variable (target), ID columns (case_id), and variables that Melinda already one-hot encoded from task 1.
- Convert columns holding string date variables into numerical representations like age in years or the month to avoid crashes with scikit-learn later on.
- Apply one-hot encoding to the categorical variables that have not been encoded yet.
- Scale the numeric columns (mean, max, median, and numeric variables) using StandardScalar so that each feature has a mean of 0 and a st. deviation of 1.
- Column Transformer to apply different data preprocessing steps to different columns of an array or pandas DataFrame at the same time, and create a single object to then call fit_transform() in order to both learn parameters from the data and modify the data.
Export the dataset

"""

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def convert(df) -> pd.DataFrame:
    """
    This method converts columns holding string date variables into numerical representations 
    like age in years or the month. Standardizes categorical variables.
    """

    df = df.copy()

    if 'date_decision' in df.columns:
        # Takes in a date like 2020-04-28 and converts it to datetime object to extract the month and add it to the dataset
        df['date_decision'] = pd.to_datetime(df['date_decision'])
        df['decision_month'] = df['date_decision'].dt.month
    if 'birth_259D' in df.columns:
        # Converts column holding birth date to their age in years
        # birth_259D is the applicant's raw date of birth in a string (1969-08-15) and what we can do here is to take the
        # date of application - date of applicant's birth to calculate their age
        df['birth_259D'] = pd.to_datetime(df['birth_259D'])
        df['age_in_years'] = (df['date_decision'] - df['birth_259D']).dt.days / 365.2425
        df.drop(columns=['birth_259D'])
    if 'approvaldate_319D' in df.columns:
        # approvaldate_319D is when the applicant's previous loan/credit line was approved
        # We want to find the # of days since prior approval by doing
        # date of decision - approval date
        df['approvaldate_319D'] = pd.to_datetime(df['approvaldate_319D'])
        df['days_since_prior_arrival'] = (df['date_decision'] - df['approvaldate_319D']).dt.days
        df.drop(columns=['approvaldate_319D'])

    if 'date_decision' in df.columns:
        df.drop(columns=['date_decision'])

    for col in ['decision_month', 'age_in_years', 'days_since_prior_arrival']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df

        
        






