"""
Task 3: Standardize categorical variables, normalize numerical features. Organize the different categories (one-hot encoding) and scale if needed.

convert()
- Separate the columns into distinct lists filtering out the target variable (target), ID columns (case_id), and variables that Melinda already one-hot encoded from task 1.
- Convert columns holding string date variables into numerical representations like age in years or the month to avoid crashes with scikit-learn later on.

main()
- Apply one-hot encoding to the categorical variables that have not been encoded yet.
- Scale the numeric columns (mean, max, median, and numeric variables) using StandardScalar so that each feature has a mean of 0 and a st. deviation of 1.
- Column Transformer to apply different data preprocessing steps to different columns of an array or pandas DataFrame at the same time, and create a single object to then call fit_transform() in order to both learn parameters from the data and modify the data.
- Export the dataset

"""

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

INPUT_DIR = "data/cleaned"
OUTPUT_DIR = "data/preprocessed"

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

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the cleaned datasets from clean_data
    train = pd.read_csv(f"{INPUT_DIR}/train_cleaned.csv")
    test = pd.read_csv(f"{INPUT_DIR}/test_cleaned.csv")

    # Apply prepreprocessing
    train = convert(train)
    test = convert(test)

    # Some columns must be left out of scaling of the numeric columns
    exclusions = ['case_id', 'target', 'MONTH', 'WEEK_NUM']

    # string/categorical columns w/ exclusion filtering
    categorical_cols = train.select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in exclusions]

    # numeric columns w/ exclusion filtering
    numeric_cols = train.select_dtypes(include=['number', 'bool']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclusions]

    print(f"There are {len(numeric_cols)} numerical columns and {len(categorical_cols)} categorical columns")

    # Column Transformer to apply different data preprocessing steps to different columns of an array or pandas DataFrame at the same time, 
    # and create a single object to then call fit_transform() in order to both learn parameters from the data and modify the data.

    # Scale the numeric columns (mean, max, median, and numeric variables) using StandardScalar so that each feature has a mean of 0 and a st. deviation of 1.

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols), 
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    # We separate exclusion columns to prevent them from being altered by standard scalar and one hot encoder

    # Get rid of the protected columns (case_id, target, etc) in train, process training data
    exclude = []
    for col in exclusions:
        if col in train.columns:
            exclude.append(col)

    X_train = train.drop(columns=exclude)
    y_train_id = train[exclude].copy()

    # Get rid of the protected columns in test, process testing data
    exclude_2 = []
    for col in exclusions:
        if col in test.columns:
            exclude_2.append(col)

    X_test = test.drop(columns=exclude_2)
    y_test_id = test[exclude_2].copy()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)  # transform for testing data

    new_categorical_features = preprocessor.named_transformers_['categorical'].get_feature_names_out(categorical_cols)
    all_features = numeric_cols + list(new_categorical_features)

    train_preprocessed = pd.DataFrame(X_train_transformed, columns=all_features)
    test_preprocessed = pd.DataFrame(X_test_transformed, columns=all_features)

    for col in exclusions:
        if col in y_test_id:
            test_preprocessed[col] = y_test_id[col].values
        elif col in y_train_id:
            train_preprocessed[col] = y_train_id[col].values

    train_preprocessed.to_csv(f"{OUTPUT_DIR}/train_preprocessed.csv", index=False)
    test_preprocessed.to_csv(f"{OUTPUT_DIR}/test_preprocessed.csv", index=False)

if __name__ == "__main__":
    main()








