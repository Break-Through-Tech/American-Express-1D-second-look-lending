import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay

from splits import make_splits, get_features_target_segment

"""
Random Forest model for second-look-lending
some of my notes for this model: https://docs.google.com/document/d/1rBe35bbr0PK3zZfL4om1B8rlPMjfe53hSjYIFfDgEUk/edit?usp=sharing 
"""

def main():
    train_df = pd.read_csv("data/preprocessed/train_preprocessed.csv")
    test_df = pd.read_csv("data/preprocessed/test_preprocessed.csv")

    splits = make_splits(train_df, test_df)

    X_train, y_train, seg_train = get_features_target_segment(splits["train"])
    X_tune, y_tune, seg_tune = get_features_target_segment(splits["tune"])
    X_holdout, y_holdout, seg_holdout = get_features_target_segment(splits["holdout"])

    classifier = RandomForestClassifier(n_estimators=100, random_state=1234, n_jobs=-1) # put all possible cpu cores to this job
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict_proba(X_tune)
    predictions = list(y_pred[:, -1]) # get last column

    auc_score = roc_auc_score(y_tune, predictions)

    print(f"auc score: {auc_score}")
    # result -> aug score of 0.75 (moderate)

if __name__ == "__main__":
    main()