"""
Score an applicant feature CSV with the saved Logistic Regression pipeline.

Run from the repository root:

    python models/predict_logistic.py
    python models/predict_logistic.py --input path/to/features.csv --output path/to/predictions.csv

The input must contain the same raw feature columns produced by notebooks/features.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT_DIR / "models" / "artifacts" / "logistic_pipeline.joblib"
DEFAULT_INPUT = ROOT_DIR / "data" / "features" / "test_features.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "predictions" / "logistic_score_predictions.csv"


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    """Convert probabilities to finite log-odds for sigmoid calibration."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def apply_calibration(artifact: dict, probabilities: np.ndarray) -> np.ndarray:
    """Apply the calibration method stored with the trained model."""
    method = artifact["calibration_method"]
    calibrator = artifact["calibrator"]
    if method == "raw":
        calibrated = probabilities
    elif method == "sigmoid":
        calibrated = calibrator.predict_proba(
            probability_logit(probabilities).reshape(-1, 1)
        )[:, 1]
    elif method == "isotonic":
        calibrated = calibrator.predict(probabilities)
    else:
        raise ValueError(f"Unsupported calibration method in artifact: {method}")
    return np.clip(np.asarray(calibrated, dtype=float), 1e-6, 1 - 1e-6)


def score_dataframe(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Return default probabilities and Stage 1 routes for an applicant dataframe."""
    required_columns = list(
        artifact.get("raw_feature_columns", artifact["preprocessor"].feature_names_in_)
    )
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Input is missing model columns: {missing_columns}")

    model_features = frame[required_columns].copy()
    numeric_columns = model_features.select_dtypes(include=["number", "bool"]).columns
    for column in numeric_columns:
        model_features[column] = model_features[column].replace(-1, np.nan)
    model_features = model_features.replace([np.inf, -np.inf], np.nan)
    transformed = np.asarray(
        artifact["preprocessor"].transform(model_features),
        dtype=np.float32,
    )
    raw_probabilities = artifact["model"].predict_proba(transformed)[:, 1]
    probabilities = apply_calibration(artifact, raw_probabilities)
    threshold = float(artifact["stage1_threshold"])

    approved = probabilities < threshold
    predictions = pd.DataFrame(
        {
            "predicted_default_probability": probabilities,
            "first_pass_approve": approved.astype(int),
            "refer_to_second_look": (~approved).astype(int),
            "stage1_route": np.where(
                approved,
                "approve_full_loan",
                "send_to_second_look",
            ),
        },
        index=frame.index,
    )
    if "case_id" in frame:
        predictions.insert(0, "case_id", frame["case_id"].to_numpy())
    if "is_thin_file" in frame:
        predictions["is_thin_file"] = frame["is_thin_file"].to_numpy()
    return predictions


def run(input_path: Path, output_path: Path, artifact_path: Path) -> None:
    """Load an artifact and feature CSV, then write reproducible predictions."""
    artifact = joblib.load(artifact_path)
    applicants = pd.read_csv(input_path, low_memory=False)
    predictions = score_dataframe(applicants, artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, float_format="%.8f")
    print(f"Scored {len(predictions):,} applicants")
    print(f"Stage 1 policy: {artifact.get('selected_policy', 'not recorded')}")
    print(f"Stage 1 threshold: {artifact['stage1_threshold']:.1%}")
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    arguments = parser.parse_args()
    run(arguments.input, arguments.output, arguments.artifact)
