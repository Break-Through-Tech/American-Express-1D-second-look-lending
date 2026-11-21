"""
Streamlit demo for the Second-Look Lending models.

Run from the repository root:

    streamlit run app.py

The comparison pages read holdout predictions written by the model scripts. The
applicant page uses packaged models when they can be loaded and otherwise falls
back to the matching precomputed holdout prediction.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = ROOT_DIR / "data" / "predictions"
ARTIFACTS_DIRS = [ROOT_DIR / "artifacts", ROOT_DIR / "models" / "artifacts"]
FEATURES_PATH = ROOT_DIR / "data" / "features" / "train_features.csv"

sys.path.insert(0, str(ROOT_DIR))
from models.predict_logistic import score_dataframe
from models.splits import get_features_target_segment

BASELINE = "logistic_regression"
LABELS = {
    "logistic_regression": "Logistic Regression (baseline)",
    "lightgbm": "LightGBM",
}
PREDICTION_SPECS = {
    "logistic_regression": {
        "filenames": (
            "logistic_holdout_predictions.csv",
            "holdout_logistic_regression.csv",
        ),
        "probability_columns": (
            "predicted_default_probability",
            "p_default",
            "pd",
        ),
    },
    "lightgbm": {
        "filenames": ("holdout_lightgbm.csv",),
        "probability_columns": ("pd", "p_default", "predicted_default_probability"),
    },
}
ARTIFACT_SPECS = {
    "logistic_regression": ("logistic_pipeline.joblib", "logistic_regression.joblib"),
    "lightgbm": ("lightgbm.joblib",),
}


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").title())


def find_artifact(filenames: tuple[str, ...]) -> Path | None:
    for folder in ARTIFACTS_DIRS:
        for filename in filenames:
            candidate = folder / filename
            if candidate.is_file():
                return candidate
    return None


def read_prediction_file(path: Path, model_name: str, probability_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"case_id", "target", "is_thin_file"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if frame["case_id"].isna().any() or frame["case_id"].duplicated().any():
        raise ValueError(f"{path.name} must contain one non-missing row per case_id")

    probability_column = next(
        (column for column in probability_columns if column in frame.columns),
        None,
    )
    if probability_column is None:
        raise ValueError(
            f"{path.name} needs one probability column from {list(probability_columns)}"
        )

    output = frame[["case_id", "target", "is_thin_file", probability_column]].copy()
    output = output.rename(columns={probability_column: f"pd_{model_name}"})
    probabilities = output[f"pd_{model_name}"]
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError(f"{path.name} contains missing or invalid probabilities")
    return output


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    """Return one row per holdout applicant and one probability column per model."""
    model_frames = []
    for model_name, spec in PREDICTION_SPECS.items():
        path = next(
            (
                PREDICTIONS_DIR / filename
                for filename in spec["filenames"]
                if (PREDICTIONS_DIR / filename).is_file()
            ),
            None,
        )
        if path is not None:
            model_frames.append(
                (
                    model_name,
                    read_prediction_file(path, model_name, spec["probability_columns"]),
                )
            )

    if not model_frames:
        return None

    _, merged = model_frames[0]
    reference = merged.set_index("case_id")[["target", "is_thin_file"]].sort_index()
    for model_name, frame in model_frames[1:]:
        candidate = frame.set_index("case_id")[["target", "is_thin_file"]].sort_index()
        if not reference.equals(candidate):
            raise ValueError(
                f"{label(model_name)} predictions do not cover the same holdout applicants "
                "with matching target and segment values."
            )
        merged = merged.merge(
            frame[["case_id", f"pd_{model_name}"]],
            on="case_id",
            validate="one_to_one",
        )
    return merged


@st.cache_resource
def load_models(model_names: tuple[str, ...]) -> tuple[dict, dict]:
    """Load packaged models without allowing an optional model to crash the app."""
    loaded = {}
    errors = {}
    for model_name in model_names:
        filenames = ARTIFACT_SPECS.get(model_name)
        if filenames is None:
            continue
        path = find_artifact(filenames)
        if path is None:
            errors[model_name] = "No packaged model artifact was found."
            continue
        try:
            loaded[model_name] = joblib.load(path)
        except Exception as error:
            errors[model_name] = f"Could not load {path.name}: {error}"
    return loaded, errors


@st.cache_data
def load_applicants(case_ids: tuple) -> pd.DataFrame:
    """Load the selected applicants' raw feature rows, indexed by case_id."""
    if not FEATURES_PATH.is_file():
        return pd.DataFrame()
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    return features[features["case_id"].isin(case_ids)].set_index("case_id")


def apply_lightgbm_prep(features: pd.DataFrame, prep_rules: dict) -> pd.DataFrame:
    """Apply the preprocessing rules stored with the LightGBM artifact."""
    prepared = features.reindex(columns=prep_rules["feature_columns"]).copy()
    for column in prepared.select_dtypes(include="bool").columns:
        prepared[column] = prepared[column].astype(int)
    for column, categories in prep_rules["categories_by_column"].items():
        prepared[column] = pd.Categorical(prepared[column], categories=categories)
    return prepared.drop(columns=prep_rules["constant_columns"], errors="ignore")


def score_applicant(
    case_id: int,
    feature_rows: pd.DataFrame,
    loaded_models: dict,
    model_names: list[str],
    predictions: pd.DataFrame,
) -> tuple[dict, dict]:
    """Score one applicant, falling back to saved holdout probabilities when needed."""
    results = {}
    errors = {}
    raw_row = None
    if not feature_rows.empty and case_id in feature_rows.index:
        raw_row = feature_rows.loc[[case_id]].reset_index()

    if "logistic_regression" in loaded_models and raw_row is not None:
        try:
            artifact = loaded_models["logistic_regression"]
            if isinstance(artifact, dict) and "preprocessor" in artifact:
                scored = score_dataframe(raw_row, artifact)
                results["logistic_regression"] = float(
                    scored.iloc[0]["predicted_default_probability"]
                )
            else:
                features, _, _ = get_features_target_segment(raw_row)
                results["logistic_regression"] = float(
                    artifact.predict_proba(features)[0, 1]
                )
        except Exception as error:
            errors["logistic_regression"] = f"Packaged scoring failed: {error}"

    if "lightgbm" in loaded_models and raw_row is not None:
        try:
            artifact = loaded_models["lightgbm"]
            features, _, _ = get_features_target_segment(raw_row)
            prepared = apply_lightgbm_prep(features, artifact["prep_rules"])
            booster = artifact["model"]
            results["lightgbm"] = float(
                booster.predict(prepared, num_iteration=booster.best_iteration)[0]
            )
        except Exception as error:
            errors["lightgbm"] = f"Packaged scoring failed: {error}"

    saved_row = predictions.loc[predictions["case_id"] == case_id].iloc[0]
    for model_name in model_names:
        if model_name not in results:
            results[model_name] = float(saved_row[f"pd_{model_name}"])
    return results, errors


st.set_page_config(page_title="Second-Look Lending", layout="wide")
st.title("Second-Look Lending")

try:
    preds = load_predictions()
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Prediction files could not be loaded: {error}")
    st.stop()

if preds is None:
    st.warning(
        f"No holdout prediction files were found in `{PREDICTIONS_DIR}`. Run "
        "`python models/logistic_baseline.py` and/or "
        "`python notebooks/train_lightgbm.py` first."
    )
    st.stop()

models = [column.removeprefix("pd_") for column in preds if column.startswith("pd_")]
st.caption(f"Testing on {len(preds):,} held-out applicants (weeks 76-91)")

PAGES = ["Model performance", "Approval decision", "Score an applicant"]
st.session_state.setdefault("page", PAGES[0])


def go_to(name: str) -> None:
    st.session_state["page"] = name


st.sidebar.markdown("**Pages**")
for name in PAGES:
    st.sidebar.button(
        name,
        on_click=go_to,
        args=(name,),
        width="stretch",
        type="primary" if st.session_state["page"] == name else "secondary",
    )
page = st.session_state["page"]

st.session_state.setdefault("threshold_val", 0.2)


def threshold_slider() -> float:
    value = st.slider(
        "Approve if predicted probability of default is below",
        0.0,
        1.0,
        step=0.01,
        value=st.session_state["threshold_val"],
    )
    st.session_state["threshold_val"] = value
    return value


if page == "Model performance":
    segments = {
        "All": preds.index == preds.index,
        "Established": preds["is_thin_file"] == 0,
        "Thin-file": preds["is_thin_file"] == 1,
    }
    rows = []
    for segment_name, mask in segments.items():
        for model_name in models:
            target = preds.loc[mask, "target"]
            probability = preds.loc[mask, f"pd_{model_name}"]
            rows.append(
                {
                    "Segment": segment_name,
                    "Model": label(model_name),
                    "ROC-AUC ↑": roc_auc_score(target, probability),
                    "LogLoss ↓": log_loss(target, probability),
                    "Brier ↓": brier_score_loss(target, probability),
                }
            )
    scores = pd.DataFrame(rows)

    st.subheader("Scores")
    segment = st.radio("Segment", list(segments), horizontal=True)
    st.dataframe(
        scores[scores["Segment"] == segment]
        .drop(columns="Segment")
        .set_index("Model")
        .style.format("{:.4f}")
    )

    if BASELINE in models and len(models) > 1:
        baseline = scores[
            (scores["Segment"] == segment) & (scores["Model"] == label(BASELINE))
        ].iloc[0]
        columns = st.columns(len(models) - 1)
        for column, model_name in zip(
            columns, [name for name in models if name != BASELINE]
        ):
            row = scores[
                (scores["Segment"] == segment)
                & (scores["Model"] == label(model_name))
            ].iloc[0]
            column.metric(
                f"{label(model_name)} ROC-AUC vs. baseline",
                f"{row['ROC-AUC ↑']:.4f}",
                f"{row['ROC-AUC ↑'] - baseline['ROC-AUC ↑']:+.4f}",
            )

    left, _ = st.columns(2)
    with left:
        st.subheader("ROC curve")
        mask = segments[segment]
        figure, axis = plt.subplots(figsize=(5, 4))
        for model_name in models:
            false_positive_rate, true_positive_rate, _ = roc_curve(
                preds.loc[mask, "target"], preds.loc[mask, f"pd_{model_name}"]
            )
            axis.plot(false_positive_rate, true_positive_rate, label=label(model_name))
        axis.plot([0, 1], [0, 1], "--", color="lightgray")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.legend()
        st.pyplot(figure)
        plt.close(figure)

elif page == "Approval decision":
    st.subheader("Approval decision")
    threshold = threshold_slider()
    st.caption(
        "This view approves applicants whose predicted probability of default is "
        f"below {threshold:.0%}."
    )
    repaid = preds["target"] == 0
    rows = []
    for model_name in models:
        approved = preds[f"pd_{model_name}"] < threshold
        established = approved[repaid & (preds["is_thin_file"] == 0)].mean()
        thin_file = approved[repaid & (preds["is_thin_file"] == 1)].mean()
        rows.append(
            {
                "Model": label(model_name),
                "Approval rate": approved.mean(),
                "Default rate (approved)": (
                    preds.loc[approved, "target"].mean() if approved.any() else 0.0
                ),
                "Repayers approved: established": established,
                "Repayers approved: thin-file": thin_file,
                "Inclusion gap": established - thin_file,
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.1%}"))
    st.caption(
        "Inclusion gap = approval rate among applicants who repaid, established "
        "minus thin-file. Closer to 0 is fairer."
    )

elif page == "Score an applicant":
    st.subheader("Score an applicant")
    threshold = threshold_slider()
    candidates = preds.groupby("is_thin_file").head(200)
    loaded_models, model_load_errors = load_models(tuple(models))
    feature_rows = load_applicants(tuple(candidates["case_id"]))

    for model_name, message in model_load_errors.items():
        st.info(
            f"{label(model_name)} packaged scoring is unavailable ({message}) "
            "Using its saved holdout probability instead."
        )
    if feature_rows.empty and loaded_models:
        st.info(
            f"`{FEATURES_PATH}` is unavailable, so saved holdout probabilities are being used."
        )

    applicant_type = st.radio(
        "Applicant type", ["Thin-file", "Established"], horizontal=True
    )
    pool = candidates[
        candidates["is_thin_file"] == int(applicant_type == "Thin-file")
    ]
    case_id = st.selectbox("Applicant (case_id)", pool["case_id"])
    target_by_case = pool.set_index("case_id")["target"]
    outcome = "defaulted" if bool(target_by_case.at[case_id]) else "repaid"
    st.markdown(f"**Historical holdout outcome:** {outcome}")

    probabilities, scoring_errors = score_applicant(
        case_id, feature_rows, loaded_models, models, preds
    )
    for model_name, message in scoring_errors.items():
        st.info(
            f"{label(model_name)} {message} Using its saved holdout probability instead."
        )

    columns = st.columns(len(probabilities))
    for column, (model_name, probability) in zip(columns, probabilities.items()):
        if model_name == BASELINE:
            decision = (
                "Approve full loan"
                if probability < threshold
                else "Send to second look"
            )
        else:
            decision = "Approve" if probability < threshold else "Decline"
        column.metric(
            label(model_name),
            f"{probability:.1%} chance of default",
            decision,
            delta_color="normal" if probability < threshold else "inverse",
        )
    st.caption(f"The decision view uses the threshold shown above ({threshold:.0%}).")
