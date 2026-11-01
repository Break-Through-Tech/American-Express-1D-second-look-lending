"""
demo.py

Demo of the Second-Look Lending models. From the repo root:

    streamlit run app/demo.py

Comparison: reads data/predictions/holdout_<model name>.csv, written by each models/train_*.py,
with columns case_id, target, is_thin_file, pd (predicted probability of default).

Score an applicant: loads the packaged models from artifacts/ and the applicants' rows from
data/features (logistic regression's input) and data/lgbm (LightGBM's input), only when that page is opened.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT_DIR / "data" / "predictions"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FEATURES_PATH = ROOT_DIR / "data" / "features" / "train_features.csv"   # logistic regression's input
LGBM_PATH = ROOT_DIR / "data" / "features" / "train_features.csv"       # LightGBM's input (no specific set for lightgbm yet)

# the saved models need the same prep code they were trained with
sys.path.insert(0, str(ROOT_DIR))
from models.splits import get_features_target_segment
from notebooks.train_lightgbm import apply_prep

BASELINE = "logistic_regression"
LABELS = {"logistic_regression": "Logistic Regression (baseline)", "lightgbm": "LightGBM"}


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    """One row per held-out applicant, one pd_<model> column per model."""
    merged = None
    for path in sorted(PREDICTIONS_DIR.glob("holdout_*.csv")):
        name = path.stem.removeprefix("holdout_")
        df = pd.read_csv(path).rename(columns={"pd": f"pd_{name}"})
        merged = df if merged is None else merged.merge(df[["case_id", f"pd_{name}"]], on="case_id")
    return merged


@st.cache_resource
def load_models() -> dict:
    """Load each packaged model once, not on every click."""
    return {
        "logistic_regression": joblib.load(ARTIFACTS_DIR / "logistic_regression.joblib"),
        "lightgbm": joblib.load(ARTIFACTS_DIR / "lightgbm.joblib"),
    }


@st.cache_data
def load_applicants(case_ids: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The chosen applicants' rows in each model's input format, indexed by case_id."""
    features_rows = pd.read_csv(FEATURES_PATH)
    lgbm_rows = pd.read_csv(LGBM_PATH)
    return (features_rows[features_rows["case_id"].isin(case_ids)].set_index("case_id"),
            lgbm_rows[lgbm_rows["case_id"].isin(case_ids)].set_index("case_id"))


def score_applicant(case_id: int, features_rows: pd.DataFrame, lgbm_rows: pd.DataFrame) -> dict:
    """Probability of default from each packaged model for one applicant."""
    models = load_models()
    X_base, _, _ = get_features_target_segment(features_rows.loc[[case_id]].reset_index())
    X_lgbm, _, _ = get_features_target_segment(lgbm_rows.loc[[case_id]].reset_index())
    lgbm, prep_rules = models["lightgbm"]["model"], models["lightgbm"]["prep_rules"]
    return {
        "logistic_regression": models["logistic_regression"].predict_proba(X_base)[0, 1],
        "lightgbm": lgbm.predict(apply_prep(X_lgbm, prep_rules), num_iteration=lgbm.best_iteration)[0],
    }


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").title())


st.set_page_config(page_title="Second-Look Lending", layout="wide")
st.title("Second-Look Lending")

preds = load_predictions()
if preds is None:
    st.warning(f"No prediction files found in `{PREDICTIONS_DIR}`. Run `python models/train_logistic_regression.py` "
               "and `python models/train_lightgbm.py` first.")
    st.stop()

models = [c.removeprefix("pd_") for c in preds.columns if c.startswith("pd_")]
st.caption(f"Testing on {len(preds):,} held-out applicants (weeks 76-91)")

# only the selected page's code runs
# ---------- navigation: one full-width button per page, the current page highlighted ----------
PAGES = ["Model performance", "Approval decision", "Score an applicant"]
st.session_state.setdefault("page", PAGES[0])


def go_to(name: str) -> None:
    st.session_state["page"] = name


st.sidebar.markdown("**Pages**")
for name in PAGES:
    # on_click runs before the page redraws, so the highlight moves on the same click
    st.sidebar.button(name, on_click=go_to, args=(name,), width="stretch",
                      type="primary" if st.session_state["page"] == name else "secondary")
page = st.session_state["page"]

# the approval threshold shared by two pages
st.session_state.setdefault("threshold", 0.2)
st.session_state["threshold"] = st.session_state["threshold"]


def threshold_slider() -> float:
    st.slider("Approve if predicted probability of default is at or below", 0.0, 1.0, step=0.01, key="threshold")
    return st.session_state["threshold"]

# ---------- scores + ROC curve ----------
if page == "Model performance":
    segments = {"All": preds.index == preds.index, "Established": preds["is_thin_file"] == 0,
                "Thin-file": preds["is_thin_file"] == 1}
    rows = []
    for segment, mask in segments.items():
        for m in models:
            y, p = preds.loc[mask, "target"], preds.loc[mask, f"pd_{m}"]
            rows.append({"Segment": segment, "Model": label(m), "ROC-AUC ↑": roc_auc_score(y, p),
                         "LogLoss ↓": log_loss(y, p), "Brier ↓": brier_score_loss(y, p)})
    scores = pd.DataFrame(rows)

    st.subheader("Scores")
    segment = st.radio("Segment", list(segments), horizontal=True)
    st.dataframe(scores[scores["Segment"] == segment].drop(columns="Segment").set_index("Model").style.format("{:.4f}"))

    if BASELINE in models and len(models) > 1:
        base = scores[(scores["Segment"] == segment) & (scores["Model"] == label(BASELINE))].iloc[0]
        cols = st.columns(len(models) - 1)
        for col, m in zip(cols, [m for m in models if m != BASELINE]):
            row = scores[(scores["Segment"] == segment) & (scores["Model"] == label(m))].iloc[0]
            col.metric(f"{label(m)} ROC-AUC vs. baseline", f"{row['ROC-AUC ↑']:.4f}",
                       f"{row['ROC-AUC ↑'] - base['ROC-AUC ↑']:+.4f}")

    left, _ = st.columns(2)
    with left:
        st.subheader("ROC curve")
        mask = segments[segment]
        fig, ax = plt.subplots(figsize=(5, 4))
        for m in models:
            fpr, tpr, _ = roc_curve(preds.loc[mask, "target"], preds.loc[mask, f"pd_{m}"])
            ax.plot(fpr, tpr, label=label(m))
        ax.plot([0, 1], [0, 1], "--", color="lightgray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        st.pyplot(fig)

# ---------- approval decision ----------
elif page == "Approval decision":
    st.subheader("Approval decision")
    threshold = threshold_slider()
    st.caption(f"Approving applicants whose predicted probability of default is at or below {threshold:.2f} "
               "(change it in the sidebar).")
    repaid = preds["target"] == 0
    rows = []
    for m in models:
        approved = preds[f"pd_{m}"] <= threshold
        est = approved[repaid & (preds["is_thin_file"] == 0)].mean()
        thin = approved[repaid & (preds["is_thin_file"] == 1)].mean()
        rows.append({"Model": label(m), "Approval rate": approved.mean(),
                     "Default rate (approved)": preds.loc[approved, "target"].mean() if approved.any() else 0.0,
                     "Repayers approved: established": est, "Repayers approved: thin-file": thin,
                     "Inclusion gap": est - thin})
    st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.1%}"))
    st.caption("Inclusion gap = approval rate among applicants who repaid, established minus thin-file. "
               "Closer to 0 is fairer.")

# ---------- score an applicant ----------
elif page == "Score an applicant":
    st.subheader("Score an applicant")
    threshold = threshold_slider()
    candidates = preds.groupby("is_thin_file").head(200)   # 200 of each type, read from the data once
    features_rows, lgbm_rows = load_applicants(tuple(candidates["case_id"]))
    applicant_type = st.radio("Applicant type", ["Thin-file", "Established"], horizontal=True)
    pool = candidates[candidates["is_thin_file"] == (applicant_type == "Thin-file")]
    case_id = st.selectbox("Applicant (case_id)", pool["case_id"])
    target_by_case = pool.set_index("case_id")["target"]
    outcome = "defaulted" if bool(target_by_case.at[case_id]) else "repaid"
    st.markdown(f"**Actual outcome:** {outcome}")

    probs = score_applicant(case_id, features_rows, lgbm_rows)
    cols = st.columns(len(probs))
    for col, (m, prob) in zip(cols, probs.items()):
        decision = "Approve" if prob <= threshold else "Decline"
        col.metric(label(m), f"{prob:.1%} chance of default", decision,
                   delta_color="normal" if decision == "Approve" else "inverse")
    st.caption(f"Approve / decline uses the threshold in the sidebar ({threshold:.2f}).")