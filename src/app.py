# ----- robust artifact loader (replace existing load_artifacts and its call) -----
import os, joblib, pandas as pd, streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")  # reference dataset used for imputation

# candidate models to try (in order)
CANDIDATE_MODELS = [
    "best_model_lightgbm.pkl",
    "best_model_smote_impute.pkl",
    "best_model_improved.pkl",
    "best_model_smote_impute.pkl",
    "diabetes_model.pkl"
]
SCALER_NAME = "scaler.pkl"

@st.cache_resource
def load_artifacts():
    """
    Try to load first available model & scaler. Return (model, scaler, ref_df, model_name).
    If no model found, return (None, None, ref_df, None).
    """
    # load reference data (used by imputation logic)
    ref_df = pd.read_csv(DATA_PATH)

    # search for model file
    found_model = None
    found_model_name = None
    for mname in CANDIDATE_MODELS:
        mpath = os.path.join(MODELS_DIR, mname)
        if os.path.exists(mpath):
            try:
                model = joblib.load(mpath)
                found_model = model
                found_model_name = mname
                break
            except Exception as e:
                # log but continue to next candidate
                print(f"Failed to load {mpath}: {e}")
                continue

    # try to load scaler if present
    scaler = None
    scaler_path = os.path.join(MODELS_DIR, SCALER_NAME)
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Failed to load scaler {scaler_path}: {e}")
            scaler = None

    return found_model, scaler, ref_df, found_model_name

# load artifacts (safe)
model, scaler, ref_df, model_name = load_artifacts()

# If no model was found, show a friendly message and stop the app (so it doesn't crash)
if model is None:
    st.warning(
        "__Model not found on server.__\n\n"
        "The app is running but predictions are disabled because no model `.pkl` was found in `models/`.\n\n"
        "Options:\n"
        "• Upload a model file (e.g. `best_model_lightgbm.pkl`) to the repo under `models/` and push.\n"
        "• Or let the app run for inspection (you can still see documentation and SHAP plots).\n\n"
        "To add a model now, push it to GitHub (see instructions in the terminal or ask me to provide the exact command)."
    )
    st.stop()
# ---------------------------------------------------------------------------------
