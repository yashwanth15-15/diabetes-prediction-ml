# robust loader — replace existing load_artifacts and its call in src/app.py
import os, joblib, pandas as pd, streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")  # reference dataset used for imputation

CANDIDATE_MODELS = [
    "best_model_lightgbm.pkl",
    "best_model_smote_impute.pkl",
    "best_model_improved.pkl",
    "diabetes_model.pkl"
]
SCALER_NAME = "scaler.pkl"

@st.cache_resource
def load_artifacts():
    # If reference data missing — return None model/scaler but keep ref_df as None
    if not os.path.exists(DATA_PATH):
        return None, None, None, None

    # load reference data
    ref_df = pd.read_csv(DATA_PATH)

    # find model
    found_model = None
    found_model_name = None
    for mname in CANDIDATE_MODELS:
        mpath = os.path.join(MODELS_DIR, mname)
        if os.path.exists(mpath):
            try:
                found_model = joblib.load(mpath)
                found_model_name = mname
                break
            except Exception as e:
                print(f"Failed to load {mpath}: {e}")
                continue

    # load scaler if present
    scaler = None
    scaler_path = os.path.join(MODELS_DIR, SCALER_NAME)
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Failed to load scaler {scaler_path}: {e}")
            scaler = None

    return found_model, scaler, ref_df, found_model_name

# call loader
model, scaler, ref_df, model_name = load_artifacts()

# if data missing — show friendly message and stop
if ref_df is None:
    st.error(
        "Reference dataset `data/diabetes.csv` not found in the repository.\n\n"
        "This file is required for preprocessing/imputation and SHAP plots.\n\n"
        "Options:\n"
        "1) (Recommended) Upload your local `data/diabetes.csv` to the repo under the `data/` folder and push. See terminal instructions below.\n"
        "2) If you prefer not to add the dataset to GitHub, I can modify the app to use a small built-in sample instead — tell me and I will generate that change."
    )
    st.stop()
