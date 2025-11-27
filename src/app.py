# src/app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# Paths & constants
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")  # reference dataset used for imputation
MISSING_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
N_AGE_BINS = 5

# Cache artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model_lightgbm.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    ref_df = pd.read_csv(DATA_PATH)
    return model, scaler, ref_df

def age_bin_median_impute_single(input_df: pd.DataFrame, ref_df: pd.DataFrame, columns, n_bins=5):
    """
    Use reference dataset to compute age bins and medians (train-time behavior approximation),
    then replace zeros in input_df per corresponding bin. Returns a new DataFrame.
    """
    # prepare bins from reference (qcut used in training)
    _, bins = pd.qcut(ref_df["Age"], q=n_bins, retbins=True, duplicates="drop")
    # assign bins to input using same bin edges
    input_df = input_df.copy()
    input_df["_age_bin"] = pd.cut(input_df["Age"], bins=bins, labels=False, include_lowest=True)

    # compute medians in reference per bin (ignoring zeros)
    medians = {}
    for b in range(len(bins)-1):
        mask = (ref_df["Age"] >= bins[b]) & (ref_df["Age"] <= bins[b+1]) if b == len(bins)-2 else (ref_df["Age"] >= bins[b]) & (ref_df["Age"] < bins[b+1])
        col_medians = {}
        for col in columns:
            vals = ref_df.loc[mask, col].replace(0, pd.NA).dropna()
            if len(vals) == 0:
                overall = ref_df[col].replace(0, pd.NA).dropna()
                col_medians[col] = float(overall.median()) if len(overall) > 0 else 0.0
            else:
                col_medians[col] = float(vals.median())
        medians[b] = col_medians

    # global fallback medians
    global_medians = {}
    for col in columns:
        vals = ref_df[col].replace(0, pd.NA).dropna()
        global_medians[col] = float(vals.median()) if len(vals) > 0 else 0.0

    # ensure input columns that will receive medians are float dtype to avoid dtype warnings
    input_df[columns] = input_df[columns].astype(float)

    # replace zeros in input_df using medians
    for idx in input_df.index:
        b = input_df.at[idx, "_age_bin"]
        for col in columns:
            val = input_df.at[idx, col]
            if (val == 0) or pd.isna(val):
                if pd.isna(b):
                    input_df.at[idx, col] = global_medians[col]
                else:
                    bin_idx = int(b)
                    med = medians.get(bin_idx, {}).get(col, global_medians[col])
                    input_df.at[idx, col] = float(med)

    input_df = input_df.drop(columns=["_age_bin"])
    return input_df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1.0)
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Is_Obese"] = (df["BMI"] >= 30).astype(int)
    df["Is_High_Glucose"] = (df["Glucose"] >= 125).astype(int)
    return df

# Load artifacts once
model, scaler, ref_df = load_artifacts()

st.title("🩺 Diabetes Prediction (LightGBM)")
st.write("This app applies the same preprocessing used during training: age-bin median imputation, feature engineering, scaling.")

# Input widgets
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
gluc = st.number_input("Glucose", min_value=0, max_value=300, value=120)
bp = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
skin = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
ins = st.number_input("Insulin", min_value=0, max_value=1000, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.6)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120, value=28)

if st.button("Predict"):
    # Build input DataFrame (raw features)
    input_df = pd.DataFrame([{
        "Pregnancies": preg,
        "Glucose": gluc,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": ins,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    # Impute zeros using reference age-bin medians
    input_df_imp = age_bin_median_impute_single(input_df, ref_df, MISSING_ZERO_COLS, n_bins=N_AGE_BINS)

    # Add engineered features (must match training pipeline)
    input_df_feat = add_features(input_df_imp)

    # Columns ordering that the scaler/model expect (original + engineered)
    final_columns = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age",
        "Glucose_Insulin_Ratio","BMI_Age","Is_Obese","Is_High_Glucose"
    ]

    # Select final columns and ensure order
    input_df_for_scale = input_df_feat[final_columns]

    # Scale (scaler returns ndarray)
    X_scaled = scaler.transform(input_df_for_scale)

    # Convert to DataFrame with column names so model sees feature names (removes warnings)
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_columns)

    # Predict (pass DataFrame)
    pred = model.predict(X_scaled_df)[0]
    prob = model.predict_proba(X_scaled_df)[0, 1]

    st.subheader("🔍 Prediction Result")
    if pred == 1:
        st.error(f"Patient is likely *Diabetic*  — Probability: {prob:.2f}")
    else:
        st.success(f"Patient is *Not Diabetic* — Probability: {prob:.2f}")
