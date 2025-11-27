# src/shap_explain.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# shap import (raise helpful error if missing)
try:
    import shap
except Exception as e:
    raise ImportError("shap is not installed. Install with: python -m pip install shap") from e

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")
os.makedirs(MODELS_DIR, exist_ok=True)

# Configs (must match training)
MISSING_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
N_AGE_BINS = 5

def age_bins_median_impute(X_train_ref: pd.DataFrame, X: pd.DataFrame, columns, n_bins=5):
    """
    Compute age-bin medians on reference dataset (X_train_ref) and replace zeros in X accordingly.
    Returns a new DataFrame (X imputed).
    """
    X = X.copy()
    # ensure columns we will write into are float to avoid dtype warnings
    X[columns] = X[columns].astype(float)

    # compute bins from reference using qcut (same as training)
    _, bins = pd.qcut(X_train_ref["Age"], q=n_bins, retbins=True, duplicates="drop")
    X["_age_bin"] = pd.cut(X["Age"], bins=bins, labels=False, include_lowest=True)

    # compute medians per bin on reference excluding zeros
    medians = {}
    for b in range(len(bins)-1):
        if b == len(bins)-2:
            mask = (X_train_ref["Age"] >= bins[b]) & (X_train_ref["Age"] <= bins[b+1])
        else:
            mask = (X_train_ref["Age"] >= bins[b]) & (X_train_ref["Age"] < bins[b+1])
        col_medians = {}
        for col in columns:
            vals = X_train_ref.loc[mask, col].replace(0, pd.NA).dropna()
            if len(vals) == 0:
                overall = X_train_ref[col].replace(0, pd.NA).dropna()
                col_medians[col] = float(overall.median()) if len(overall) > 0 else 0.0
            else:
                col_medians[col] = float(vals.median())
        medians[b] = col_medians

    # global fallbacks
    global_medians = {}
    for col in columns:
        vals = X_train_ref[col].replace(0, pd.NA).dropna()
        global_medians[col] = float(vals.median()) if len(vals) > 0 else 0.0

    # replace zeros in X based on bin medians
    for idx in X.index:
        b = X.at[idx, "_age_bin"]
        for col in columns:
            val = X.at[idx, col]
            if (val == 0) or pd.isna(val):
                if pd.isna(b):
                    X.at[idx, col] = global_medians[col]
                else:
                    bin_idx = int(b)
                    X.at[idx, col] = medians.get(bin_idx, {}).get(col, global_medians[col])
    X = X.drop(columns=["_age_bin"])
    return X

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1.0)
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Is_Obese"] = (df["BMI"] >= 30).astype(int)
    df["Is_High_Glucose"] = (df["Glucose"] >= 125).astype(int)
    return df

def main():
    # load artifacts
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    model_path = os.path.join(MODELS_DIR, "best_model_lightgbm.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Saved scaler/model not found in models/. Run training first.")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # load the reference dataset (the one used during training)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # impute zeros per age-bin using reference (use X as both ref and dataset)
    X_imp = age_bins_median_impute(X, X, MISSING_ZERO_COLS, n_bins=N_AGE_BINS)

    # add engineered features
    X_feat = add_engineered_features(X_imp)

    # ensure final column order matches scaler/model training order
    final_columns = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age",
        "Glucose_Insulin_Ratio","BMI_Age","Is_Obese","Is_High_Glucose"
    ]
    X_final = X_feat[final_columns]

    # scale
    X_scaled = scaler.transform(X_final)
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_columns)

    # create SHAP explainer
    explainer = shap.Explainer(model)
    print("Computing SHAP values (this may take a moment)...")
    shap_values = explainer(X_scaled_df)

    # summary plot (dot/violin)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_final, show=False)
    plt.tight_layout()
    summary_path = os.path.join(MODELS_DIR, "shap_summary.png")
    plt.savefig(summary_path)
    plt.close()
    print("Saved SHAP summary plot ->", summary_path)

    # mean absolute shap importance (bar)
    plt.figure(figsize=(8,6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    bar_path = os.path.join(MODELS_DIR, "shap_bar.png")
    plt.savefig(bar_path)
    plt.close()
    print("Saved SHAP bar plot ->", bar_path)

  # Determine top feature automatically
top_feature = None
try:
    importance = np.abs(shap_values.values).mean(axis=0)
    top_feature = X_final.columns[np.argmax(importance)]
    print("Top feature for dependence plot:", top_feature)
except:
    print("Could not auto-detect top feature")

# Dependence plot
try:
    if top_feature:
        plt.figure(figsize=(8,6))
        shap.plots.scatter(shap_values[:, top_feature], color=shap_values, show=False)
        plt.tight_layout()
        dep_path = os.path.join(MODELS_DIR, "shap_dependence.png")
        plt.savefig(dep_path)
        plt.close()
        print("Saved SHAP dependence plot ->", dep_path)
    else:
        print("Dependence plot skipped — no top feature detected.")
except Exception as e:
    print("Could not create dependence plot:", e)


if __name__ == "__main__":
    main()
