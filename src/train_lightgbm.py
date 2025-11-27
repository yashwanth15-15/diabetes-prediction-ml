# src/train_lightgbm.py
import os
import warnings
from time import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
RANDOM_STATE = 42

# Columns where 0 means missing in PIMA dataset
MISSING_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def add_features(df):
    df = df.copy()
    # Prevent division by zero for Insulin if zero -> will be handled before calling this normally
    df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1.0)
    df["BMI_Age"] = df["BMI"] * df["Age"]
    df["Is_Obese"] = (df["BMI"] >= 30).astype(int)
    df["Is_High_Glucose"] = (df["Glucose"] >= 125).astype(int)  # threshold can be tuned
    return df

def age_bins_median_impute(X_train, X_valid, columns, n_bins=5):
    """
    Replace zeros with median per age-bin computed on X_train.
    """
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    # create age bins on train and apply same bins to valid
    X_train["_age_bin"], bins = pd.qcut(X_train["Age"], q=n_bins, retbins=True, labels=False, duplicates="drop")
    X_valid["_age_bin"] = pd.cut(X_valid["Age"], bins=bins, labels=False, include_lowest=True)

    for col in columns:
        # compute median in each age bin on training data but ignoring zeros
        medians = {}
        for b in sorted(X_train["_age_bin"].dropna().unique()):
            mask = X_train["_age_bin"] == b
            col_vals = X_train.loc[mask, col].replace(0, np.nan)
            med = col_vals.median()
            if pd.isna(med):
                med = X_train[col].replace(0, np.nan).median()
            medians[b] = med
        # fallback median
        fallback = X_train[col].replace(0, np.nan).median()
        # replace zeros per bin for train and valid
        for df_, name in [(X_train, "train"), (X_valid, "valid")]:
            for idx in df_.index:
                b = df_.at[idx, "_age_bin"]
                if pd.isna(b):
                    df_.at[idx, col] = df_.at[idx, col] if df_.at[idx, col] != 0 else fallback
                else:
                    med = medians.get(int(b), fallback)
                    if df_.at[idx, col] == 0:
                        df_.at[idx, col] = med
    # drop helper column
    X_train = X_train.drop(columns=["_age_bin"])
    X_valid = X_valid.drop(columns=["_age_bin"])
    return X_train, X_valid

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "cm": cm}

def main():
    df = load_data()
    # basic sanity
    if "Outcome" not in df.columns:
        raise ValueError("Expectation: 'Outcome' column in CSV")
    # add initial engineered features (these use raw values; imputation must happen first normally)
    # We'll perform imputation first and then add features for consistency
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Impute zeros with median per age-bin
    X_train_imp, X_test_imp = age_bins_median_impute(X_train, X_test, MISSING_ZERO_COLS, n_bins=5)
    print("Done age-bin median imputation for:", MISSING_ZERO_COLS)

    # Add engineered features after imputation
    X_train_feat = add_features(X_train_imp)
    X_test_feat = add_features(X_test_imp)

    # Prepare scaler and scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)

    # Apply SMOTE to training set
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    print("After SMOTE, class distribution:", np.bincount(y_res))

    # LightGBM classifier
    lgb_clf = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # Hyperparameter distributions for randomized search
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "num_leaves": [15, 31, 41, 63],
        "max_depth": [-1, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0, 0.01, 0.1, 1]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rnd = RandomizedSearchCV(
        lgb_clf,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    print("\nRunning RandomizedSearchCV for LightGBM...")
    t0 = time()
    rnd.fit(X_res, y_res)
    print(f"RandomizedSearchCV done in {time()-t0:.1f}s")
    print("Best LightGBM params:", rnd.best_params_)

    best_lgb = rnd.best_estimator_

    # Evaluate on test data
    print("\n=== Evaluating best LightGBM on test set ===")
    y_pred = best_lgb.predict(X_test_scaled)
    results = evaluate(y_test, y_pred)

    # Save model and scaler
    model_path = os.path.join(MODELS_DIR, "best_model_lightgbm.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(best_lgb, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved LightGBM model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")

if __name__ == "__main__":
    main()
