# src/train_smote_impute.py
import os
import numpy as np
import pandas as pd
import joblib
from time import time
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
RANDOM_STATE = 42

# Columns that use 0 as missing in PIMA dataset
MISSING_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def replace_zeros_with_median(X_train, X_valid, columns):
    """Compute medians on X_train and replace zeros in train and valid."""
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    for col in columns:
        med = X_train[col].replace(0, np.nan).median()
        # if median is nan (rare), fallback to overall median
        if np.isnan(med):
            med = X_train[col].median()
        X_train[col] = X_train[col].replace(0, med)
        X_valid[col] = X_valid[col].replace(0, med)
    return X_train, X_valid

def evaluate_and_print(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy:  {:.4f}".format(acc))
    print("F1-score:  {:.4f}".format(f1))
    print("Precision: {:.4f}".format(prec))
    print("Recall:    {:.4f}".format(rec))
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "cm": cm}

def main():
    df = load_data()
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # single stratified split (we will do CV inside RandomizedSearchCV)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Replace zeros with median (compute medians only on train)
    X_train, X_test = replace_zeros_with_median(X_train, X_test, MISSING_ZERO_COLS)
    print("Replaced zeros with median for columns:", MISSING_ZERO_COLS)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to training data (after scaling)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    print("After SMOTE, class distribution:", np.bincount(y_res))

    # Candidate classifiers and param distributions
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight=None)  # we used SMOTE so no class_weight
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)

    param_dist = [
        {
            "estimator": [rf],
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [4, 6, 8, None],
            "estimator__min_samples_split": [2, 4, 6],
            "estimator__min_samples_leaf": [1, 2, 4],
            "estimator__max_features": ["sqrt", "log2", None]
        },
        {
            "estimator": [xgb],
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [3, 4, 6],
            "estimator__learning_rate": [0.01, 0.05, 0.1],
            "estimator__subsample": [0.6, 0.8, 1.0],
            "estimator__colsample_bytree": [0.6, 0.8, 1.0]
        }
    ]

    from sklearn.ensemble import VotingClassifier
    # We'll use wrapper that allows RandomizedSearchCV to search over two different estimator types.
    # Create a simple meta-estimator class that holds an 'estimator' parameter (scikit-learn >=0.24 supports pipelined param names)
    # Simpler approach: perform separate randomized searches for RF and XGB then compare. We'll do two searches.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ----- Randomized Search for RandomForest -----
    rf_param = {
        "n_estimators": [100,200,300],
        "max_depth": [4,6,8,None],
        "min_samples_split": [2,4,6],
        "min_samples_leaf": [1,2,4],
        "max_features": ["sqrt","log2", None]
    }
    rnd_rf = RandomizedSearchCV(
        rf,
        param_distributions=rf_param,
        n_iter=20,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    print("\nRunning RandomizedSearchCV for RandomForest...")
    t0 = time()
    rnd_rf.fit(X_res, y_res)
    print("RF RandomizedSearch done in {:.1f}s".format(time()-t0))
    print("Best RF params:", rnd_rf.best_params_)
    best_rf = rnd_rf.best_estimator_

    # Evaluate RF on test set
    print("\n--- Evaluating best RandomForest on test set ---")
    res_rf = evaluate_and_print(best_rf, X_test_scaled, y_test)

    # ----- Randomized Search for XGBoost -----
    xgb_param = {
        "n_estimators": [100,200,300],
        "max_depth": [3,4,6],
        "learning_rate": [0.01,0.05,0.1],
        "subsample": [0.6,0.8,1.0],
        "colsample_bytree": [0.6,0.8,1.0]
    }
    rnd_xgb = RandomizedSearchCV(
        xgb,
        param_distributions=xgb_param,
        n_iter=20,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    print("\nRunning RandomizedSearchCV for XGBoost...")
    t0 = time()
    rnd_xgb.fit(X_res, y_res)
    print("XGB RandomizedSearch done in {:.1f}s".format(time()-t0))
    print("Best XGB params:", rnd_xgb.best_params_)
    best_xgb = rnd_xgb.best_estimator_

    print("\n--- Evaluating best XGBoost on test set ---")
    res_xgb = evaluate_and_print(best_xgb, X_test_scaled, y_test)

    # Choose best by F1
    best_choice = ("random_forest", best_rf, res_rf["f1"]) if res_rf["f1"] >= res_xgb["f1"] else ("xgboost", best_xgb, res_xgb["f1"])
    print("\n=== Best on test set ===")
    print("Choice:", best_choice[0], "F1:", best_choice[2])

    # Save chosen model + scaler + meta info
    chosen_model = best_choice[1]
    model_path = os.path.join(MODELS_DIR, "best_model_smote_impute.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(chosen_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved best model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")

if __name__ == "__main__":
    main()
