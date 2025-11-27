# src/predict.py
import os
import joblib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")

# Candidate model filenames in order of preference.
CANDIDATE_MODELS = [
    "best_model_lightgbm.pkl",
    "best_model_smote_impute.pkl",
    "best_model_improved.pkl",
    "diabetes_model.pkl"
]

SCALER_NAME = "scaler.pkl"

def find_model_and_scaler():
    """Return (model_path, scaler_path) for the first candidate that exists, else (None, None)."""
    for name in CANDIDATE_MODELS:
        mpath = os.path.join(MODELS_DIR, name)
        if os.path.exists(mpath):
            scaler_path = os.path.join(MODELS_DIR, SCALER_NAME)
            if os.path.exists(scaler_path):
                return mpath, scaler_path
            else:
                # model exists but scaler missing -> return model path and None
                return mpath, None
    return None, None

def load_model():
    """Load first available model and scaler. Raise FileNotFoundError with helpful message if missing."""
    model_path, scaler_path = find_model_and_scaler()
    if model_path is None:
        raise FileNotFoundError(
            "No model found. Searched for: {} in '{}'.\n"
            "Upload one of these model files to the 'models/' folder or change the app to use an available model.".format(
                ", ".join(CANDIDATE_MODELS), MODELS_DIR
            )
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
    return model, scaler, os.path.basename(model_path)

def predict(sample):
    """
    sample: list or array of shape (8,) in the original order:
    [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    returns (label, probability, model_name)
    """
    model, scaler, model_name = load_model()
    X = np.array(sample).reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)
    # If model has predict_proba:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0, 1]
    else:
        # fallback: use decision_function when available, or numeric prediction
        try:
            score = model.decision_function(X)
            prob = float(1 / (1 + np.exp(-score)))  # approximate
        except Exception:
            prob = float(model.predict(X)[0])
    label = int(model.predict(X)[0])
    return label, float(prob), model_name

if __name__ == "__main__":
    # Example usage shown only when running this file directly, not on import
    example = [2, 120, 70, 20, 79, 25.6, 0.5, 28]
    try:
        label, probability, model_name = predict(example)
        print(f"Using model: {model_name}")
        print(f"Predicted label: {label} (1=diabetes, 0=no diabetes), probability={probability:.4f}")
    except FileNotFoundError as e:
        print("Model file not found. Details:", e)
