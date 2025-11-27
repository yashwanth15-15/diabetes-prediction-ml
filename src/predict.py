# src/predict.py
import os
import joblib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "models")

def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return model, scaler

def predict(sample):
    model, scaler = load_model()
    sample = np.array(sample).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    pred = model.predict(sample_scaled)
    prob = model.predict_proba(sample_scaled)[0,1]
    return int(pred[0]), float(prob)

if __name__ == "__main__":
    # Example sample in same column order as dataset:
    # [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    example = [2, 120, 70, 20, 79, 25.6, 0.5, 28]
    label, probability = predict(example)
    print(f"Predicted label: {label} (1=diabetes, 0=no diabetes), probability_of_diabetes={probability:.4f}")
