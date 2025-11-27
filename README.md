\# 🩺 Diabetes Prediction Using Machine Learning  

\*\*LightGBM + SMOTE + Age-Bin Imputation + Feature Engineering + SHAP Explainability\*\*



This project predicts diabetes using the PIMA Indians Diabetes Dataset and a full modern ML pipeline that includes:

\- Smart preprocessing (zero → median per age-bin)

\- Feature engineering

\- Standard scaling

\- SMOTE oversampling

\- LightGBM with RandomizedSearchCV tuning

\- Model explainability using SHAP

\- Streamlit web app



---



\## 🚀 Final Model Performance (LightGBM)



| Metric      | Score   |

|-------------|---------|

| Accuracy    | 76.62%  |

| F1-score    | 0.6842  |

| Precision   | 0.6500  |

| Recall      | 0.7222  |



\### Confusion Matrix

\[\[79 21]

\[15 39]]

\## 📂 Project Structure

diabetes-prediction-ml

│── data/ # dataset

│── models/

│ ├── best\_model\_lightgbm.pkl

│ ├── scaler.pkl

│ ├── shap\_summary.png

│ ├── shap\_bar.png

│ └── shap\_dependence\_GIR.png

│── src/

│ ├── train\_lightgbm.py # final model training

│ ├── train\_improved.py

│ ├── predict.py

│ ├── shap\_explain.py

│ └── app.py # Streamlit app

│── notebook/

│ └── summary.ipynb # (content provided below)

│── requirements.txt

│── README.md



\## ▶️ Training the Final Model

```bash

python src/train\_lightgbm.py

▶️ Generating SHAP Explainability Visuals

python src/shap\_explain.py

▶️ Running the Streamlit App

streamlit run src/app.py

🌐 Live Demo (Optional)



Deploy using Streamlit Cloud or HuggingFace Spaces.

📊 SHAP Plots



Feature Importance (Summary Plot)





Mean Absolute SHAP Values





Dependence Plot: Glucose\_Insulin\_Ratio





✨ Author



Yashwanth Bankapalli









