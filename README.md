# 💳 Credit Risk Modeling & Prediction System

An end-to-end Machine Learning web application designed to assess creditworthiness and predict the risk profile (Good vs. Bad credit risk) of loan applicants using an optimized **XGBoost Classifier** deployed via **Streamlit**.

---

## 📌 Project Overview

Financial institutions face substantial challenges in assessing whether an applicant is likely to default on a credit facility. This project leverages historical financial and demographic data from the German Credit dataset to build an automated risk classification model. 

The pipeline handles:
- Categorical data encoding using pre-fitted scikit-learn encoders.
- Feature preprocessing and handling of skewed financial attributes.
- Inference using an XGBoost gradient boosting pipeline.
- Interactive user inference via an interactive Streamlit dashboard.

---

## 📂 Repository Structure

```text
├── german_credit_data.csv          # Raw German Credit Risk dataset
├── analysis_model.ipynb            # EDA, data preprocessing, and model training notebook
├── app.py                          # Streamlit web application script
├── xgb_credit_model.pkl            # Trained XGBoost classification model
├── Checking account_encoder.pkl    # Preprocessor for checking account status
├── Housing_encoder.pkl             # Preprocessor for housing categories
├── Saving accounts_encoder.pkl     # Preprocessor for savings account status
├── Sex_encoder.pkl                 # Preprocessor for gender encoding
├── target_encoder.pkl              # Target variable decoder (Risk: Good/Bad)
└── README.md                       # Documentation
