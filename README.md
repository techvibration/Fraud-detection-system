# 💳 Credit Card Fraud Detection System

A machine learning–powered system for detecting fraudulent credit card transactions using XGBoost and class imbalance handling. Built with a modular, production‑ready workflow and served with a lightweight Flask API.

---

##  Features

- **High‑accuracy fraud detection** with XGBoost
- **Class imbalance addressed** using `scale_pos_weight` tuning
- **Robust evaluation**: ROC‑AUC, confusion matrix, classification report
- **Validation pipeline** with separate `X_val.csv` and `y_val.csv`
- **Flask API** to visualize model performance
- **Preprocessing via ColumnTransformer** (optional)
- Easy deployment with `joblib`-saved models

---

##  Model Summary

- **Algorithm**: XGBoost Classifier
- **Class Imbalance**: Handled using `scale_pos_weight`  
- **Cross-Validation**: 5×10 Repeated Stratified K-Folds  
- **Metric**: ROC‑AUC Score

 **Validation Results**:
- **ROC-AUC Score**: *0.9995+*
- **Confusion Matrix**: Near-perfect classification
- **Cross-Validation AUC**: 0.99998



 
