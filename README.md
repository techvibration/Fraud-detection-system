# AI‑Driven Fraud Detection System

Hey, This is our end-to-end pipeline for detecting fraudulent credit card transaction using machine learning. We chose XGBoost to train our model becuase of the huge class imbalance. You can check out our model's features and performance below. 

---

## 🔧 Installation & Usage

1. Clone and install:
    ```
   -git clone https://github.com/techvibration/fraud-detection-system.git
   -cd fraud-detection
   -pip install -r requirements.txt
    ```
 2. Train the model:
    ```
    python src/fd_model.py
    ```
 3. Launch the dashboard:
    ```
    streamlit run dashboard.py
    ```      
  

##  Features

- **High‑accuracy fraud detection** with XGBoost
- **Class imbalance addressed** using `scale_pos_weight` tuning
- **Robust evaluation**: ROC‑AUC, confusion matrix, classification report
- **Validation pipeline** with separate `X_val1.csv` and `y_val1.csv`
- **Streamlit** to show transaction trends and number of anomalies caught
- Easy deployment with `joblib`-saved models

---

##  Model Summary

- **Algorithm**: XGBoost Classifier
- **Class Imbalance**: Handled using `scale_pos_weight`  
- **Cross-Validation**: 5×10 Repeated Stratified K-Folds  
- **Metric**: ROC‑AUC Score

## 📈 Model Performance

| Metric                        | Value   | Threshold |
|-------------------------------|--------:|:---------:|
| **Mean CV ROC‑AUC**           | 0.9793  | –         |
| **Test ROC‑AUC**              | 0.9182  | –         |
| **Accuracy**                  | 1.00    | 0.40      |
| **Precision (fraud class)**   | 0.85    | 0.40      |
| **Recall (fraud class)**      | 0.84    | 0.40      |
| **F1‑Score (fraud class)**    | 0.84    | 0.40      |

---

🎓 Next Steps

-Experiment with additional feature engineering (time‑based velocity, merchant categories).
-Compare unsupervised anomaly detection methods (Isolation Forest, Autoencoder).
-Integrate a real‑time data stream (Kafka, Pub/Sub) for live scoring.

---

📜 License

This project is released under the MIT License. See LICENSE for details.






 
