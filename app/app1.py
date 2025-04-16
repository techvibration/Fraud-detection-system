from flask import Flask, render_template_string
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

app = Flask(__name__)

# Load model and validation data

model = joblib.load("model1.pkl")
X_val = pd.read_csv("X_val.csv")
y_val = pd.read_csv("y_val.csv")["Class"]

# Predict
y_prob = model.predict_proba(X_val)[:, 1]
y_pred = (y_prob >= 0.4).astype(int)  # Custom threshold if desired

# Metrics
report = classification_report(y_val, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

# Plot confusion matrix
def plot_confusion_matrix():
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Plot ROC
def plot_roc_curve():
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def dashboard():
    cm_img = plot_confusion_matrix()
    roc_img = plot_roc_curve()
    html = f'''
    <html>
    <head><title>Credit Card Fraud Detection - Dashboard</title></head>
    <body style="font-family: Arial; text-align: center; padding: 20px;">
        <h1>Model Performance Dashboard</h1>

        <h2>Confusion Matrix</h2>
        <img src="data:image/png;base64,{cm_img}" />

        <h2>ROC Curve</h2>
        <img src="data:image/png;base64,{roc_img}" />

        <h2>Classification Report</h2>
        <pre>{classification_report(y_val, y_pred)}</pre>

        <h2>ROC AUC Score: {roc_auc:.4f}</h2>
    </body>
    </html>
    '''
    return render_template_string(html)

if __name__ == '__main__':
    app.run(debug=True)
