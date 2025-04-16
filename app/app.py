import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, send_file
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

app = Flask(__name__)

# ——— Load model, preprocessor, and validation data ———
model        = joblib.load('model1.pkl')
preprocessor = joblib.load('ct.pkl')

X_val = pd.read_csv('X_val.csv')               # preprocessed features
y_val = pd.read_csv('y_val.csv')['Class']      # or ['Machine failure'] as appropriate

# Pre‑transform if your X_val is raw; otherwise skip this line
# X_t = preprocessor.transform(X_val)
# For simplicity assume X_val is already preprocessed

# Compute once on startup (so endpoints are fast)
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Metrics
roc_auc = roc_auc_score(y_val, y_prob)
cm      = confusion_matrix(y_val, y_pred)
report  = classification_report(y_val, y_pred, output_dict=True)

@app.route('/validate', methods=['GET'])
def validate():
    """Return JSON with ROC‑AUC, confusion matrix, and classification report."""
    return jsonify({
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    })

@app.route('/plot_confusion_matrix', methods=['GET'])
def plot_confusion_matrix():
    """Return a PNG of the normalized confusion matrix."""
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
        xlabel='Predicted', ylabel='Actual',
        title='Confusion Matrix (Normalized)'
    )

    # Annotate with raw counts and percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i,j]}\n{cm_norm[i,j]*100:.1f}%",
                ha='center', va='center',
                color='white' if cm_norm[i,j] > 0.5 else 'black'
            )

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/plot_roc_curve', methods=['GET'])
def plot_roc_curve():
    """Return a PNG of the ROC curve."""
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc_score = roc_auc

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set(
        xlim=[0, 1], ylim=[0, 1],
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title='ROC Curve'
    )
    ax.legend(loc='lower right')

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

