import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

SEED = 42
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

data = load_breast_cancer(as_frame=True)
X = data.data.copy()

# sklearn's target is 0=malignant and 1=benign.
# We keep this mapping explicit so the meaning is clear.
y = data.target.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=SEED))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=SEED))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        class_weight="balanced"
    )
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    }
    results.append(row)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(pd.Series(row))
    print("\nClassification Report:")
    print(classification_report(
        y_test, pred,
        target_names=["malignant", "benign"],
        digits=4
    ))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

# Save comparison table
results_df.to_csv("results/model_comparison.csv", index=False)

# Select the model with highest test ROC-AUC for this demonstration.
# For a real clinical application, model selection would require a separate
# validation strategy and external clinical validation.
best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]

best_pred = best_model.predict(X_test)
best_prob = best_model.predict_proba(X_test)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["malignant", "benign"]
)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_name}")
fig.tight_layout()
fig.savefig("results/confusion_matrix.png", dpi=180)
plt.close(fig)

# ROC curve
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_test, best_prob, ax=ax)
ax.set_title(f"ROC Curve - {best_name}")
fig.tight_layout()
fig.savefig("results/roc_curve.png", dpi=180)
plt.close(fig)

# Model comparison chart
fig, ax = plt.subplots(figsize=(10, 6))
results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].plot(
    kind="bar", ax=ax
)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("results/model_comparison.png", dpi=180)
plt.close(fig)

# Save selected model and feature metadata.
joblib.dump(best_model, "models/breast_cancer_model.joblib")
metadata = {
    "model": best_name,
    "feature_names": list(X.columns),
    "target_mapping": {"0": "malignant", "1": "benign"}
}
with open("models/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSelected model: {best_name}")
print("Saved model to models/breast_cancer_model.joblib")
print("Saved plots and comparison table to results/")
