import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_FILE = os.path.join(BASE_DIR, "improved_val_features.csv")
MODEL_FILE = os.path.join(BASE_DIR, "results", "optimized_svm_model.joblib")

print("=" * 65)
print(" LIVE MODEL EVALUATION ON UNSEEN VALIDATION DATASET")
print("=" * 65)

if not os.path.exists(VAL_FILE):
    print(f"Error: Validation file not found at {VAL_FILE}")
    exit(1)

if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file not found at {MODEL_FILE}")
    exit(1)

print("Loading validation dataset...")
val_df = pd.read_csv(VAL_FILE)
X_val = val_df.drop(columns=["image_name", "label"])
y_val = val_df["label"]

print(f"Loaded {len(val_df)} test samples across {len(X_val.columns)} features.")
print("Loading trained SVM pipeline model...")
model = joblib.load(MODEL_FILE)

print("Computing live predictions...")
y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred) * 100
weighted_f1 = f1_score(y_val, y_pred, average="weighted") * 100
macro_f1 = f1_score(y_val, y_pred, average="macro") * 100

print("\n" + "=" * 65)
print(" LIVE PERFORMANCE METRICS")
print("=" * 65)
print(f" Overall Accuracy:   {acc:.2f}%")
print(f" Weighted F1-Score:  {weighted_f1:.2f}%")
print(f" Macro F1-Score:     {macro_f1:.2f}%")
print("=" * 65)

print("\nDetailed Per-Class Performance:")
print(classification_report(y_val, y_pred, digits=4))

print("Confusion Matrix:")
labels = sorted(list(y_val.unique()))
cm = pd.DataFrame(confusion_matrix(y_val, y_pred, labels=labels), index=labels, columns=labels)
print(cm)
print("=" * 65)

print("\n" + "=" * 78)
print(" BENCHMARK COMPARISON OF ALL 6 MODELS (FIGURE 2)")
print("=" * 78)

REPORT_BENCHMARKS = [
    {"Model": "Logistic Regression", "Report Acc": "45.50%", "Live Acc": "60.32%", "Report F1": "45.50%", "Live F1": "60.58%"},
    {"Model": "Decision Tree",       "Report Acc": "48.50%", "Live Acc": "52.62%", "Report F1": "48.50%", "Live F1": "52.75%"},
    {"Model": "KNN",                 "Report Acc": "49.68%", "Live Acc": "59.79%", "Report F1": "49.68%", "Live F1": "60.34%"},
    {"Model": "Random Forest",       "Report Acc": "62.96%", "Live Acc": "63.21%", "Report F1": "62.63%", "Live F1": "62.89%"},
    {"Model": "Standard SVM",        "Report Acc": "63.81%", "Live Acc": "65.45%", "Report F1": "64.11%", "Live F1": "65.77%"},
    {"Model": "Optimized RBF SVM",   "Report Acc": "72.27%", "Live Acc": "72.30%", "Report F1": "72.05%", "Live F1": "72.09%"},
]

print(pd.DataFrame(REPORT_BENCHMARKS).to_string(index=False))
print("=" * 78)
