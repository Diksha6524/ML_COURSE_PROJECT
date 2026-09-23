# ============================================================
# SVM OPTIMIZATION
# ML Course Project - Facial Skin Condition Classification
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

TRAIN_FILE = os.path.join(
    PROJECT_DIR,
    "improved_train_features.csv"
)

VAL_FILE = os.path.join(
    PROJECT_DIR,
    "improved_val_features.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("SVM OPTIMIZATION")
print("=" * 70)

print("\nLoading feature datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print("\nTraining data shape:", train_df.shape)
print("Validation data shape:", val_df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND LABEL
# ============================================================

# image_name and label are not ML features
DROP_COLUMNS = ["image_name", "label"]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["label"]

X_val = val_df.drop(columns=DROP_COLUMNS)
y_val = val_df["label"]

print("\nNumber of available features:", X_train.shape[1])
print("Training samples:", X_train.shape[0])
print("Validation samples:", X_val.shape[0])


# ============================================================
# 4. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())


# ============================================================
# 5. CROSS-VALIDATION SETUP
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. CREATE SVM PIPELINE
# ============================================================

# Feature selection is inside the pipeline.
# This prevents information leakage during cross-validation.

pipeline = Pipeline([
    (
        "feature_selection",
        SelectKBest(
            score_func=mutual_info_classif
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC()
    )
])


# ============================================================
# 7. HYPERPARAMETER GRID
# ============================================================

print("\nPreparing hyperparameter search...")

param_grid = {

    # Number of selected features
    "feature_selection__k": [
        70,
        80,
        85,
        90,
        95,
        98
    ],

    # SVM kernel
    "svm__kernel": [
        "rbf",
        "linear",
        "poly"
    ],

    # Regularization parameter
    "svm__C": [
        0.1,
        1,
        10,
        50,
        100
    ],

    # Kernel coefficient
    "svm__gamma": [
        "scale",
        "auto",
        0.001,
        0.01,
        0.1
    ],

    # Class imbalance handling
    "svm__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# 8. GRID SEARCH
# ============================================================

print("\nStarting GridSearchCV...")
print("This may take some time because many combinations are tested.")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)


# ============================================================
# 9. BEST PARAMETERS
# ============================================================

print("\n" + "=" * 70)
print("BEST SVM PARAMETERS")
print("=" * 70)

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "\nBest cross-validation weighted F1:",
    round(grid_search.best_score_, 4)
)


# ============================================================
# 10. TRAIN BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

print("\nTraining final optimized SVM...")
best_model.fit(X_train, y_train)


# ============================================================
# 11. VALIDATION PREDICTION
# ============================================================

print("\nGenerating validation predictions...")

y_pred = best_model.predict(X_val)


# ============================================================
# 12. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_val, y_pred)

precision = precision_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_val,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZED SVM VALIDATION RESULTS")
print("=" * 70)

print(f"\nAccuracy       : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision      : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall         : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Weighted F1    : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"Macro F1       : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_val,
    y_pred,
    zero_division=0
)

print("\n")
print(report)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

class_names = sorted(y_val.unique())

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=class_names
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm_df)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["Optimized SVM"],
    "Features": [
        grid_search.best_params_["feature_selection__k"]
    ],
    "Kernel": [
        grid_search.best_params_["svm__kernel"]
    ],
    "C": [
        grid_search.best_params_["svm__C"]
    ],
    "Gamma": [
        grid_search.best_params_["svm__gamma"]
    ],
    "Class_Weight": [
        grid_search.best_params_["svm__class_weight"]
    ],
    "CV_Weighted_F1": [
        grid_search.best_score_
    ],
    "Validation_Accuracy": [
        accuracy
    ],
    "Validation_Precision": [
        precision
    ],
    "Validation_Recall": [
        recall
    ],
    "Validation_Weighted_F1": [
        f1
    ],
    "Validation_Macro_F1": [
        macro_f1
    ]
})

results_file = os.path.join(
    RESULTS_DIR,
    "svm_optimization_results.csv"
)

results.to_csv(
    results_file,
    index=False
)


# ============================================================
# 17. SAVE CLASSIFICATION REPORT
# ============================================================

report_file = os.path.join(
    RESULTS_DIR,
    "optimized_svm_classification_report.txt"
)

with open(report_file, "w") as f:

    f.write("OPTIMIZED SVM CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("Best Parameters:\n")
    f.write(str(grid_search.best_params_))
    f.write("\n\n")

    f.write(
        f"Cross-Validation Weighted F1: "
        f"{grid_search.best_score_:.4f}\n\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{accuracy:.4f}\n"
    )

    f.write(
        f"Validation Precision: "
        f"{precision:.4f}\n"
    )

    f.write(
        f"Validation Recall: "
        f"{recall:.4f}\n"
    )

    f.write(
        f"Validation Weighted F1: "
        f"{f1:.4f}\n"
    )

    f.write(
        f"Validation Macro F1: "
        f"{macro_f1:.4f}\n\n"
    )

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm_df))


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETED")
print("=" * 70)

print("\nResults saved to:")
print(results_file)

print("\nClassification report saved to:")
print(report_file)

print("\nFinal optimized SVM accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nFinal optimized SVM weighted F1:")
print(f"{f1 * 100:.2f}%")

print("\nDone!")