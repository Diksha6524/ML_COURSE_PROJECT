# ============================================================
# VISUALIZATION OF ML MODEL RESULTS
# ML Course Project - Facial Skin Condition Classification
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

TRAIN_FILE = os.path.join(
    PROJECT_DIR,
    "improved_train_features.csv"
)

VAL_FILE = os.path.join(
    PROJECT_DIR,
    "improved_val_features.csv"
)


# ============================================================
# 2. CREATE RESULTS FOLDER
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 3. MODEL COMPARISON DATA
# ============================================================

# These are the results obtained during your experiments.

model_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Random Forest",
        "SVM",
        "Optimized SVM"
    ],

    "Accuracy": [
        45.50,
        48.50,
        49.68,
        62.96,
        63.81,
        72.27
    ],

    "F1_Score": [
        45.50,
        48.50,
        49.68,
        62.63,
        64.11,
        72.05
    ]
})


# ============================================================
# 4. DISPLAY MODEL RESULTS
# ============================================================

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print("\n")
print(
    model_results.to_string(
        index=False
    )
)


# ============================================================
# 5. ACCURACY COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    model_results["Model"],
    model_results["Accuracy"]
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Accuracy Comparison of Machine Learning Models"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(
    0,
    100
)

# Display accuracy value above each bar

for i, value in enumerate(
    model_results["Accuracy"]
):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()


accuracy_file = os.path.join(
    RESULTS_DIR,
    "model_accuracy_comparison.png"
)

plt.savefig(
    accuracy_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 6. F1-SCORE COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    model_results["Model"],
    model_results["F1_Score"]
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Weighted F1-Score (%)"
)

plt.title(
    "Weighted F1-Score Comparison of Machine Learning Models"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(
    0,
    100
)

# Display F1 value above each bar

for i, value in enumerate(
    model_results["F1_Score"]
):

    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )


plt.tight_layout()


f1_file = os.path.join(
    RESULTS_DIR,
    "model_f1_comparison.png"
)

plt.savefig(
    f1_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 7. SAVE MODEL COMPARISON TABLE
# ============================================================

comparison_file = os.path.join(
    RESULTS_DIR,
    "model_comparison_final.csv"
)

model_results.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

# Your optimized SVM confusion matrix

class_names = [
    "Acne",
    "Dark Spots",
    "Eczema",
    "Normal",
    "Rosacea",
    "Wrinkles"
]


cm = [
    [101, 10, 45, 9, 18, 2],
    [10, 31, 2, 6, 4, 8],
    [20, 4, 202, 2, 9, 3],
    [12, 10, 18, 180, 2, 18],
    [12, 1, 11, 0, 84, 0],
    [3, 3, 5, 12, 0, 77]
]


# Convert to DataFrame

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)


# ============================================================
# 9. PLOT CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm_df.values,
    display_labels=class_names
)

display.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "Confusion Matrix - Optimized SVM"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()


confusion_file = os.path.join(
    RESULTS_DIR,
    "optimized_svm_confusion_matrix.png"
)

plt.savefig(
    confusion_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. SAVE CONFUSION MATRIX
# ============================================================

cm_file = os.path.join(
    RESULTS_DIR,
    "optimized_svm_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_file
)


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print("\nFiles saved inside:")
print(RESULTS_DIR)

print("\nGenerated files:")

print(
    "\n1. model_accuracy_comparison.png"
)

print(
    "2. model_f1_comparison.png"
)

print(
    "3. optimized_svm_confusion_matrix.png"
)

print(
    "4. model_comparison_final.csv"
)

print(
    "5. optimized_svm_confusion_matrix.csv"
)

# print("\nDone! 🎉")