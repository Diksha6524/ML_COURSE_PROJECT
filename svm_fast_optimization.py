# ============================================================
# FAST SVM OPTIMIZATION
# ML Course Project - Facial Skin Condition Classification
# ============================================================

import os
import time
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
print("FAST SVM OPTIMIZATION")
print("=" * 70)

print("\nLoading data...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print("Training shape   :", train_df.shape)
print("Validation shape :", val_df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND LABEL
# ============================================================

X_train = train_df.drop(
    columns=["image_name", "label"]
)

y_train = train_df["label"]

X_val = val_df.drop(
    columns=["image_name", "label"]
)

y_val = val_df["label"]

print("\nNumber of features:", X_train.shape[1])
print("Training samples :", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# 4. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. PARAMETERS TO TEST
# ============================================================

# We already know that 90 features gave the best result.
# Therefore we focus around that region.

feature_values = [
    70,
    80,
    90,
    98
]

C_values = [
    1,
    10,
    50,
    100
]

gamma_values = [
    "scale",
    0.01,
    0.1
]


# ============================================================
# 6. STORE RESULTS
# ============================================================

results = []

best_cv_f1 = 0
best_configuration = None


# ============================================================
# 7. RUN FAST OPTIMIZATION
# ============================================================

total_tests = (
    len(feature_values)
    * len(C_values)
    * len(gamma_values)
)

current_test = 0

print("\nTotal configurations to test:", total_tests)
print("\nStarting optimization...\n")


for k in feature_values:

    for C in C_values:

        for gamma in gamma_values:

            current_test += 1

            print(
                f"[{current_test}/{total_tests}] "
                f"Features={k}, C={C}, Gamma={gamma}"
            )

            start_time = time.time()

            # ------------------------------------------------
            # Create pipeline
            # ------------------------------------------------

            model = Pipeline([

                (
                    "feature_selection",
                    SelectKBest(
                        score_func=mutual_info_classif,
                        k=k
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=C,
                        gamma=gamma,
                        class_weight=None
                    )
                )
            ])

            # ------------------------------------------------
            # Cross-validation
            # ------------------------------------------------

            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=-1
            )

            mean_cv_f1 = cv_scores.mean()

            elapsed = time.time() - start_time

            print(
                f"    CV F1 = {mean_cv_f1:.4f} "
                f"({mean_cv_f1 * 100:.2f}%) "
                f"| Time = {elapsed:.1f}s"
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append({

                "Features": k,

                "C": C,

                "Gamma": gamma,

                "CV_F1_Mean": mean_cv_f1,

                "CV_F1_Std": cv_scores.std()

            })

            # ------------------------------------------------
            # Track best configuration
            # ------------------------------------------------

            if mean_cv_f1 > best_cv_f1:

                best_cv_f1 = mean_cv_f1

                best_configuration = {

                    "Features": k,

                    "C": C,

                    "Gamma": gamma

                }


# ============================================================
# 8. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV_F1_Mean",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP SVM CONFIGURATIONS")
print("=" * 70)

print(
    results_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 9. BEST CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    "\nBest number of features:",
    best_configuration["Features"]
)

print(
    "Best C:",
    best_configuration["C"]
)

print(
    "Best Gamma:",
    best_configuration["Gamma"]
)

print(
    "Best CV Weighted F1:",
    f"{best_cv_f1:.4f} "
    f"({best_cv_f1 * 100:.2f}%)"
)


# ============================================================
# 10. TRAIN BEST MODEL
# ============================================================

print("\nTraining best SVM on complete training data...")

best_model = Pipeline([

    (
        "feature_selection",
        SelectKBest(
            score_func=mutual_info_classif,
            k=best_configuration["Features"]
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            kernel="rbf",
            C=best_configuration["C"],
            gamma=best_configuration["Gamma"],
            class_weight=None
        )
    )
])

best_model.fit(
    X_train,
    y_train
)

# Save the trained optimized SVM
MODEL_FILE = os.path.join(
    RESULTS_DIR,
    "optimized_svm_model.joblib"
)

joblib.dump(
    best_model,
    MODEL_FILE
)

print("\nOptimized SVM model saved to:")
print(MODEL_FILE)


# ============================================================
# 11. VALIDATION PREDICTION
# ============================================================

print("\nPredicting validation data...")

y_pred = best_model.predict(X_val)


# ============================================================
# 12. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_val,
    y_pred
)

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

weighted_f1 = f1_score(
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
# 13. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULTS")
print("=" * 70)

print(
    f"\nAccuracy        : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision       : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall          : "
    f"{recall * 100:.2f}%"
)

print(
    f"Weighted F1     : "
    f"{weighted_f1 * 100:.2f}%"
)

print(
    f"Macro F1        : "
    f"{macro_f1 * 100:.2f}%"
)


# ============================================================
# 14. COMPARE WITH CURRENT BEST
# ============================================================

CURRENT_BEST = 0.717345

print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT BEST")
print("=" * 70)

print(
    f"\nPrevious best accuracy : "
    f"{CURRENT_BEST * 100:.2f}%"
)

print(
    f"New accuracy           : "
    f"{accuracy * 100:.2f}%"
)

difference = accuracy - CURRENT_BEST

if difference > 0:

    print(
        f"\n🎉 IMPROVEMENT: "
        f"+{difference * 100:.2f} percentage points"
    )

elif difference < 0:

    print(
        f"\nNew model is lower by "
        f"{abs(difference) * 100:.2f} percentage points"
    )

else:

    print("\nAccuracy is the same as the previous best.")


# ============================================================
# 15. CLASSIFICATION REPORT
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
# 16. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

class_names = sorted(
    y_val.unique()
)

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
# 17. SAVE ALL OPTIMIZATION RESULTS
# ============================================================

optimization_file = os.path.join(
    RESULTS_DIR,
    "fast_svm_optimization_results.csv"
)

results_df.to_csv(
    optimization_file,
    index=False
)


# ============================================================
# 18. SAVE BEST MODEL RESULTS
# ============================================================

final_results = pd.DataFrame({

    "Model": [
        "Fast Optimized SVM"
    ],

    "Features": [
        best_configuration["Features"]
    ],

    "C": [
        best_configuration["C"]
    ],

    "Gamma": [
        best_configuration["Gamma"]
    ],

    "CV_Weighted_F1": [
        best_cv_f1
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
        weighted_f1
    ],

    "Validation_Macro_F1": [
        macro_f1
    ]

})

final_file = os.path.join(
    RESULTS_DIR,
    "fast_svm_final_results.csv"
)

final_results.to_csv(
    final_file,
    index=False
)


# ============================================================
# 19. SAVE CLASSIFICATION REPORT
# ============================================================

report_file = os.path.join(
    RESULTS_DIR,
    "fast_svm_classification_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "FAST OPTIMIZED SVM RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Best Configuration:\n"
    )

    f.write(
        str(best_configuration)
    )

    f.write("\n\n")

    f.write(
        f"CV Weighted F1: "
        f"{best_cv_f1:.4f}\n"
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
        f"{weighted_f1:.4f}\n"
    )

    f.write(
        f"Validation Macro F1: "
        f"{macro_f1:.4f}\n\n"
    )

    f.write(
        "Classification Report:\n"
    )

    f.write(
        report
    )

    f.write(
        "\n\nConfusion Matrix:\n"
    )

    f.write(
        str(cm_df)
    )


# ============================================================
# 20. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETED!")
print("=" * 70)

print("\nFiles saved:")

print(
    "\n1.",
    optimization_file
)

print(
    "2.",
    final_file
)

print(
    "3.",
    report_file
)

print("\nDone! 🎉")