import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

TRAIN_FILE = os.path.join(
    BASE_PATH,
    "improved_train_features.csv"
)

VAL_FILE = os.path.join(
    BASE_PATH,
    "improved_val_features.csv"
)

RESULTS_DIR = os.path.join(
    BASE_PATH,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("IMPROVED MODEL TRAINING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

X_train = train_df.drop(
    columns=["image_name", "label"]
)

y_train = train_df["label"]

X_val = val_df.drop(
    columns=["image_name", "label"]
)

y_val = val_df["label"]

print("\nTraining shape   :", X_train.shape)
print("Validation shape :", X_val.shape)
print("Number of features:", X_train.shape[1])


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# TEST DIFFERENT FEATURE COUNTS
# ============================================================

feature_counts = [
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    98
]

results = []


# ============================================================
# LOOP THROUGH FEATURE COUNTS
# ============================================================

for k in feature_counts:

    print("\n" + "=" * 70)
    print(f"TESTING TOP {k} FEATURES")
    print("=" * 70)


    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    print("\nTraining SVM...")

    svm_pipeline = Pipeline([
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
            "model",
            SVC(
                class_weight="balanced"
            )
        )
    ])


    svm_params = {
        "model__C": [1, 10, 30],
        "model__gamma": ["scale", 0.01, 0.1],
        "model__kernel": ["rbf"]
    }


    svm_grid = GridSearchCV(
        svm_pipeline,
        svm_params,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0
    )


    svm_grid.fit(
        X_train,
        y_train
    )


    svm_pred = svm_grid.predict(
        X_val
    )


    svm_accuracy = accuracy_score(
        y_val,
        svm_pred
    )

    svm_precision = precision_score(
        y_val,
        svm_pred,
        average="weighted",
        zero_division=0
    )

    svm_recall = recall_score(
        y_val,
        svm_pred,
        average="weighted",
        zero_division=0
    )

    svm_f1 = f1_score(
        y_val,
        svm_pred,
        average="weighted",
        zero_division=0
    )


    print(
        f"SVM Accuracy : {svm_accuracy:.4f}"
    )

    print(
        f"SVM F1       : {svm_f1:.4f}"
    )

    print(
        "Best SVM parameters:"
    )

    print(
        svm_grid.best_params_
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    rf_pipeline = Pipeline([
        (
            "feature_selection",
            SelectKBest(
                score_func=mutual_info_classif,
                k=k
            )
        ),

        (
            "model",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])


    rf_params = {

        "model__n_estimators": [200],

        "model__max_depth": [
            None,
            15,
            25
        ],

        "model__min_samples_split": [
            2,
            5
        ],

        "model__min_samples_leaf": [
            1,
            2
        ]
    }


    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_params,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0
    )


    rf_grid.fit(
        X_train,
        y_train
    )


    rf_pred = rf_grid.predict(
        X_val
    )


    rf_accuracy = accuracy_score(
        y_val,
        rf_pred
    )

    rf_precision = precision_score(
        y_val,
        rf_pred,
        average="weighted",
        zero_division=0
    )

    rf_recall = recall_score(
        y_val,
        rf_pred,
        average="weighted",
        zero_division=0
    )

    rf_f1 = f1_score(
        y_val,
        rf_pred,
        average="weighted",
        zero_division=0
    )


    print(
        f"RF Accuracy : {rf_accuracy:.4f}"
    )

    print(
        f"RF F1       : {rf_f1:.4f}"
    )

    print(
        "Best RF parameters:"
    )

    print(
        rf_grid.best_params_
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({

        "Features": k,

        "SVM_Accuracy":
            svm_accuracy,

        "SVM_Precision":
            svm_precision,

        "SVM_Recall":
            svm_recall,

        "SVM_F1":
            svm_f1,

        "RF_Accuracy":
            rf_accuracy,

        "RF_Precision":
            rf_precision,

        "RF_Recall":
            rf_recall,

        "RF_F1":
            rf_f1
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n" + "=" * 70)
print("FINAL IMPROVED MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# BEST SVM
# ============================================================

best_svm = results_df.loc[
    results_df["SVM_F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST IMPROVED SVM")
print("=" * 70)

print(
    f"Features : {int(best_svm['Features'])}"
)

print(
    f"Accuracy : {best_svm['SVM_Accuracy']:.4f}"
)

print(
    f"Precision: {best_svm['SVM_Precision']:.4f}"
)

print(
    f"Recall   : {best_svm['SVM_Recall']:.4f}"
)

print(
    f"F1 Score : {best_svm['SVM_F1']:.4f}"
)


# ============================================================
# BEST RANDOM FOREST
# ============================================================

best_rf = results_df.loc[
    results_df["RF_F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST IMPROVED RANDOM FOREST")
print("=" * 70)

print(
    f"Features : {int(best_rf['Features'])}"
)

print(
    f"Accuracy : {best_rf['RF_Accuracy']:.4f}"
)

print(
    f"Precision: {best_rf['RF_Precision']:.4f}"
)

print(
    f"Recall   : {best_rf['RF_Recall']:.4f}"
)

print(
    f"F1 Score : {best_rf['RF_F1']:.4f}"
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    RESULTS_DIR,
    "improved_model_results.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)