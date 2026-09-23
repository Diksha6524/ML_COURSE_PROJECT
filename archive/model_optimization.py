import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

TRAIN_FILE = os.path.join(BASE_PATH, "train_features.csv")
VAL_FILE = os.path.join(BASE_PATH, "val_features.csv")

RESULTS_DIR = os.path.join(BASE_PATH, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL OPTIMIZATION")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

X_train = train_df.drop(columns=["image_name", "label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["image_name", "label"])
y_val = val_df["label"]

print(f"\nTraining samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Original features  : {X_train.shape[1]}")


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("SELECTING FEATURES USING MUTUAL INFORMATION")
print("=" * 70)

# Calculate mutual information ONLY on training data
mi_scores = mutual_info_classif(
    X_train,
    y_train,
    random_state=42
)

mi_df = pd.DataFrame({
    "Feature": X_train.columns,
    "MI_Score": mi_scores
})

mi_df = mi_df.sort_values(
    by="MI_Score",
    ascending=False
)

print("\nTop 25 features:")
print(mi_df.head(25).to_string(index=False))


# ============================================================
# DIFFERENT NUMBERS OF FEATURES
# ============================================================

feature_counts = [10, 15, 20, 25, 30, 35, 42]

results = []

best_score = 0
best_setting = None


# ============================================================
# TEST EACH FEATURE COUNT
# ============================================================

for k in feature_counts:

    print("\n" + "=" * 70)
    print(f"TESTING TOP {k} FEATURES")
    print("=" * 70)

    selected_features = mi_df.head(k)["Feature"].tolist()

    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print(f"\nTraining Random Forest with {k} features...")

    rf_pipeline = Pipeline([
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
        "model__max_depth": [None, 15, 25],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
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
        X_train_selected,
        y_train
    )

    rf_pred = rf_grid.predict(
        X_val_selected
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

    print(f"Random Forest Accuracy : {rf_accuracy:.4f}")
    print(f"Random Forest F1       : {rf_f1:.4f}")

    print("Best RF parameters:")
    print(rf_grid.best_params_)




    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    print(f"\nTraining SVM with {k} features...")

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            SVC(
                class_weight="balanced"
            )
        )
    ])

    svm_params = {
        "model__C": [0.1, 1, 10],
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
        X_train_selected,
        y_train
    )

    svm_pred = svm_grid.predict(
        X_val_selected
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

    print(f"SVM Accuracy : {svm_accuracy:.4f}")
    print(f"SVM F1       : {svm_f1:.4f}")

    print("Best SVM parameters:")
    print(svm_grid.best_params_)


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append({
        "Features": k,
        "RF_Accuracy": rf_accuracy,
        "RF_Precision": rf_precision,
        "RF_Recall": rf_recall,
        "RF_F1": rf_f1,
        "SVM_Accuracy": svm_accuracy,
        "SVM_Precision": svm_precision,
        "SVM_Recall": svm_recall,
        "SVM_F1": svm_f1
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["RF_F1", "SVM_F1"],
    ascending=False
)

print("\n\n" + "=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# BEST RESULT
# ============================================================

best_row = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST FEATURE CONFIGURATION")
print("=" * 70)

print(f"\nNumber of features: {int(best_row['Features'])}")

print(
    f"Random Forest Accuracy: "
    f"{best_row['RF_Accuracy']:.4f}"
)

print(
    f"Random Forest F1: "
    f"{best_row['RF_F1']:.4f}"
)

print(
    f"SVM Accuracy: "
    f"{best_row['SVM_Accuracy']:.4f}"
)

print(
    f"SVM F1: "
    f"{best_row['SVM_F1']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "feature_selection_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\nResults saved to:")
print(results_file)

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETED")
print("=" * 70)