import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


BASE_PATH = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

TRAIN_FILE = os.path.join(BASE_PATH, "train_features.csv")
VAL_FILE = os.path.join(BASE_PATH, "val_features.csv")

print("LOADING FEATURE DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print(f"\nTraining data shape   : {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")




#label seprations
X_train = train_df.drop(columns=["image_name", "label"])
y_train = train_df["label"]

X_val = val_df.drop(columns=["image_name", "label"])
y_val = val_df["label"]


print(f"\nNumber of input features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

#models

models = {

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])
}
# print("\nMODELS TO BE TRAINED")

#training and evaluation




results = []

best_model_name = None
best_model = None
best_f1 = 0


for model_name, model in models.items():

    print(f"TRAINING: {model_name}")
    

    # Train
    model.fit(X_train, y_train)

    # Predict validation data
    y_pred = model.predict(X_val)

    # Metrics
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
       
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    #model slection with high f1
    if f1 > best_f1:

        best_f1 = f1
        best_model_name = model_name
        best_model = model

    results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


#best model

print("BEST MODEL")

print(f"Best Model: {best_model_name}")

print(f"Weighted F1 Score: {best_f1:.4f}")





RESULTS_FILE = os.path.join(
    BASE_PATH,
    "model_comparison_results.csv"
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print(f"\nResults saved to:")
print(RESULTS_FILE)