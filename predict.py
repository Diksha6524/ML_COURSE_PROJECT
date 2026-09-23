# ============================================================
# SKIN CONDITION PREDICTION
# ML Course Project
# ============================================================

import os
import joblib
import pandas as pd

import improved_feature_extraction as feature_extractor


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

MODEL_FILE = os.path.join(
    PROJECT_DIR,
    "results",
    "optimized_svm_model.joblib"
)

TRAIN_FEATURE_FILE = os.path.join(
    PROJECT_DIR,
    "improved_train_features.csv"
)


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_FILE):

    print("\nERROR: Model file not found!")
    print("\nExpected:")
    print(MODEL_FILE)

    exit()


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("=" * 65)
print("SKIN CONDITION PREDICTION")
print("=" * 65)

print("\nLoading optimized SVM model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully!")


# ============================================================
# 4. LOAD FEATURE NAMES
# ============================================================

# This guarantees that the new image's features are supplied
# in the same order as the training data.

train_df = pd.read_csv(
    TRAIN_FEATURE_FILE
)

feature_columns = [
    column
    for column in train_df.columns
    if column not in ["image_name", "label"]
]

print(
    "\nExpected number of features:",
    len(feature_columns)
)


# ============================================================
# 5. GET IMAGE PATH
# ============================================================

print("\n" + "-" * 65)

image_path = input(
    "\nEnter the complete path of your image: "
).strip()

# Remove quotes if the path was pasted with quotes
image_path = image_path.strip('"').strip("'")


# ============================================================
# 6. CHECK IMAGE
# ============================================================

if not os.path.exists(image_path):

    print("\nERROR: Image not found!")

    print("\nPath entered:")
    print(image_path)

    exit()


# ============================================================
# 7. EXTRACT FEATURES
# ============================================================

print("\nProcessing image...")
print("Extracting handcrafted features...")

features = feature_extractor.extract_features(
    image_path
)


if features is None:

    print("\nERROR: Could not read the image.")

    exit()


# ============================================================
# 8. CONVERT FEATURES TO DATAFRAME
# ============================================================

new_image_df = pd.DataFrame(
    [features]
)


# ============================================================
# 9. CHECK FEATURE COUNT
# ============================================================

print(
    "Extracted features:",
    new_image_df.shape[1]
)

if new_image_df.shape[1] != len(feature_columns):

    print(
        "\nERROR: Feature count does not match training data!"
    )

    print(
        "Expected:",
        len(feature_columns)
    )

    print(
        "Got:",
        new_image_df.shape[1]
    )

    exit()


# ============================================================
# 10. REORDER FEATURES
# ============================================================

new_image_df = new_image_df[
    feature_columns
]


# ============================================================
# 11. PREDICT
# ============================================================

print("\nRunning optimized SVM...")

prediction = model.predict(
    new_image_df
)

predicted_class = prediction[0]


# ============================================================
# 12. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 65)
print("PREDICTION RESULT")
print("=" * 65)

print(
    f"\nPredicted Skin Condition: {predicted_class}"
)


# ============================================================
# 13. DECISION SCORES
# ============================================================

decision_scores = model.decision_function(
    new_image_df
)

print("\nSVM Decision Scores:")

classes = model.classes_

if decision_scores.ndim == 1:

    print(
        f"{predicted_class}: "
        f"{decision_scores[0]:.4f}"
    )

else:

    scores = decision_scores[0]

    for class_name, score in zip(
        classes,
        scores
    ):

        print(
            f"{class_name:15s}: {score:.4f}"
        )


# ============================================================
# 14. FINAL NOTE
# ============================================================

print("\n" + "=" * 65)

print(
    "Note: This is an ML classification result, "
    "not a medical diagnosis."
)

print("=" * 65)