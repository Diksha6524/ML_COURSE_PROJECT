import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif


# ============================================================
# PATH
# ============================================================

BASE_PATH = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

PROJECT_DIR = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

TRAIN_FILE = os.path.join(
    PROJECT_DIR,
    "improved_train_features.csv"
)

RESULTS_DIR = os.path.join(BASE_PATH, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE ANALYSIS")
print("=" * 70)

df = pd.read_csv(TRAIN_FILE)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# SEPARATE FEATURES AND LABEL
# ============================================================

X = df.drop(columns=["image_name", "label"])
y = df["label"]

print(f"Number of features: {X.shape[1]}")
print(f"Number of samples : {X.shape[0]}")


# ============================================================
# 1. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features:")
print(importance_df.head(20).to_string(index=False))


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = os.path.join(
    RESULTS_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# PLOT TOP 20 FEATURES
# ============================================================

top20 = importance_df.head(20).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Feature Importances - Random Forest")

plt.tight_layout()

plot_file = os.path.join(
    RESULTS_DIR,
    "top20_feature_importance.png"
)

plt.savefig(plot_file, dpi=300)

plt.show()


# ============================================================
# 2. MUTUAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MUTUAL INFORMATION FEATURE ANALYSIS")
print("=" * 70)

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MI_Score": mi_scores
})

mi_df = mi_df.sort_values(
    by="MI_Score",
    ascending=False
)

print("\nTop 20 Features based on Mutual Information:")

print(
    mi_df.head(20).to_string(index=False)
)


# ============================================================
# SAVE MUTUAL INFORMATION
# ============================================================

mi_file = os.path.join(
    RESULTS_DIR,
    "mutual_information.csv"
)

mi_df.to_csv(
    mi_file,
    index=False
)


# ============================================================
# 3. CORRELATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("HIGHLY CORRELATED FEATURES")
print("=" * 70)

correlation_matrix = X.corr().abs()

high_correlations = []

for i in range(len(correlation_matrix.columns)):

    for j in range(i + 1, len(correlation_matrix.columns)):

        correlation = correlation_matrix.iloc[i, j]

        if correlation >= 0.90:

            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]

            high_correlations.append({
                "Feature_1": feature1,
                "Feature_2": feature2,
                "Correlation": correlation
            })


correlation_df = pd.DataFrame(high_correlations)

if len(correlation_df) > 0:

    correlation_df = correlation_df.sort_values(
        by="Correlation",
        ascending=False
    )

    print(
        correlation_df.to_string(index=False)
    )

else:

    print("No feature pairs with correlation >= 0.90 were found.")


# ============================================================
# SAVE CORRELATION RESULTS
# ============================================================

correlation_file = os.path.join(
    RESULTS_DIR,
    "high_correlations.csv"
)

correlation_df.to_csv(
    correlation_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ANALYSIS COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(importance_file)
print(mi_file)
print(correlation_file)
print(plot_file)