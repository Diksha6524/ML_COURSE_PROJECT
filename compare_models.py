import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "results", "model_comparison_final.csv")

print("=" * 48)
print("    MODEL PERFORMANCE COMPARISON (FIGURE 2)")
print("=" * 48)

df = pd.read_csv(CSV_FILE)
df["Accuracy"] = df["Accuracy"].apply(lambda x: f"{float(x):.2f}%")
df["F1_Score"] = df["F1_Score"].apply(lambda x: f"{float(x):.2f}%")
df.columns = ["Model", "Accuracy", "F1-Score"]

print(df.to_string(index=False))
print("=" * 48)
