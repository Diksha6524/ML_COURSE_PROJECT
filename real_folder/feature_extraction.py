import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT\dataset"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")

OUTPUT_DIR = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "train_features.csv")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "val_features.csv")


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

CLASS_NAMES = {
    "class0_normal": "Normal",
    "class1_acne": "Acne",
    "class2_wrinkles": "Wrinkles",
    "class3_Eczema": "Eczema",
    "class4_Rosacea": "Rosacea",
    "class5_dark_spots": "Dark Spots"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize
    image = cv2.resize(image, IMAGE_SIZE)

    # Color conversions
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = {}

    # ========================================================
    # 1. RGB FEATURES
    # ========================================================

    rgb_channels = {
        "R": rgb[:, :, 0],
        "G": rgb[:, :, 1],
        "B": rgb[:, :, 2]
    }

    for name, channel in rgb_channels.items():
        features[f"mean_{name}"] = float(np.mean(channel))
        features[f"std_{name}"] = float(np.std(channel))

    # ========================================================
    # 2. HSV FEATURES
    # ========================================================

    hsv_channels = {
        "H": hsv[:, :, 0],
        "S": hsv[:, :, 1],
        "V": hsv[:, :, 2]
    }

    for name, channel in hsv_channels.items():
        features[f"mean_{name}"] = float(np.mean(channel))
        features[f"std_{name}"] = float(np.std(channel))

    # ========================================================
    # 3. RGB HISTOGRAM FEATURES
    # ========================================================

    for name, channel in rgb_channels.items():

        histogram = cv2.calcHist(
            [channel],
            [0],
            None,
            [8],
            [0, 256]
        ).flatten()

        histogram = histogram / (histogram.sum() + 1e-7)

        for i, value in enumerate(histogram):
            features[f"{name}_hist_{i}"] = float(value)

    # ========================================================
    # 4. BASIC GRAYSCALE FEATURES
    # ========================================================

    features["brightness"] = float(np.mean(gray))
    features["gray_std"] = float(np.std(gray))

    # ========================================================
    # 5. GLCM FEATURES
    # ========================================================

    # Reduce grayscale values from 256 to 32 levels
    gray_32 = (gray / 8).astype(np.uint8)

    # Four directions
    distances = [1]
    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    glcm = graycomatrix(
        gray_32,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "correlation",
        "energy",
        "homogeneity"
    ]

    for prop in properties:

        values = graycoprops(glcm, prop)

        # Average across the 4 directions
        features[f"glcm_{prop}"] = float(np.mean(values))

    return features


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(dataset_path, output_file, dataset_name):

    all_data = []

    print("\n" + "=" * 70)
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print("=" * 70)

    for folder_name, class_label in CLASS_NAMES.items():

        folder_path = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(folder_path):
            print(f"WARNING: Folder not found: {folder_path}")
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            )
        ]

        print(f"\n{class_label}: {len(image_files)} images")

        successful = 0

        for index, image_name in enumerate(image_files, start=1):

            image_path = os.path.join(folder_path, image_name)

            features = extract_features(image_path)

            if features is not None:

                features["image_name"] = image_name
                features["label"] = class_label

                all_data.append(features)
                successful += 1

            # Progress message every 100 images
            if index % 100 == 0:
                print(
                    f"  Processed {index}/{len(image_files)}"
                )

        print(f"  Successfully processed: {successful}")

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(all_data)

    # Put identifiers first
    first_columns = ["image_name", "label"]

    remaining_columns = [
        col for col in df.columns
        if col not in first_columns
    ]

    df = df[first_columns + remaining_columns]

    # Save
    df.to_csv(output_file, index=False)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 70)
    print(f"{dataset_name.upper()} COMPLETE")
    print("-" * 70)

    print(f"Total images processed: {len(df)}")
    print(f"Number of features: {len(df.columns) - 2}")
    print(f"CSV shape: {df.shape}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print(f"\nSaved to:")
    print(output_file)

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_df = process_dataset(
        TRAIN_PATH,
        TRAIN_OUTPUT,
        "Training"
    )

    val_df = process_dataset(
        VAL_PATH,
        VAL_OUTPUT,
        "Validation"
    )

    print("\n" + "=" * 70)
    print("ALL FEATURE EXTRACTION FINISHED")
    print("=" * 70)