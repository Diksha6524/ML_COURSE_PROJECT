import os
import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ============================================================
# PATHS
# ============================================================

BASE_PATH = r"C:\Users\vires\Desktop\ML_COURSE_PROJECT"

DATASET_PATH = os.path.join(BASE_PATH, "dataset")

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")

TRAIN_OUTPUT = os.path.join(
    BASE_PATH,
    "improved_train_features.csv"
)

VAL_OUTPUT = os.path.join(
    BASE_PATH,
    "improved_val_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

# LBP settings
LBP_POINTS = 8
LBP_RADIUS = 1

# Histogram bins
HIST_BINS = 8


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    "class0_normal": "Normal",
    "class1_acne": "Acne",
    "class2_wrinkles": "Wrinkles",
    "class3_Eczema": "Eczema",
    "class4_Rosacea": "Rosacea",
    "class5_dark_spots": "Dark Spots"
}


# ============================================================
# BASIC STATISTICAL FEATURES
# ============================================================

def channel_statistics(channel, prefix):

    features = {}

    features[f"mean_{prefix}"] = np.mean(channel)
    features[f"std_{prefix}"] = np.std(channel)
    features[f"min_{prefix}"] = np.min(channel)
    features[f"max_{prefix}"] = np.max(channel)

    return features


# ============================================================
# COLOR HISTOGRAM
# ============================================================

def color_histogram(channel, prefix):

    features = {}

    hist = cv2.calcHist(
        [channel],
        [0],
        None,
        [HIST_BINS],
        [0, 256]
    )

    hist = hist.flatten()

    # Normalize histogram
    hist = hist / (hist.sum() + 1e-8)

    for i, value in enumerate(hist):

        features[f"{prefix}_hist_{i}"] = value

    return features


# ============================================================
# GLCM FEATURES
# ============================================================

def extract_glcm_features(gray):

    features = {}

    # Quantize image from 256 gray levels → 32 levels
    gray_quantized = (gray / 8).astype(np.uint8)

    distances = [1, 2]

    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    glcm = graycomatrix(
        gray_quantized,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for prop in properties:

        values = graycoprops(
            glcm,
            prop
        )

        # Mean across distances and directions
        features[f"glcm_{prop}"] = np.mean(values)

        # Standard deviation
        features[f"glcm_{prop}_std"] = np.std(values)

    return features


# ============================================================
# LBP FEATURES
# ============================================================

def extract_lbp_features(gray):

    features = {}

    lbp = local_binary_pattern(
        gray,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )

    # Uniform LBP gives P+2 bins
    n_bins = LBP_POINTS + 2

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_bins + 1),
        range=(0, n_bins)
    )

    hist = hist.astype(float)

    # Normalize
    hist = hist / (hist.sum() + 1e-8)

    for i, value in enumerate(hist):

        features[f"lbp_{i}"] = value

    # Additional LBP statistics
    features["lbp_mean"] = np.mean(lbp)
    features["lbp_std"] = np.std(lbp)

    return features


# ============================================================
# EDGE FEATURES
# ============================================================

def extract_edge_features(gray):

    features = {}

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_pixels = np.sum(edges > 0)

    total_pixels = edges.shape[0] * edges.shape[1]

    features["edge_density"] = (
        edge_pixels / total_pixels
    )

    # Sobel gradient
    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        sobel_x ** 2 +
        sobel_y ** 2
    )

    features["gradient_mean"] = np.mean(
        magnitude
    )

    features["gradient_std"] = np.std(
        magnitude
    )

    return features


# ============================================================
# COLOR RATIO FEATURES
# ============================================================

def extract_color_ratio_features(image):

    features = {}

    b = image[:, :, 0].astype(float)
    g = image[:, :, 1].astype(float)
    r = image[:, :, 2].astype(float)

    # Redness-related ratio
    red_ratio = r / (
        r + g + b + 1e-8
    )

    features["red_ratio_mean"] = np.mean(
        red_ratio
    )

    features["red_ratio_std"] = np.std(
        red_ratio
    )

    # Green-red difference
    rg_difference = r - g

    features["red_green_difference_mean"] = np.mean(
        rg_difference
    )

    features["red_green_difference_std"] = np.std(
        rg_difference
    )

    # Redness proportion
    redness_mask = (
        (r > g) &
        (r > b)
    )

    features["redness_percentage"] = np.mean(
        redness_mask
    )

    return features


# ============================================================
# EXTRACT FEATURES FROM ONE IMAGE
# ============================================================

def extract_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize
    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    features = {}

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    b, g, r = cv2.split(image)

    features.update(
        channel_statistics(r, "R")
    )

    features.update(
        channel_statistics(g, "G")
    )

    features.update(
        channel_statistics(b, "B")
    )

    # Histograms
    features.update(
        color_histogram(r, "R")
    )

    features.update(
        color_histogram(g, "G")
    )

    features.update(
        color_histogram(b, "B")
    )


    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    features.update(
        channel_statistics(h, "H")
    )

    features.update(
        channel_statistics(s, "S")
    )

    features.update(
        channel_statistics(v, "V")
    )


    # --------------------------------------------------------
    # LAB
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    features.update(
        channel_statistics(
            l_channel,
            "L"
        )
    )

    features.update(
        channel_statistics(
            a_channel,
            "A"
        )
    )

    features.update(
        channel_statistics(
            b_channel,
            "LAB_B"
        )
    )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    features.update(
        channel_statistics(
            gray,
            "gray"
        )
    )


    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    features["brightness"] = np.mean(
        gray
    )

    features["brightness_std"] = np.std(
        gray
    )


    # --------------------------------------------------------
    # GLCM
    # --------------------------------------------------------

    features.update(
        extract_glcm_features(gray)
    )


    # --------------------------------------------------------
    # LBP
    # --------------------------------------------------------

    features.update(
        extract_lbp_features(gray)
    )


    # --------------------------------------------------------
    # EDGE FEATURES
    # --------------------------------------------------------

    features.update(
        extract_edge_features(gray)
    )


    # --------------------------------------------------------
    # COLOR RATIO FEATURES
    # --------------------------------------------------------

    features.update(
        extract_color_ratio_features(image)
    )


    return features


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(dataset_path):

    all_data = []

    print("\n" + "=" * 70)
    print(
        f"PROCESSING: {dataset_path}"
    )
    print("=" * 70)

    for folder_name, label in CLASS_NAMES.items():

        class_path = os.path.join(
            dataset_path,
            folder_name
        )

        if not os.path.exists(class_path):

            print(
                f"\nWARNING: {class_path} not found"
            )

            continue

        image_files = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ]

        print(
            f"\n{label}: {len(image_files)} images"
        )

        successful = 0

        for index, filename in enumerate(
            image_files,
            start=1
        ):

            image_path = os.path.join(
                class_path,
                filename
            )

            features = extract_features(
                image_path
            )

            if features is not None:

                features["image_name"] = filename
                features["label"] = label

                all_data.append(
                    features
                )

                successful += 1

            if index % 100 == 0:

                print(
                    f"  Processed "
                    f"{index}/{len(image_files)}"
                )

        print(
            f"  Successfully processed: "
            f"{successful}"
        )

    return pd.DataFrame(all_data)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_df = process_dataset(
        TRAIN_PATH
    )

    print("\n" + "=" * 70)
    print("TRAINING FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Training shape: {train_df.shape}"
    )

    print(
        f"Number of features: "
        f"{train_df.shape[1] - 2}"
    )

    print("\nClass distribution:")

    print(
        train_df["label"].value_counts()
    )

    train_df.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    print(
        f"\nSaved training features to:"
    )

    print(TRAIN_OUTPUT)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_df = process_dataset(
        VAL_PATH
    )

    print("\n" + "=" * 70)
    print("VALIDATION FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Validation shape: {val_df.shape}"
    )

    print(
        f"Number of features: "
        f"{val_df.shape[1] - 2}"
    )

    print("\nClass distribution:")

    print(
        val_df["label"].value_counts()
    )

    val_df.to_csv(
        VAL_OUTPUT,
        index=False
    )

    print(
        f"\nSaved validation features to:"
    )

    print(VAL_OUTPUT)


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL IMPROVED FEATURE EXTRACTION COMPLETED")
    print("=" * 70)