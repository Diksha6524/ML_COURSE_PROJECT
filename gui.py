import os
import cv2
import joblib
import numpy as np
import pandas as pd
import gradio as gr

import improved_feature_extraction as feature_extractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "results", "optimized_svm_model.joblib")
TRAIN_FEATURE_FILE = os.path.join(BASE_DIR, "improved_train_features.csv")

CLASS_DESCRIPTIONS = {
    "Normal": "Healthy skin with balanced tone, smooth texture, and uniform pigmentation.",
    "Acne": "Inflammatory condition involving sebaceous glands and hair follicles (papules, pustules, comedones).",
    "Wrinkles": "Surface creases and fine lines associated with aging or loss of skin elasticity.",
    "Eczema": "Atopic dermatitis causing dry, itchy, inflamed, and reddened skin patches.",
    "Rosacea": "Chronic facial redness, visible capillaries, and small inflammatory bumps.",
    "Dark Spots": "Hyperpigmentation patches from localized excess melanin, sun spots, or post-inflammatory marks."
}

# Load model and columns
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = None

if os.path.exists(TRAIN_FEATURE_FILE):
    train_df = pd.read_csv(TRAIN_FEATURE_FILE, nrows=1)
    feature_cols = [c for c in train_df.columns if c not in ["image_name", "label"]]
else:
    feature_cols = []

def classify_skin_image(image):
    if image is None:
        return "Please upload an image.", {}
    
    if model is None or not feature_cols:
        return "Error: Model or feature dataset missing.", {}
    
    try:
        # Convert RGB image (from Gradio) to BGR for OpenCV feature extraction
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract 98 features
        raw_features = feature_extractor.extract_features(bgr_image)
        if raw_features is None:
            return "Error extracting features from image.", {}
        
        feat_dict = {col: raw_features.get(col, 0.0) for col in feature_cols}
        feat_df = pd.DataFrame([feat_dict])
        
        # Run model inference
        prediction = model.predict(feat_df)[0]
        classes = model.classes_
        
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feat_df)[0]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(feat_df)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / np.sum(exp_scores)
        else:
            probabilities = np.ones(len(classes)) / len(classes)
            
        confidences = {str(classes[i]): float(probabilities[i]) for i in range(len(classes))}
        top_prob = confidences.get(prediction, 0.0) * 100
        
        summary = f"### Top Prediction: {prediction}\n"
        summary += f"**Confidence Level**: {top_prob:.1f}%\n\n"
        summary += f"**Clinical Overview**: {CLASS_DESCRIPTIONS.get(prediction, '')}\n\n"
        summary += "---\n⚠️ **Medical Disclaimer**: Created for academic coursework & technical feature analysis only."
        
        return summary, confidences
    except Exception as e:
        return f"Processing Error: {str(e)}", {}

demo = gr.Interface(
    fn=classify_skin_image,
    inputs=gr.Image(type="numpy", label="Upload Skin Photo"),
    outputs=[
        gr.Markdown(label="AI Diagnosis Summary"),
        gr.Label(num_top_classes=6, label="Class Probabilities")
    ],
    title="🩺 Skin Condition Classification System",
    description="Upload a skin image to predict conditions across 6 classes (Acne, Dark Spots, Eczema, Normal, Rosacea, Wrinkles) using 98 extracted features & an Optimized RBF SVM model.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
