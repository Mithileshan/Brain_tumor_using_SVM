"""
Streamlit UI for Brain Tumor SVM Classification
Phase 5: Interactive demo for predictions
"""

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from src.bt_svm.config import Config
from src.bt_svm.predict import Predictor


st.set_page_config(page_title="Brain Tumor SVM Classifier", layout="wide")

st.title("🧠 Brain Tumor Classification (SVM)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_path = st.text_input("Model path", value="artifacts/models/dev/model.joblib")
    pca_path = st.text_input("PCA path", value="artifacts/models/dev/pca.joblib")
    confidence_threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader("Choose an MRI scan...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

with col2:
    st.subheader("🔍 Prediction Results")
    
    if uploaded_file:
        # Save temp file for prediction (cross-platform)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "streamlit_mri_temp.jpg")
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load model
            config = Config()
            predictor = Predictor(model_path, pca_path, config)
            
            # Predict
            result = predictor.predict_image(temp_path)
            
            # Display results
            st.success("✅ Prediction Complete")
            st.write("\n")
            
            col_pred, col_label = st.columns(2)
            with col_pred:
                st.metric("Predicted Class", result['predicted_class'])
            with col_label:
                st.metric("Class Index", result['class_index'])
            
            if result['confidence'] is not None:
                st.metric("Confidence Score", f"{result['confidence']:.4f}")
            
            # Detailed info
            st.subheader("📋 Detailed Information")
            st.json(result)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        st.info("👆 Upload an MRI image to make a prediction")

# Footer
st.markdown("---")
st.markdown("""
### 📚 Model Information
- **Algorithm:** Support Vector Machine (SVM) with RBF kernel
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **Input:** 200×200 grayscale MRI images
- **Feature Extraction:** PCA (98% variance)
- **Framework:** scikit-learn
""")
