"""
Inference module for predictions
"""

import logging
from pathlib import Path
import cv2
import numpy as np
import joblib

logger = logging.getLogger(__name__)


class Predictor:
    """Make predictions on new images"""
    
    def __init__(self, model_path: str, pca_path: str, config):
        self.model = joblib.load(model_path)
        self.pca = joblib.load(pca_path)
        self.config = config
        self.class_names = {idx: name for idx, name in enumerate(config.classes)}
    
    def predict_image(self, image_path: str) -> dict:
        """
        Predict on single image
        
        Returns:
            dict with prediction and confidence
        """
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize
        img = cv2.resize(img, (self.config.img_size, self.config.img_size))
        
        # Preprocess
        img_flat = img.reshape(1, -1).astype(np.float32) / 255.0
        
        # Apply PCA
        img_pca = self.pca.transform(img_flat)
        
        # Predict
        pred_idx = self.model.predict(img_pca)[0]
        pred_class = self.class_names[pred_idx]
        
        # Get decision function for confidence
        try:
            decision = self.model.decision_function(img_pca)
            confidence = float(np.max(np.abs(decision)))
        except:
            confidence = None
        
        return {
            "image_path": str(image_path),
            "predicted_class": pred_class,
            "class_index": int(pred_idx),
            "confidence": confidence
        }
