"""
Evaluation module for model assessment
"""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import joblib

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self, model_path: str, pca_path: str = None):
        self.model = joblib.load(model_path)
        self.pca = joblib.load(pca_path) if pca_path else None
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 class_names: list = None) -> Dict:
        """
        Evaluate model on test set
        
        Returns:
            dict with detailed metrics
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        }
        
        return metrics
