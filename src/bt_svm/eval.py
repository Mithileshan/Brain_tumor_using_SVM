"""
Enhanced evaluation module with metrics collection and visualization
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations"""
    
    def __init__(self, model_path: str, pca_path: str, config):
        self.model = joblib.load(model_path)
        self.pca = joblib.load(pca_path)
        self.config = config
        self.class_names = config.classes
    
    def evaluate(self, X_test_pca: np.ndarray, y_test: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Comprehensive evaluation
        
        Returns:
            (metrics_dict, predictions)
        """
        y_pred = self.model.predict(X_test_pca)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            "per_class": {}
        }
        
        # Per-class metrics
        for idx, class_name in enumerate(self.class_names):
            class_mask = y_test == idx
            if class_mask.sum() > 0:
                class_y_test = y_test[class_mask]
                class_y_pred = y_pred[class_mask]
                metrics["per_class"][class_name] = {
                    "support": int(class_mask.sum()),
                    "recall": float(recall_score(class_y_test, class_y_pred, labels=[idx], average='macro', zero_division=0))
                }
        
        return metrics, y_pred
    
    def confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_test, y_pred)
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray, 
                              output_path: str):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Brain Tumor Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {output_path}")
        plt.close()
    
    def plot_classification_report(self, y_test: np.ndarray, y_pred: np.ndarray,
                                   output_path: str):
        """Save classification report"""
        report = classification_report(y_test, y_pred, target_names=self.class_names, digits=4)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Classification report saved: {output_path}")
        return report
