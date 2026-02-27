"""
Training module for SVM classifier
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.bt_svm.config import Config
from src.bt_svm.data import DataLoader
from src.bt_svm.preprocess import Preprocessor


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def train_model(config: Config) -> dict:
    """
    Full training pipeline
    
    Returns:
        dict with metrics and model info
    """
    logger.info("=" * 60)
    logger.info("BRAIN TUMOR SVM TRAINING - PHASE 1")
    logger.info("=" * 60)
    
    # Validate config
    config.validate()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load data
    loader = DataLoader(config)
    X_train_raw, y_train = loader.load_split("train")
    X_test_raw, y_test = loader.load_split("test")
    
    logger.info(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")
    
    # Preprocess
    preprocessor = Preprocessor(config)
    X_train = preprocessor.preprocess(X_train_raw)
    X_test = preprocessor.preprocess(X_test_raw)
    
    logger.info(f"Preprocessed: Train {X_train.shape}, Test {X_test.shape}")
    
    # Apply PCA
    X_train_pca = preprocessor.fit_pca(X_train)
    X_test_pca = preprocessor.transform_pca(X_test)
    
    logger.info(f"After PCA: Train {X_train_pca.shape}, Test {X_test_pca.shape}")
    
    # Train model
    logger.info(f"Training {config.model_type.upper()} model...")
    
    if config.model_type == "svm":
        model = SVC(kernel=config.svm_kernel, C=config.svm_C, random_state=config.random_state, verbose=1)
    else:
        model = LogisticRegression(max_iter=1000, random_state=config.random_state, verbose=1)
    
    model.fit(X_train_pca, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train_pca)
    y_pred_test = model.predict(X_test_pca)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_type": config.model_type,
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "train_f1_weighted": float(f1_score(y_train, y_pred_train, average='weighted')),
        "test_f1_weighted": float(f1_score(y_test, y_pred_test, average='weighted')),
        "test_precision_weighted": float(precision_score(y_test, y_pred_test, average='weighted')),
        "test_recall_weighted": float(recall_score(y_test, y_pred_test, average='weighted')),
        "n_classes": len(config.classes),
        "classes": config.classes,
        "pca_features": X_train_pca.shape[1],
        "pca_variance": config.pca_variance,
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Train Accuracy:  {metrics['train_accuracy']:.4f}")
    logger.info(f"Test Accuracy:   {metrics['test_accuracy']:.4f}")
    logger.info(f"Test F1 (weighted): {metrics['test_f1_weighted']:.4f}")
    logger.info(f"Test Precision:  {metrics['test_precision_weighted']:.4f}")
    logger.info(f"Test Recall:     {metrics['test_recall_weighted']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred_test, target_names=config.classes)}")
    
    # Save artifacts
    logger.info("\nSaving artifacts...")
    
    model_path = os.path.join(config.output_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")
    
    pca_path = os.path.join(config.output_dir, "pca.joblib")
    joblib.dump(preprocessor.pca, pca_path)
    logger.info(f"PCA saved: {pca_path}")
    
    metrics_path = os.path.join(config.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")
    
    config_path = os.path.join(config.output_dir, "config.json")
    config.to_json(config_path)
    logger.info(f"Config saved: {config_path}")
    
    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train SVM classifier for brain tumors")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/models/dev", help="Output directory")
    parser.add_argument("--img-size", type=int, default=200, help="Image size")
    parser.add_argument("--pca-variance", type=float, default=0.98, help="PCA variance threshold")
    parser.add_argument("--model-type", type=str, default="svm", choices=["svm", "logistic_regression"])
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    config = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        pca_variance=args.pca_variance,
        model_type=args.model_type,
        verbose=args.verbose
    )
    
    try:
        metrics = train_model(config)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
