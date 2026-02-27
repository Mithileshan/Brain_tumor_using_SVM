"""
Evaluation script - generates metrics, confusion matrix, and reports
Phase 2-3: Generate detailed evaluation results
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bt_svm.config import Config
from src.bt_svm.data import DataLoader
from src.bt_svm.preprocess import Preprocessor
from src.bt_svm.eval import ModelEvaluator
import joblib


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = True):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def evaluate(config: Config, model_path: str, pca_path: str):
    """
    Full evaluation pipeline
    """
    logger.info("=" * 70)
    logger.info("BRAIN TUMOR SVM EVALUATION - PHASE 2-3")
    logger.info("=" * 70)
    
    # Load data
    loader = DataLoader(config)
    X_test_raw, y_test = loader.load_split("test")
    
    # Preprocess
    preprocessor = Preprocessor(config)
    X_test = preprocessor.preprocess(X_test_raw)
    X_test_pca = preprocessor.transform_pca(X_test)
    
    logger.info(f"Eval data: {X_test_pca.shape}")
    
    # Evaluate
    evaluator = ModelEvaluator(model_path, pca_path, config)
    metrics, y_pred = evaluator.evaluate(X_test_pca, y_test)
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 70)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (weighted):    {metrics['recall_weighted']:.4f}")
    logger.info(f"F1 (weighted):        {metrics['f1_weighted']:.4f}")
    
    # Generate visualizations
    os.makedirs("reports", exist_ok=True)
    
    # Confusion matrix
    cm_path = "reports/confusion_matrix.png"
    evaluator.plot_confusion_matrix(y_test, y_pred, cm_path)
    
    # Classification report
    report_path = "reports/classification_report.txt"
    report_text = evaluator.plot_classification_report(y_test, y_pred, report_path)
    logger.info(f"\n{report_text}")
    
    # Save metrics
    metrics['timestamp'] = datetime.now().isoformat()
    metrics_path = "reports/metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")
    
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SVM model")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model", type=str, default="artifacts/models/dev/model.joblib", help="Model path")
    parser.add_argument("--pca", type=str, default="artifacts/models/dev/pca.joblib", help="PCA path")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    config = Config(data_dir=args.data_dir, verbose=args.verbose)
    
    try:
        metrics = evaluate(config, args.model, args.pca)
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
