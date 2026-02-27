"""
Brain Tumor SVM Classification Package
Production-grade multi-class classification (glioma, meningioma, pituitary, no_tumor)
"""

__version__ = "1.0.0"
__author__ = "Brain Tumor Team"

from .config import Config
from .data import DataLoader
from .preprocess import Preprocessor

__all__ = ["Config", "DataLoader", "Preprocessor"]
