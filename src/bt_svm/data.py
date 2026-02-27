"""
Data loading and validation module for brain tumor classification
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate brain tumor MRI images"""
    
    def __init__(self, config):
        self.config = config
        self.class_mapping = {cls_name: idx for idx, cls_name in enumerate(config.classes)}
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}
        self.manifest = []
    
    def load_split(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from train or test split
        
        Args:
            split: "train" or "test"
            
        Returns:
            X: array of shape (N, img_size, img_size, 1)
            y: array of shape (N,) with class indices
        """
        data_path = self.config.get_data_path(split)
        X, y, filepaths = [], [], []
        
        logger.info(f"Loading {split} data from {data_path}")
        
        for class_name in self.config.classes:
            class_path = data_path / class_name
            
            if not class_path.exists():
                logger.warning(f"Missing class folder: {class_path}")
                continue
            
            class_idx = self.class_mapping[class_name]
            img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            logger.info(f"  {class_name}: {len(img_files)} images")
            
            for img_file in img_files:
                img_path = class_path / img_file
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        logger.warning(f"Could not read image: {img_path}")
                        continue
                    
                    # Resize to target size
                    img = cv2.resize(img, (self.config.img_size, self.config.img_size))
                    
                    X.append(img)
                    y.append(class_idx)
                    filepaths.append(str(img_path))
                    
                except Exception as e:
                    logger.error(f"Error loading {img_path}: {e}")
                    continue
        
        X = np.array(X, dtype=np.uint8)
        y = np.array(y, dtype=np.int32)
        
        logger.info(f"Loaded: X={X.shape}, y={y.shape}, classes count={np.bincount(y)}")
        
        # Store manifest
        self.manifest = [
            {
                "filepath": fp,
                "label": self.reverse_mapping[label],
                "label_idx": int(label),
                "width": self.config.img_size,
                "height": self.config.img_size,
                "split": split
            }
            for fp, label in zip(filepaths, y)
        ]
        
        return X, y
    
    def save_manifest(self, output_path: str):
        """Save dataset manifest to JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        logger.info(f"Manifest saved to {output_path}")
    
    def class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution"""
        unique, counts = np.unique(y, return_counts=True)
        return {self.reverse_mapping[idx]: int(count) for idx, count in zip(unique, counts)}
