"""
Configuration module for SVM training & inference
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Config:
    """Training configuration"""
    
    # Data
    data_dir: str = "data"
    train_dir: str = "Training"
    test_dir: str = "Testing"
    img_size: int = 200
    
    # Classes (4-class classification)
    classes: list = None
    
    # Preprocessing
    normalize_pixels: bool = True
    pca_variance: float = 0.98
    random_state: int = 42
    test_split: float = 0.2
    
    # Training
    model_type: str = "svm"  # "svm" or "logistic_regression"
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    
    # Output
    output_dir: str = "artifacts/models/dev"
    verbose: bool = True
    
    def __post_init__(self):
        if self.classes is None:
            self.classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    
    def to_dict(self):
        """Convert config to dictionary"""
        return asdict(self)
    
    def to_json(self, path: str):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
    
    @classmethod
    def from_json(cls, path: str):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def get_data_path(self, split: str = "train") -> Path:
        """Get path to train/test split"""
        split_name = self.train_dir if split == "train" else self.test_dir
        return Path(self.data_dir) / split_name
    
    def validate(self):
        """Validate config"""
        train_path = self.get_data_path("train")
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        test_path = self.get_data_path("test")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        if self.pca_variance <= 0 or self.pca_variance > 1.0:
            raise ValueError("pca_variance must be between 0 and 1")
        
        if self.test_split <= 0 or self.test_split >= 1.0:
            raise ValueError("test_split must be between 0 and 1")
