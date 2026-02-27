"""
Unit tests for SVM package
Phase 6: Quality assurance
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

from src.bt_svm.config import Config
from src.bt_svm.data import DataLoader
from src.bt_svm.preprocess import Preprocessor


class TestConfig:
    """Test configuration"""
    
    def test_config_creation(self):
        """Test config object creation"""
        config = Config(data_dir="data")
        assert config.data_dir == "data"
        assert len(config.classes) == 4
    
    def test_config_dict(self):
        """Test config to dict conversion"""
        config = Config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "classes" in config_dict


class TestPreprocessor:
    """Test preprocessing pipeline"""
    
    def test_preprocessor_creation(self):
        """Test preprocessor initialization"""
        config = Config()
        preprocessor = Preprocessor(config)
        assert preprocessor is not None
    
    def test_preprocess_shape(self):
        """Test preprocessing output shape"""
        config = Config()
        preprocessor = Preprocessor(config)
        
        # Create dummy image
        dummy_images = np.random.randint(0, 255, (10, 200, 200), dtype=np.uint8)
        processed = preprocessor.preprocess(dummy_images)
        
        assert processed.shape == (10, 40000)
        assert processed.dtype == np.float32
    
    def test_pca_dimensionality_reduction(self):
        """Test PCA reduces dimensions correctly"""
        config = Config(pca_variance=0.95)
        preprocessor = Preprocessor(config)
        
        # Create dummy training data
        X_train = np.random.randn(100, 40000)
        X_pca = preprocessor.fit_pca(X_train)
        
        # PCA should reduce dimensions
        assert X_pca.shape[1] < 40000
        assert X_pca.shape[0] == 100


class TestDataLoader:
    """Test data loading"""
    
    def test_dataloader_creation(self):
        """Test dataloader initialization"""
        config = Config(data_dir="data")
        loader = DataLoader(config)
        assert loader is not None
        assert loader.class_mapping is not None
    
    def test_class_mapping(self):
        """Test class mapping"""
        config = Config()
        loader = DataLoader(config)
        
        expected_classes = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
        assert list(loader.class_mapping.keys()) == expected_classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
