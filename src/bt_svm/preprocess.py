"""
Preprocessing module for image data
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """Image preprocessing pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = None
        self.fitted = False
    
    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images: flatten + normalize
        
        Args:
            images: array of shape (N, H, W) or (N, H, W, 1)
            
        Returns:
            processed: array of shape (N, H*W)
        """
        # Ensure 3D
        if images.ndim == 4:
            images = images[:, :, :, 0]
        
        # Flatten
        N = images.shape[0]
        flattened = images.reshape(N, -1)
        
        # Normalize pixels
        if self.config.normalize_pixels:
            flattened = flattened.astype(np.float32) / 255.0
        
        logger.debug(f"Flattened shape: {flattened.shape}, dtype: {flattened.dtype}")
        return flattened
    
    def fit_pca(self, X_train: np.ndarray) -> np.ndarray:
        """
        Fit PCA on training data
        
        Args:
            X_train: preprocessed training array
            
        Returns:
            X_transformed: PCA-reduced training data
        """
        self.pca = PCA(n_components=self.config.pca_variance, random_state=self.config.random_state)
        X_pca = self.pca.fit_transform(X_train)
        
        logger.info(f"PCA fitted: {X_train.shape[1]} -> {X_pca.shape[1]} features")
        logger.info(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        self.fitted = True
        return X_pca
    
    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted PCA to data
        
        Args:
            X: preprocessed data
            
        Returns:
            X_transformed: PCA-reduced data
        """
        if not self.fitted:
            raise RuntimeError("PCA not fitted. Call fit_pca() first.")
        return self.pca.transform(X)
