#!/usr/bin/env python3
"""
SVM Brain Tumor Classification Inference Script

Usage:
    python scripts/predict.py --model models/svm_model.pkl --pca models/pca.pkl --image sample.jpg
    python scripts/predict.py --model models/svm_model.pkl --pca models/pca.pkl --image sample.jpg --verbose
    python scripts/predict.py --help
"""

import argparse
import sys
from pathlib import Path
import pickle

import numpy as np
import cv2


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference using trained SVM brain tumor classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python scripts/predict.py --model models/svm_model.pkl --pca models/pca.pkl --image sample.jpg
  
  # With verbose output
  python scripts/predict.py --model models/svm_model.pkl --pca models/pca.pkl --image sample.jpg --verbose
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained SVM model (svm_model.pkl)'
    )
    
    parser.add_argument(
        '--pca',
        type=str,
        required=True,
        help='Path to PCA transformer (pca.pkl)'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image for inference'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=200,
        help='Image size (default: 200)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    return parser.parse_args()


def load_model_and_pca(model_path, pca_path):
    """Load trained model and PCA transformer."""
    print("Loading model and PCA...")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(pca_path).exists():
        raise FileNotFoundError(f"PCA not found: {pca_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"  ✓ Model: {model_path}")
        
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        print(f"  ✓ PCA: {pca_path}")
        
        return model, pca
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def load_image(image_path, img_size=200):
    """Load and preprocess image."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        img = cv2.resize(img, (img_size, img_size))
        return img
    except Exception as e:
        raise RuntimeError(f"Error loading image: {e}")


def preprocess_image(img, pca):
    """Flatten, normalize, and apply PCA."""
    # Flatten
    img_flat = img.reshape(1, -1)
    
    # Normalize
    img_norm = img_flat / 255.0
    
    # Apply PCA
    img_pca = pca.transform(img_norm)
    
    return img_pca


def predict(model, img_pca, verbose=False):
    """Make prediction."""
    prediction = model.predict(img_pca)[0]
    
    if verbose:
        # Try to get decision scores if available
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(img_pca)[0]
            confidence = 1.0 / (1.0 + np.exp(-scores))  # Sigmoid
            if verbose:
                print(f"  Decision score: {scores:.4f}")
                print(f"  Confidence: {confidence:.4f}")
    
    return prediction


def main():
    """Main inference function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🧠 SVM Brain Tumor Classification - Inference")
    print("="*70 + "\n")
    
    try:
        # Load model and PCA
        print("📦 Loading pipeline...")
        model, pca = load_model_and_pca(args.model, args.pca)
        print()
        
        # Load image
        print(f"📷 Loading image: {args.image}")
        img = load_image(args.image, img_size=args.img_size)
        print(f"  Shape: {img.shape}")
        print()
        
        # Preprocess
        print("⚙️  Preprocessing...")
        img_pca = preprocess_image(img, pca)
        print(f"  Input shape: {img.shape[0] * img.shape[1]}")
        print(f"  After PCA: {img_pca.shape[1]}")
        print()
        
        # Predict
        print("🔮 Making prediction...")
        prediction = predict(model, img_pca, verbose=args.verbose)
        
        # Decode prediction
        class_names = {0: 'No Tumor', 1: 'Tumor Present'}
        predicted_class = class_names.get(prediction, 'Unknown')
        
        print("\n" + "="*70)
        print("✅ Prediction Result")
        print("="*70)
        print(f"\n🎯 Image: {Path(args.image).name}")
        print(f"   Prediction: {predicted_class} (Class {prediction})")
        
        if args.verbose:
            print(f"\n   Model type: {type(model).__name__}")
            print(f"   PCA components: {pca.n_components_}")
            print(f"   Input image size: {args.img_size}×{args.img_size}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
