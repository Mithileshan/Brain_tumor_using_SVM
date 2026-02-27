#!/usr/bin/env python3
"""
SVM Brain Tumor Classification Training Script

Extracted from: PROJECT Brain Tumor Classification.ipynb

Usage:
    python scripts/train.py --data-dir brain_tumor/ --output-dir models/
    python scripts/train.py --data-dir brain_tumor/ --output-dir models/ --test-split 0.2 --random-state 42
    python scripts/train.py --help
"""

import argparse
import sys
from pathlib import Path
import pickle

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SVM model for brain tumor classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py --data-dir brain_tumor/ --output-dir models/
  python scripts/train.py --data-dir brain_tumor/ --img-size 200 --test-split 0.2
  python scripts/train.py --data-dir brain_tumor/ --pca-variance 0.98 --verbose
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='brain_tumor',
        help='Path to dataset directory (default: brain_tumor/)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save models (default: models/)'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.2,
        help='Test set fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=200,
        help='Image resize size (default: 200)'
    )
    
    parser.add_argument(
        '--pca-variance',
        type=float,
        default=0.98,
        help='PCA variance to retain (default: 0.98)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=10,
        help='Random state for reproducibility (default: 10)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed training information'
    )
    
    return parser.parse_args()


def load_dataset(data_dir, img_size=200):
    """
    Load dataset from directory structure.
    
    Expected structure:
    data_dir/
    ├── Training/
    │   ├── no_tumor/
    │   └── pituitary_tumor/
    └── Testing/
        ├── no_tumor/
        └── pituitary_tumor/
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    
    training_path = data_path / 'Training'
    if not training_path.exists():
        raise FileNotFoundError(f"Training directory not found: {training_path}")
    
    # Class mapping
    classes = {'no_tumor': 0, 'pituitary_tumor': 1}
    
    X = []
    Y = []
    
    print(f"Loading images from {training_path}...")
    for class_name, class_id in classes.items():
        class_path = training_path / class_name
        if not class_path.exists():
            print(f"⚠️  Warning: Class directory not found: {class_path}")
            continue
        
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.bmp'))
        print(f"  {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            try:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"    ⚠️  Failed to read: {img_file}")
                    continue
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                Y.append(class_id)
            except Exception as e:
                print(f"    ⚠️  Error processing {img_file}: {e}")
    
    if len(X) == 0:
        raise ValueError(f"No images found in {training_path}")
    
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"\n✓ Dataset loaded: {len(X)} images")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {np.unique(Y)}")
    print(f"  Class distribution: {np.bincount(Y)}")
    
    return X, Y


def preprocess_data(X_train, X_test, pca_variance=0.98, verbose=False):
    """
    Preprocess images: flatten, normalize, apply PCA.
    """
    print("\n📊 Preprocessing data...")
    
    # Flatten images
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    print(f"  Flattened shape: {X_train_flat.shape} → {X_test_flat.shape}")
    
    # Normalize to [0, 1]
    X_train_norm = X_train_flat / 255.0
    X_test_norm = X_test_flat / 255.0
    print(f"  ✓ Normalized to [0, 1]")
    
    # Apply PCA
    pca = PCA(pca_variance)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    print(f"  ✓ PCA applied: {pca_variance:.0%} variance retained")
    print(f"    Original shape: {X_train_norm.shape[1]}")
    print(f"    Reduced shape: {X_train_pca.shape[1]}")
    print(f"    Components: {pca.n_components_}")
    
    if verbose:
        print(f"    Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_train_pca, X_test_pca, pca


def train_models(X_train, X_test, y_train, y_test, verbose=False):
    """Train SVM and Logistic Regression models."""
    print("\n🤖 Training models...")
    
    # Train SVM
    print("  Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, verbose=(1 if verbose else 0))
    svm.fit(X_train, y_train)
    print("  ✓ SVM trained")
    
    # Train Logistic Regression
    print("  Training Logistic Regression...")
    lr = LogisticRegression(C=0.1, max_iter=1000, verbose=(1 if verbose else 0))
    lr.fit(X_train, y_train)
    print("  ✓ Logistic Regression trained")
    
    return svm, lr


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """Evaluate model performance."""
    print(f"\n📈 Evaluating {model_name}...")
    
    # Train predictions
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print(f"  Training Accuracy:  {train_acc:.4f}")
    print(f"  Testing Accuracy:   {test_acc:.4f}")
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall:             {recall:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")
    
    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['No Tumor', 'Tumor']))
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
    }


def save_models(svm, lr, pca, output_dir='models/'):
    """Save trained models and preprocessor."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving models to {output_path}...")
    
    # Save SVM
    svm_path = output_path / 'svm_model.pkl'
    with open(svm_path, 'wb') as f:
        pickle.dump(svm, f)
    print(f"  ✓ SVM: {svm_path}")
    
    # Save Logistic Regression
    lr_path = output_path / 'lr_model.pkl'
    with open(lr_path, 'wb') as f:
        pickle.dump(lr, f)
    print(f"  ✓ Logistic Regression: {lr_path}")
    
    # Save PCA
    pca_path = output_path / 'pca.pkl'
    with open(pca_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"  ✓ PCA: {pca_path}")
    
    return svm_path, lr_path, pca_path


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("🧠 SVM Brain Tumor Classification - Training Script")
    print("="*70 + "\n")
    
    try:
        # Load dataset
        X, Y = load_dataset(args.data_dir, img_size=args.img_size)
        
        # Split dataset
        print(f"\nSplitting dataset ({args.test_split:.0%} test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y,
            test_size=args.test_split,
            random_state=args.random_state,
            stratify=Y
        )
        print(f"  Train: {len(X_train)} images")
        print(f"  Test: {len(X_test)} images")
        print(f"  Train distribution: {np.bincount(y_train)}")
        print(f"  Test distribution: {np.bincount(y_test)}")
        
        # Preprocess
        X_train_pca, X_test_pca, pca = preprocess_data(
            X_train, X_test,
            pca_variance=args.pca_variance,
            verbose=args.verbose
        )
        
        # Train models
        svm, lr = train_models(X_train_pca, X_test_pca, y_train, y_test, verbose=args.verbose)
        
        # Evaluate
        svm_results = evaluate_model(svm, X_train_pca, X_test_pca, y_train, y_test, "SVM")
        lr_results = evaluate_model(lr, X_train_pca, X_test_pca, y_train, y_test, "Logistic Regression")
        
        # Save models
        save_models(svm, lr, pca, output_dir=args.output_dir)
        
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"  SVM Test Accuracy: {svm_results['test_acc']:.4f}")
        print(f"  LR Test Accuracy:  {lr_results['test_acc']:.4f}")
        print(f"\n💾 Models saved to: {args.output_dir}")
        print(f"   - svm_model.pkl")
        print(f"   - lr_model.pkl")
        print(f"   - pca.pkl")
        
        print(f"\n🎯 Next steps:")
        print(f"   - Use with: python scripts/predict.py --model {args.output_dir}/svm_model.pkl --image <image.jpg>")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
