# Trained SVM Models

This directory contains trained SVM models for brain tumor classification.

## Files

### Model Checkpoints
- **`svm_model.pkl`** - Trained SVM classifier with RBF kernel
- **`lr_model.pkl`** - Trained Logistic Regression classifier (for comparison)
- **`pca.pkl`** - PCA transformer fitted on training data

## Usage

### Training

```bash
python scripts/train.py --data-dir brain_tumor/ --output-dir models/
```

### Single Image Prediction

```bash
python scripts/predict.py \
  --model models/svm_model.pkl \
  --pca models/pca.pkl \
  --image sample.jpg
```

### Batch Prediction

```bash
for img in brain_tumor/Testing/no_tumor/*.jpg; do
  python scripts/predict.py \
    --model models/svm_model.pkl \
    --pca models/pca.pkl \
    --image "$img"
done
```

## Model Card

| Property | Value |
|---|---|
| Algorithm | Support Vector Machine (SVM) |
| Kernel | RBF (Radial Basis Function) |
| Feature Extraction | OpenCV Image Loading |
| Dimensionality Reduction | PCA (98% variance) |
| Input Format | Grayscale MRI image (200×200 px) |
| Feature Dimension | 40,000 → PCA reduced |
| Output | Binary classification (0: No Tumor, 1: Tumor) |
| Framework | scikit-learn |

## Training Details

### Data Preprocessing
1. **Load:** Grayscale images from `brain_tumor/` folders
2. **Resize:** All images resized to 200×200 pixels
3. **Flatten:** Images flattened to 1D arrays (40,000 features)
4. **Normalize:** Pixel values scaled to [0, 1]
5. **Dimensionality Reduction:** PCA applied to retain 98% variance

### Model Parameters

**SVM:**
```python
kernel = 'rbf'         # Radial Basis Function kernel
C = 1.0                # Regularization parameter
gamma = 'scale'        # Kernel coefficient
```

**Logistic Regression:**
```python
C = 0.1                # Inverse regularization strength
max_iter = 1000        # Maximum iterations
```

**PCA:**
```python
n_components = 0.98    # Retain 98% variance
```

### Train/Test Split
- Train: 80% (random selection)
- Test: 20% (random selection)
- Stratified split to maintain class balance

## Performance Metrics

Expected metrics on test set (depends on dataset):

| Metric | Example Value |
|---|---|
| Training Accuracy | 95-98% |
| Testing Accuracy | 85-95% |
| Precision | 85-95% |
| Recall | 85-95% |
| F1 Score | 85-95% |

See training logs or `results/metrics.txt` for exact values.

## File Sizes

```
svm_model.pkl     ~5-10 MB   (depends on training data size)
lr_model.pkl      ~5-10 MB   (similar to SVM)
pca.pkl           ~1-2 MB    (PCA components)
```

## Loading Models Programmatically

```python
import pickle

# Load model
with open('models/svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Load PCA
with open('models/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Use for inference
prediction = svm.predict(transformed_image)
```

## Model Versioning

To track versions:

1. **By date:** `svm_model_2026-02-27.pkl`
2. **By iteration:** `svm_model_v1.pkl`, `svm_model_v2.pkl`
3. **Git tags:** Tag commits with model names
4. **DVC:** Use `dvc add models/` for version control

## Hyperparameter Tuning

To improve model performance:

```bash
# Experiment with parameters
python scripts/train.py --data-dir brain_tumor/ --pca-variance 0.99
python scripts/train.py --data-dir brain_tumor/ --img-size 256
python scripts/train.py --data-dir brain_tumor/ --test-split 0.15
```

## Limitations

- **Grayscale only:** Ignores color information (if available)
- **Fixed size:** Resizes all images to 200×200, may lose detail
- **PCA compression:** May discard useful information
- **SVM scalability:** Slower on very large datasets (100K+ images)
- **Linear kernels:** RBF kernel assumes separable data in high dimensions

## Next Steps

- [ ] Export model to ONNX for deployment
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Cross-validation analysis
- [ ] Compare with deep learning models (CNN)
- [ ] Create inference API (Flask/FastAPI)
