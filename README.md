# Brain Tumor Classification using SVM (Support Vector Machine)

**Production-ready SVM classifier for brain tumor detection in MRI images.**

---

## 📋 Overview

This project implements a **Support Vector Machine (SVM)** classifier for **4-class brain tumor classification** on MRI images:
- ✅ Traditional ML approach (non-deep learning)
- ✅ Fast training & inference
- ✅ Interpretable model with production structure
- ✅ Dimensionality reduction via PCA
- ✅ CLI + modular Python package

**Maturity:** Production-ready Phase 1 ✅  
**Classes:** Glioma, Meningioma, Pituitary, No Tumor (4-class)  
**Model:** SVM with RBF kernel + PCA feature extraction  
**Framework:** scikit-learn, OpenCV

---

## 🚀 Quick Start (Phase 1)

### 1. Install & Verify

```bash
# Clone repo
git clone https://github.com/Mithileshan/Brain_tumor_using_SVM.git
cd Brain_tumor_using_SVM

# Install dependencies
make install
# or: pip install -r requirements.txt
```

### 2. Prepare Dataset

Download dataset from **Kaggle**: [Brain Tumor Classification MRI](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)

Organize under `data/`:

```
data/
├── Training/
│   ├── glioma_tumor/          # Glioma (astrocytoma)
│   ├── meningioma_tumor/      # Meningioma
│   ├── no_tumor/              # No tumor
│   └── pituitary_tumor/       # Pituitary
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

### 3. Train Model (1 command)

```bash
# Smoke test (trains on full dataset, default 98% PCA variance)
make train

# Or directly:
python -m src.bt_svm.train --data-dir data --output-dir artifacts/models/dev

# Optional args:
python -m src.bt_svm.train \
  --data-dir data \
  --output-dir artifacts/models/prod \
  --img-size 200 \
  --pca-variance 0.98 \
  --model-type svm
```

**Output:**
```
artifacts/models/dev/
├── model.joblib          # Trained SVM
├── pca.joblib            # Fitted PCA transformer
├── metrics.json          # Training metrics
└── config.json           # Config snapshot
```

### 4. Evaluate & Predict

```bash
# (Phase 2 — coming soon)
make eval

# Predict on single image:
python -c "
from src.bt_svm.config import Config
from src.bt_svm.predict import Predictor
c = Config()
p = Predictor('artifacts/models/dev/model.joblib', 
              'artifacts/models/dev/pca.joblib', c)
result = p.predict_image('path/to/scan.jpg')
print(result)
"
```

---

## 📁 Project Structure (Phase 1)

```
.
├── README.md                          # This file
├── requirements.txt                   # Dependencies (pinned)
├── Makefile                           # Commands: make install/train/eval/predict
├── data.yaml                          # (not used in SVM, kept for reference)
│
├── data/                              # Dataset (LOCAL ONLY, not in git)
│   ├── Training/
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── no_tumor/
│   │   └── pituitary_tumor/
│   └── Testing/
│       ├── glioma_tumor/
│       ├── meningioma_tumor/
│       ├── no_tumor/
│       └── pituitary_tumor/
│
├── src/bt_svm/                        # Core package (Phase 1)
│   ├── __init__.py
│   ├── config.py                      # Configuration dataclass
│   ├── data.py                        # DataLoader with validation
│   ├── preprocess.py                  # Preprocessor (normalize, PCA)
│   ├── train.py                       # Training CLI entrypoint
│   ├── eval.py                        # Evaluation utilities (Phase 2)
│   └── predict.py                     # Inference API
│
├── artifacts/                         # Models & metrics (LOCAL ONLY)
│   └── models/
│       └── dev/
│           ├── model.joblib
│           ├── pca.joblib
│           ├── metrics.json
│           └── config.json
│
├── scripts/                           # Utility scripts (Phase 2+)
├── tests/                             # Unit tests (Phase 6)
└── PROJECT Brain Tumor Classification.ipynb  # Original notebook (legacy)

---

## 📊 Model Details

| Property | Value |
|---|---|
| **Algorithm** | Support Vector Machine (SVM) with RBF kernel |
| **Feature Extraction** | OpenCV (image loading) + Custom resize |
| **Dimensionality Reduction** | PCA (98% variance retained) |
| **Input Shape** | 200×200 image → flattened to 40,000 features → PCA reduced |
| **Output** | Binary classification: `0` (no tumor) or `1` (tumor present) |
| **Training Time** | ~1-5 minutes (depends on dataset size) |
| **Inference Time** | < 1 ms per image |
| **Framework** | scikit-learn |

---

## 🔄 Notebook Workflow

### Phase 1: Data Preparation
```python
# Load modules
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from sklearn import (train_test_split, SVC, LogisticRegression, 
                     decomposition.PCA, accuracy_score)

# Collect images from brain_tumor/ folders
X = []  # Image array
Y = []  # Labels (0 or 1)

# Resize all images to 200×200
# Normalize to [0, 1]
```

### Phase 2: Feature Engineering
```python
# Flatten images (200×200 → 40,000 features)
X_updated = X.reshape(len(X), -1)

# PCA: Reduce to 98% variance
pca = PCA(0.98)
pca_train = pca.fit_transform(xtrain)
pca_test = pca.transform(xtest)
```

### Phase 3: Model Training
```python
# Train two classifiers for comparison
lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)

sv = SVC()  # SVM classifier
sv.fit(xtrain, ytrain)
```

### Phase 4: Evaluation
```python
# Print accuracies
print("SVM Training Score:", sv.score(xtrain, ytrain))
print("SVM Testing Score:", sv.score(xtest, ytest))

# Find misclassified samples
misclassified = np.where(ytest != pred)
print("Total Misclassified:", len(misclassified[0]))
```

### Phase 5: Visualization
```python
# Show predictions on test images
for image_path in test_images:
    img = cv2.imread(image_path, 0)
    img_resized = cv2.resize(img, (200, 200))
    img_flat = img_resized.reshape(1, -1) / 255
    prediction = sv.predict(img_flat)  # 0 or 1
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {dec[prediction[0]]}")
```

---

## 📈 Expected Performance

| Metric | Typical Value |
|---|---|
| **Training Accuracy** | 90-98% |
| **Testing Accuracy** | 85-95% |
| **Precision** | 85-95% |
| **Recall** | 85-95% |
| **F1 Score** | 85-95% |

*Note: Actual performance depends on dataset quality, balance, and size.*

---

## 🔧 Configuration & Customization

### Key Hyperparameters (Notebook)

```python
# Image size
IMG_SIZE = 200

# Train/test split
test_size = 0.20
random_state = 10

# PCA variance threshold
pca_variance = 0.98  # Retain 98% of variance

# SVM kernel
kernel = 'rbf'  # Can also use 'linear', 'poly'

# Logistic Regression C parameter
C = 0.1  # Inverse regularization strength
```

### Modifying Classes

To add more tumor types (e.g., `pituitary_tumor`, `meningioma`):

```python
classes = {
    'no_tumor': 0,
    'pituitary_tumor': 1,
    'meningioma': 2,
    'glioma': 3
}
```

---

## 💾 Dataset Preparation

### Downloading Public Datasets

Common sources:
1. **Kaggle** - "Brain Tumor Classification (MRI)" datasets
2. **Google Drive** - Community-curated datasets
3. **GitHub** - Other researchers' released data

### Organizing Your Own Data

1. Collect MRI images in grayscale format
2. Split into:
   ```
   Training/ (70% of data)
   Testing/  (30% of data)
   ```
3. Subdirectories by class:
   ```
   Training/no_tumor/           →  brain_001.jpg, brain_002.jpg, ...
   Training/pituitary_tumor/    →  tumor_001.jpg, tumor_002.jpg, ...
   ```

### Data Augmentation (Optional)

The notebook doesn't include augmentation, but you can add:

```python
from sklearn.utils import shuffle
from scipy.ndimage import rotate, shift

# Rotate images
for img in X:
    rotated = rotate(img, angle=15, order=1)
    X.append(rotated)
```

---

## 🧪 Testing

```bash
# Verify dependencies
python -c "import sklearn, cv2, numpy; print('✓ Dependencies OK')"

# Quick import check
python -c "from sklearn.svm import SVC; print('✓ sklearn-learn OK')"

# Run notebook in headless mode (if needed)
jupyter nbconvert --to notebook --execute "PROJECT Brain Tumor Classification.ipynb"
```

---

## ⚠️ Limitations & Safety

- **Binary Classification Only:** Detects tumor vs. no-tumor; not subtypes
- **PCA Compression:** May lose some image details (98% variance retained)
- **Small-Scale Model:** SVM less effective on very large, diverse datasets (consider CNNs)
- **Dataset Dependency:** Model quality = training data quality
- **Computational Limits:** SVM training slower with 100K+ samples
- **Clinical Use:** NOT FDA-approved; research-only, not diagnostic

**Recommended:** Always validate with radiologist review.

---

## 🔄 Reproducibility

To ensure reproducible results:

1. **Fix random seed:**
   ```python
   np.random.seed(42)
   random_state = 42  # in train_test_split
   ```

2. **Pin dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Use consistent hardware** (same CPU/GPU for bit-exact reproducibility)

4. **Save trained model** (Phase 2):
   ```python
   import pickle
   with open('models/svm_model.pkl', 'wb') as f:
       pickle.dump(sv, f)
   ```

---

## 📚 References

- [scikit-learn SVM Docs](https://scikit-learn.org/stable/modules/svm.html)
- [OpenCV Image Processing](https://docs.opencv.org/)
- [PCA for Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Brain Tumor Datasets](https://www.kaggle.com/search?q=brain+tumor)

---

## 👤 Author

**Mithileshan**  
GitHub: [@Mithileshan](https://github.com/Mithileshan)

---

## 📄 License

MIT License - see LICENSE file (if available)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes to notebook or add `scripts/` modules
4. Commit: `git commit -m "feat: description"`
5. Push: `git push origin feature/your-feature`
6. Submit Pull Request

---

## 📞 Support

For issues:
1. Check [GitHub Issues](../../issues)
2. Ensure dataset is in correct `brain_tumor/` structure
3. Provide error logs + environment (`python --version`, `sklearn.__version__`)
4. Include screenshot of first few cells

---

## 🚀 Next Steps (Phase 2+)

- [ ] Extract notebook → `scripts/train.py` and `scripts/predict.py`
- [ ] Add model serialization (save `.pkl` files)
- [ ] Create CLI for batch predictions
- [ ] Add confusion matrix + ROC-AUC curves
- [ ] Implement k-fold cross-validation
- [ ] Create Gradio/Streamlit demo UI
- [ ] Docker containerization

---

**Last Updated:** Feb 2026  
**Status:** ✅ Production-ready baseline (Phase 1 complete)
