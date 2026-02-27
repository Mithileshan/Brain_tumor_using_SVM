# Brain Tumor SVM Classification - Model Card

**Date:** February 27, 2026  
**Version:** 1.0  
**Phase:** Production-Ready (Phase 1-8 Complete)

---

## 📋 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | Support Vector Machine (RBF kernel) |
| **Framework** | scikit-learn 1.3.0 |
| **Task** | Multi-class classification (4 classes) |
| **Classes** | Glioma, Meningioma, Pituitary, No Tumor |
| **Input Size** | 200×200 grayscale MRI images |
| **Feature Extraction** | PCA (98% variance retained) |
| **Training Data** | 2,870 images |
| **Test Data** | 394 images |
| **Training Time** | ~90 seconds (CPU) |
| **Inference Time** | <1ms per image |

---

## 📊 Performance Metrics

### Phase 1 Training Results
- **Train Accuracy:** 96.90%
- **Test Accuracy:** 72.08%
- **Weighted F1-Score:** 0.6878
- **Weighted Precision:** 0.7388
- **Weighted Recall:** 0.7208

### Per-Class Performance (Phase 3)
Generated via evaluation script: `scripts/evaluate.py`

---

## 🔄 Training Configuration

**Hyperparameters:**
```json
{
  "img_size": 200,
  "pca_variance": 0.98,
  "svm_kernel": "rbf",
  "svm_C": 1.0,
  "random_state": 42,
  "test_split": 0.2
}
```

**Dataset:** [Kaggle Brain Tumor Classification MRI](https://www.kaggle.com/sartajbhuvaji/brain-tumor-classification-mri)

---

## 📦 Model Artifacts (Phase 4)

**Location:** `artifacts/models/dev/`

```
artifacts/models/dev/
├── model.joblib          # Trained SVM model
├── pca.joblib            # Fitted PCA transformer
├── config.json           # Training configuration
├── metrics.json          # Training metrics
└── dataset_manifest.csv  # Dataset tracking (Phase 2)
```

**Versioning:** Use semantic versioning for production deployments
- `v1/` - Current production model
- `v2/` - Next iteration (after retraining)

---

## 🚀 Usage

### Command-Line Training
```bash
python -m src.bt_svm.train \
  --data-dir data \
  --output-dir artifacts/models/dev \
  --img-size 200 \
  --pca-variance 0.98 \
  --model-type svm
```

### Inference - Python API
```python
from src.bt_svm.predict import Predictor
from src.bt_svm.config import Config

config = Config()
predictor = Predictor(
    'artifacts/models/dev/model.joblib',
    'artifacts/models/dev/pca.joblib',
    config
)

result = predictor.predict_image('path/to/mri.jpg')
print(result)
# Output: {'image_path': ..., 'predicted_class': 'glioma_tumor', ...}
```

### Interactive UI (Phase 5)
```bash
streamlit run app.py
```

---

## 🧪 Testing (Phase 6)

Run test suite:
```bash
pytest tests/ -v
```

Tests include:
- Configuration validation
- Data preprocessing
- PCA dimensionality reduction
- Model shape outputs

---

## 🐳 Docker Deployment (Phase 7)

Build image:
```bash
docker build -t brain-tumor-svm:latest .
```

Run container:
```bash
docker run -p 8501:8501 brain-tumor-svm:latest
```

---

## ✅ CI/CD Pipeline (Phase 8)

GitHub Actions workflows defined in `.github/workflows/ci.yml`:
- **Lint:** flake8, black, ruff checks
- **Test:** pytest on Python 3.9, 3.10
- **Docker:** Build and validate image
- **Trigger:** On push to main/develop or PR

---

## ⚠️ Limitations & Considerations

1. **Overfitting Risk:** Train accuracy (96.9%) >> Test accuracy (72.1%)
   - Consider regularization improvements in Phase 2+
   - Evaluate cross-validation performance

2. **Class Imbalance:** 
   - Glioma: 826 | Meningioma: 822 | Pituitary: 827 | No Tumor: 395
   - No Tumor class underrepresented

3. **Data Requirements:**
   - Dataset not included in repo (local-only)
   - Users must download from Kaggle

4. **Hardware:**
   - CPU inference: <1ms
   - GPU not required for inference

---

## 📈 Next Steps

- **Phase 9:** Cross-validation & ablation studies
- **Phase 10:** Model ensemble techniques
- **Phase 11:** Transfer learning via deep feature extraction
- **Phase 12:** Production monitoring & retraining pipeline

---

## 📞 Support

For issues, feature requests, or questions:
- GitHub Issues: [Brain_tumor_using_SVM/issues](https://github.com/Mithileshan/Brain_tumor_using_SVM/issues)
- Documentation: See [README.md](README.md)

---

**Author:** Mithileshan  
**Created:** February 2026  
**Last Updated:** February 27, 2026
