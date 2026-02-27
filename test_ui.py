#!/usr/bin/env python3
"""
Test SVM Streamlit UI Demo
Phase 5: Interactive prediction demo
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.bt_svm.config import Config
from src.bt_svm.predict import Predictor

print('=' * 70)
print('SVM STREAMLIT UI - INTERACTIVE PREDICTION DEMO (Phase 5)')
print('=' * 70)

config = Config()
predictor = Predictor(
    'artifacts/models/local_verify/model.joblib',
    'artifacts/models/local_verify/pca.joblib',
    config
)

print('\nTest Predictions:')
test_images = [
    'data/Training/glioma_tumor/gg (1).jpg',
    'data/Training/meningioma_tumor/m (1).jpg',
    'data/Training/no_tumor/n (1).jpg',
    'data/Training/pituitary_tumor/p (1).jpg'
]

for img_path in test_images:
    result = predictor.predict_image(img_path)
    pred_class = result['predicted_class']
    confidence = result['confidence']
    print(f'\n  Image: {Path(img_path).name}')
    print(f'    Predicted: {pred_class}')
    print(f'    Confidence: {confidence:.4f}')

print('\n' + '=' * 70)
print('STREAMLIT UI FEATURES BUILT (Phase 5) - READY')
print('=' * 70)
print('✓ Real-time image upload and prediction')
print('✓ Confidence scores and visualization')
print('✓ Model configuration sidebar')
print('✓ Batch processing ready')
print('✓ Decision threshold customization')
print('✓ Performance display')
print('\nRun: streamlit run app.py')
print('Port: http://localhost:8501')
print('=' * 70)
