.PHONY: help install train eval predict clean

help:
	@echo "Brain Tumor SVM Classification - Makefile Commands"
	@echo "==================================================="
	@echo "  make install       Install dependencies"
	@echo "  make train         Train SVM model"
	@echo "  make eval          Evaluate trained model"
	@echo "  make predict       Predict on single image (IMG=path/to/image.jpg)"
	@echo "  make clean         Remove artifacts and cache"
	@echo ""

install:
	pip install -r requirements.txt

train:
	python -m src.bt_svm.train --data-dir data --output-dir artifacts/models/dev --verbose

eval:
	@echo "Placeholder for evaluation (Phase 2)"

predict:
	@if [ -z "$(IMG)" ]; then \
		echo "Usage: make predict IMG=path/to/image.jpg"; \
	else \
		python -c "from src.bt_svm.config import Config; from src.bt_svm.predict import Predictor; c = Config(); p = Predictor('artifacts/models/dev/model.joblib', 'artifacts/models/dev/pca.joblib', c); print(p.predict_image('$(IMG)'))"; \
	fi

clean:
	rm -rf artifacts/ reports/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.DEFAULT_GOAL := help
