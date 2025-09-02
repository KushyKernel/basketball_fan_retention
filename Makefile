# Makefile for Basketball Fan Retention & Revenue Optimization

# Variables
PYTHON = python
PIP = pip
JUPYTER = jupyter
NOTEBOOKS_DIR = notebooks
DATA_RAW = data/raw
DATA_PROCESSED = data/processed

# Default target
.PHONY: all
all: install data models analysis

# Setup and installation
.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name basketball_fan_retention

# Clean targets
.PHONY: clean
clean:
	rm -rf $(DATA_RAW)/api/*
	rm -rf $(DATA_RAW)/bbref/*
	rm -rf $(DATA_PROCESSED)/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: clean-data
clean-data:
	rm -rf $(DATA_RAW)/*
	rm -rf $(DATA_PROCESSED)/*

# Data collection and processing
.PHONY: data
data: synthetic-data api-data bbref-data

.PHONY: synthetic-data
synthetic-data:
	$(PYTHON) scripts/generate_synthetic_data.py

.PHONY: api-data
api-data:
	$(PYTHON) scripts/collect_api_data.py

.PHONY: bbref-data
bbref-data:
	$(PYTHON) scripts/scrape_bbref.py

# Feature engineering
.PHONY: features
features:
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/02_feature_engineering.ipynb --to notebook --inplace

# Model training and evaluation
.PHONY: models
models:
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/03_churn_modeling.ipynb --to notebook --inplace
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/04_model_explanation.ipynb --to notebook --inplace
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/05_survival_clv.ipynb --to notebook --inplace

# Optimization
.PHONY: optimization
optimization:
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/06_offer_optimization.ipynb --to notebook --inplace

# Analysis pipeline
.PHONY: analysis
analysis: features models optimization

# Run all notebooks in sequence
.PHONY: run-notebooks
run-notebooks:
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/01_data_exploration.ipynb --to notebook --inplace
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/02_feature_engineering.ipynb --to notebook --inplace
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/03_churn_modeling.ipynb --to notebook --inplace
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/04_model_explanation.ipynb --to notebook --inplace
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/05_survival_clv.ipynb --to notebook --inplace
	$(JUPYTER) nbconvert --execute $(NOTEBOOKS_DIR)/06_offer_optimization.ipynb --to notebook --inplace

# Generate HTML reports
.PHONY: reports
reports:
	$(JUPYTER) nbconvert --to html $(NOTEBOOKS_DIR)/*.ipynb --output-dir reports/

# Development targets
.PHONY: lint
lint:
	flake8 src/ scripts/
	black --check src/ scripts/

.PHONY: format
format:
	black src/ scripts/

.PHONY: test
test:
	pytest tests/

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all          - Run complete pipeline (install, data, models, analysis)"
	@echo "  install      - Install Python dependencies"
	@echo "  data         - Collect all data (synthetic, API, web scraping)"
	@echo "  features     - Run feature engineering"
	@echo "  models       - Train and evaluate all models"
	@echo "  optimization - Run offer allocation optimization"
	@echo "  analysis     - Run analysis pipeline (features + models + optimization)"
	@echo "  run-notebooks- Execute all notebooks in sequence"
	@echo "  reports      - Generate HTML reports from notebooks"
	@echo "  clean        - Clean cache and processed data"
	@echo "  clean-data   - Clean all data directories"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black"
	@echo "  test         - Run tests"
