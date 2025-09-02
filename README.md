# Basketball Fan Retention & Revenue Optimisation

A analytics system for predicting basketball fan churn, estimating customer lifetime value, and optimizing marketing offers to maximize revenue and retention.

## Project Overview

This repository provides a complete end-to-end pipeline for basketball fan retention analysis, covering:

- **Data Collection**: NBA API integration, web scraping, and synthetic data generation
- **Feature Engineering**: RFM analysis, behavioral patterns, and team performance integration
- **Churn Prediction**: Machine learning models with SMOTE, calibration, and interpretability
- **Survival Analysis**: Cox proportional hazards modeling for lifetime estimation
- **CLV Modeling**: Customer lifetime value calculation with discounting
- **Offer Optimization**: Linear programming for optimal marketing campaign allocation

## Repository Structure

```
basketball_fan_retention/
├── config/
│   └── config.yaml              # Configuration parameters
├── data/
│   ├── raw/
│   │   ├── api/                 # Ball Don't Lie API cache
│   │   ├── bbref/               # Basketball Reference scraped data
│   │   └── synth/               # Synthetic customer data
│   └── processed/
│       ├── models/              # Trained models and artifacts
│       ├── figures/             # Analysis plots and visualizations
│       ├── ltv/                 # CLV estimates and survival results
│       └── assignments/         # Optimal customer-offer assignments
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and data quality assessment
│   ├── 02_feature_engineering.ipynb  # RFM, behavioral, and team features
│   ├── 03_churn_modeling.ipynb       # Churn prediction models
│   ├── 04_model_explanation.ipynb    # SHAP interpretability analysis
│   ├── 05_survival_clv.ipynb         # Survival analysis and CLV modeling
│   └── 06_offer_optimization.ipynb   # Campaign allocation optimization
├── scripts/
│   ├── generate_synthetic_data.py    # Synthetic data generation CLI
│   ├── collect_api_data.py          # Ball Don't Lie API data collection
│   └── scrape_bbref.py              # Basketball Reference scraping
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration utilities
│   ├── data_collection.py           # API and scraping classes
│   └── synthetic_data.py            # Synthetic data generation
├── Makefile                          # Automated pipeline commands
└── requirements.txt                  # Python dependencies
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/KushyKernel/basketball_fan_retention.git
cd basketball_fan_retention

# Install dependencies
pip install -r requirements.txt

# Setup Jupyter kernel
python -m ipykernel install --user --name basketball_fan_retention
```

### 2. Generate Synthetic Data

```bash
# Generate 10k customers with 24 months of data
python scripts/generate_synthetic_data.py

# Or use the Makefile
make synthetic-data
```

### 3. Run Analysis Pipeline

```bash
# Run complete pipeline
make all

# Or run individual components
make features      # Feature engineering
make models        # Train churn models
make optimization  # Offer allocation
```

### 4. Explore Results

Launch Jupyter and explore the notebooks in order:
1. `01_data_exploration.ipynb` - Data overview and quality assessment
2. `02_feature_engineering.ipynb` - RFM and behavioral features
3. `03_churn_modeling.ipynb` - Churn prediction models
4. `04_model_explanation.ipynb` - SHAP interpretability
5. `05_survival_clv.ipynb` - CLV and survival analysis
6. `06_offer_optimization.ipynb` - Campaign optimization

## Data Sources

### 1. Ball Don't Lie API
- **URL**: https://www.balldontlie.io/api/v1
- **Data**: NBA games, teams, players, box scores
- **Features**: Rate limiting, caching, retry logic with exponential backoff
- **Usage**: `python scripts/collect_api_data.py --seasons 2023 2024`

### 2. Basketball Reference Scraping
- **URL**: https://www.basketball-reference.com
- **Data**: Team game logs, attendance figures, historical statistics
- **Features**: Respectful scraping with rate limits, data cleaning
- **Usage**: `python scripts/scrape_bbref.py --attendance --game-logs`

### 3. Synthetic Customer Data
- **Generation**: 10,000 customers across 4 segments (casual, die-hard, family, corporate)
- **Attributes**: Demographics, subscription tiers, pricing, signup dates
- **Interactions**: Monthly engagement, spending, support tickets, app usage
- **Realism**: Statistical distributions based on industry benchmarks

## Feature Engineering

### Customer Features (RFM Analysis)
- **Recency**: Days since last interaction
- **Frequency**: Number of monthly interactions
- **Monetary**: Total and average spending patterns
- **Tenure**: Subscription length and lifecycle stage
- **Engagement**: Viewing minutes, app logins, content consumption

### Team Performance Features
- **Win Rate**: Monthly team performance vs league average
- **Point Differential**: Scoring margin trends
- **Schedule Impact**: Back-to-back games, travel fatigue
- **Star Players**: Absence tracking and impact

### Behavioral Features
- **Rolling Averages**: 3-month engagement trends
- **Trend Analysis**: Feature slope calculations
- **Consistency Metrics**: Coefficient of variation in engagement
- **Interaction Features**: Cross-feature relationships

## Modeling Approach

### 1. Churn Prediction
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Class Imbalance**: SMOTE oversampling and class weighting
- **Evaluation**: ROC-AUC, PR-AUC, calibration curves
- **Interpretability**: SHAP values for feature importance

### 2. Survival Analysis
- **Method**: Cox Proportional Hazards regression
- **Output**: Expected remaining subscription months
- **Validation**: Concordance index and residual analysis
- **Applications**: CLV estimation and retention targeting

### 3. Customer Lifetime Value
- **Formula**: `CLV = Σ(Monthly_Revenue_t × Discount_Factor_t)`
- **Discounting**: 5% annual rate (configurable)
- **Segmentation**: CLV quintiles for targeted strategies
- **Sensitivity**: Monte Carlo uncertainty quantification

### 4. Offer Optimization
- **Method**: Mixed-Integer Linear Programming (MILP)
- **Objective**: Maximize expected CLV uplift
- **Constraints**: Budget limits, contact frequency caps
- **Solver**: PuLP with CBC/CPLEX backend

## Key Metrics & KPIs

### Model Performance
- **Churn Model ROC-AUC**: Target >0.80
- **Churn Model PR-AUC**: Target >0.60
- **Survival Model C-Index**: Target >0.70
- **CLV Prediction MAPE**: Target <20%

### Business Metrics
- **Customer Retention Rate**: Baseline vs. optimized campaigns
- **Revenue per Customer**: Monthly and lifetime values
- **Campaign ROI**: Revenue uplift vs. offer costs
- **Churn Reduction**: Percentage point improvement

## Configuration

Key parameters in `config/config.yaml`:

```yaml
# Model Configuration
TRAIN_TEST_SPLIT_DATE: "2024-01-01"
VALIDATION_MONTHS: 6
CLASS_WEIGHT_STRATEGY: "balanced"
SMOTE_SAMPLING_STRATEGY: 0.5

# CLV Configuration  
DISCOUNT_RATE: 0.05
MONTHLY_DISCOUNT_RATE: 0.004074

# Optimization Configuration
CAMPAIGN_BUDGET: 100000
OFFER_A_COST: 25
OFFER_B_COST: 50
OFFER_A_UPLIFT: 0.15
OFFER_B_UPLIFT: 0.25
```

## Results & Outputs

### Model Artifacts
- `data/processed/models/churn_model_artifacts.pkl` - Trained churn models
- `data/processed/models/feature_importance.csv` - Feature rankings
- `data/processed/models/churn_risk_scores.csv` - Customer risk scores

### CLV Estimates
- `data/processed/ltv/clv_estimates.csv` - Customer lifetime values
- `data/processed/ltv/survival_curves.csv` - Survival probabilities
- `data/processed/ltv/clv_segments.csv` - Value-based segmentation

### Campaign Optimization
- `data/processed/assignments/optimal_allocation.csv` - Customer-offer assignments
- `data/processed/assignments/campaign_impact.csv` - Expected results
- `data/processed/assignments/holdout_groups.csv` - Control groups for testing

### Visualizations
- `data/processed/figures/` - All analysis plots and charts
- ROC/PR curves, feature importance, survival curves
- CLV distributions, optimization results, campaign simulations

## Development & Testing

### Code Quality
```bash
# Linting
make lint

# Code formatting  
make format

# Run tests
make test
```

### Pipeline Validation
```bash
# Clean and regenerate all data
make clean
make all

# Generate HTML reports
make reports
```

## Usage

### Custom Data Sources
1. Replace synthetic data with real customer data
2. Update feature engineering for specific business context
3. Adjust model parameters for industry characteristics

### Model Tuning
1. Hyperparameter optimization using cross-validation
2. Feature selection with recursive elimination
3. Ensemble methods for improved performance

### Production Deployment
1. Model serving with Flask/FastAPI
2. Automated retraining pipelines
3. A/B testing framework for offer effectiveness

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Dependencies

Core requirements:
- **Python 3.10+**
- **pandas**, **numpy**, **scipy** - Data manipulation
- **scikit-learn**, **xgboost** - Machine learning
- **lifelines** - Survival analysis
- **pulp** - Optimization
- **shap** - Model interpretability
- **matplotlib**, **seaborn**, **plotly** - Visualization

See `requirements.txt` for complete dependency list with versions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Email: analytics@basketball.com
- Documentation: See individual notebook cells for detailed implementation

---

**Note**: This repository contains synthetic data for demonstration purposes. In production, replace with real customer data and validate all model assumptions against actual business metrics.
