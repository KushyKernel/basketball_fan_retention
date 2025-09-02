# Notebooks Directory

This directory contains the interactive Jupyter notebooks that form the core analysis pipeline for basketball fan retention and revenue optimization.

## Analysis Workflow

The notebooks are designed to be executed in sequence, with each building on the outputs of the previous ones:

```
01_data_exploration.ipynb     →  02_feature_engineering.ipynb  →  03_churn_modeling.ipynb
                              ↓                                ↓
06_offer_optimization.ipynb  ←  05_survival_clv.ipynb        ←  04_model_explanation.ipynb
```

## Notebook Descriptions

### 1. `01_data_exploration.ipynb` - Data Discovery & Quality Assessment
**Purpose**: Understand the synthetic dataset structure and validate data quality

**Key Sections**:
- Dataset overview and summary statistics
- Customer segment analysis and distribution
- Interaction patterns and temporal trends
- Data quality assessment and missing value analysis
- Economic factor impact visualization

**Outputs**:
- Data quality report
- Exploratory visualization plots
- Customer segment profiles
- Baseline metrics and KPIs

**Runtime**: ~5 minutes

---

### 2. `02_feature_engineering.ipynb` - RFM Analysis & Feature Creation
**Purpose**: Create modeling features and perform RFM (Recency, Frequency, Monetary) analysis

**Key Sections**:
- RFM score calculation and customer segmentation
- Behavioral feature engineering (trends, rolling averages)
- Team performance integration
- Interaction effect features
- Feature correlation analysis

**Outputs**:
- Engineered feature datasets
- RFM segment classification
- Feature importance rankings
- Correlation matrices

**Runtime**: ~10 minutes

---

### 3. `03_churn_modeling.ipynb` - Churn Prediction Models
**Purpose**: Build and validate machine learning models for churn prediction

**Key Sections**:
- Train/test split with temporal considerations
- Class imbalance handling (SMOTE oversampling)
- Model training (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter optimization with Optuna
- Model evaluation and performance metrics

**Outputs**:
- Trained model artifacts (`.pkl` files)
- Performance metrics (ROC-AUC, PR-AUC, calibration curves)
- Feature importance scores
- Prediction probabilities for all customers

**Runtime**: ~15 minutes

---

### 4. `04_model_explanation.ipynb` - SHAP Interpretability Analysis
**Purpose**: Understand model decisions and create explanations for business stakeholders

**Key Sections**:
- SHAP value calculation for global and local explanations
- Feature importance decomposition
- Individual customer prediction explanations
- Business rule extraction from model patterns
- Bias and fairness analysis

**Outputs**:
- SHAP explanation plots and summaries
- Individual customer decision explanations
- Business rule documentation
- Model interpretation reports

**Runtime**: ~8 minutes

---

### 5. `05_survival_clv.ipynb` - Survival Analysis & Customer Lifetime Value
**Purpose**: Estimate time-to-churn and calculate customer lifetime value

**Key Sections**:
- Cox Proportional Hazards survival modeling
- Kaplan-Meier survival curves by customer segment
- CLV calculation with economic discounting
- Customer value segmentation (quintiles)
- Risk stratification for targeted interventions

**Outputs**:
- Survival model artifacts
- CLV estimates for all customers
- Customer value segments
- Risk-based customer classifications
- Survival probability curves

**Runtime**: ~12 minutes

---

### 6. `06_offer_optimization.ipynb` - Marketing Campaign Optimization
**Purpose**: Optimize marketing budget allocation using linear programming

**Key Sections**:
- Campaign scenario definition and constraints
- Mixed-integer linear programming (MILP) optimization
- Expected revenue calculation and uplift modeling
- Budget allocation across customer segments
- A/B testing framework and holdout group selection

**Outputs**:
- Optimal customer-offer assignments
- Expected campaign performance metrics
- Budget allocation recommendations
- A/B testing experimental design
- Campaign simulation results

**Runtime**: ~10 minutes

## Getting Started

### Prerequisites
```bash
# Install required packages
pip install -r requirements.txt

# Setup Jupyter kernel
python -m ipykernel install --user --name basketball_fan_retention

# Start Jupyter Lab
jupyter lab
```

### Data Requirements
Before running the notebooks, ensure you have generated the synthetic data:

```bash
# Generate data
python scripts/generate_realistic_data.py

# Validate data quality
python scripts/validate_data.py --all
```

### Execution Order
1. **Start with Notebook 01** - Always begin with data exploration
2. **Run sequentially** - Each notebook depends on outputs from previous ones
3. **Check outputs** - Verify key files are generated before proceeding
4. **Review results** - Examine visualization and metrics at each step

## Key Outputs & Artifacts

### Model Artifacts
- `data/processed/models/churn_model_artifacts.pkl` - Trained churn models
- `data/processed/models/survival_model.pkl` - Cox regression model
- `data/processed/models/feature_importance.csv` - Feature rankings

### Customer Analytics
- `data/processed/clv/customer_clv_estimates.csv` - CLV for all customers
- `data/processed/clv/clv_quintiles_summary.csv` - Value segment summaries
- `data/processed/clv/customers_at_risk.csv` - High churn risk customers

### Campaign Optimization
- `data/processed/offer_optimization/optimal_allocation.csv` - Customer-offer assignments
- `data/processed/offer_optimization/campaign_impact.csv` - Expected results
- `data/processed/offer_optimization/budget_scenario_analysis.csv` - Multi-budget analysis

### Visualizations
- `data/processed/figures/` - All plots and charts
- ROC/PR curves, feature importance, survival curves
- CLV distributions, campaign simulations, SHAP explanations

## Best Practices

### Development Workflow
1. **Clear outputs** before re-running notebooks to avoid inconsistencies
2. **Save checkpoints** after each major section
3. **Document assumptions** and parameter choices
4. **Version control** notebook outputs and key decisions

### Performance Tips
- **Use sampling** for initial development (10% of data)
- **Profile memory usage** for large datasets
- **Cache expensive operations** (model training, SHAP calculations)
- **Parallel processing** where supported

### Reproducibility
- **Set random seeds** in all notebooks (already configured)
- **Version pin dependencies** in requirements.txt
- **Document environment** (Python version, package versions)
- **Use configuration files** for key parameters

## Customization

### Business Context
- **Modify customer segments** to match your organization
- **Adjust economic factors** for different markets
- **Update team performance data** with real NBA statistics
- **Customize offers and campaigns** for your marketing strategy

### Technical Modifications
- **Scale dataset size** by adjusting generation parameters
- **Add new features** following existing patterns
- **Experiment with different models** (neural networks, ensemble methods)
- **Integrate external data sources** (weather, social media, economics)

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce dataset size or use chunked processing
2. **Missing dependencies**: Check requirements.txt and install missing packages
3. **Model convergence**: Adjust hyperparameters or feature scaling
4. **Visualization errors**: Ensure matplotlib backend is properly configured

### Performance Issues
- **Slow model training**: Use smaller hyperparameter search space
- **Large memory usage**: Clear unnecessary variables and use `del` statements
- **Long execution times**: Profile code and optimize bottlenecks

### Getting Help
- **Error messages**: Check notebook cell outputs for detailed error information
- **Documentation**: See individual notebook markdown cells for explanations
- **Community**: Open GitHub issues for bugs or feature requests

## Advanced Usage

### Batch Processing
```bash
# Run all notebooks programmatically
jupyter nbconvert --execute --to notebook notebooks/*.ipynb

# Generate HTML reports
jupyter nbconvert --to html notebooks/*.ipynb
```

### Integration with MLOps
- **Model versioning**: Track model artifacts with MLflow or similar
- **Automated retraining**: Schedule periodic model updates
- **A/B testing**: Integrate with experimentation platforms
- **Monitoring**: Track model performance and data drift

For detailed technical documentation, see [../docs/model_documentation.md](../docs/model_documentation.md)
