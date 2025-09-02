# Configuration Directory

This directory contains configuration files for the basketball fan retention analysis project. Configuration management is centralized here to maintain consistency across different environments and deployment scenarios.

## Configuration Files

### `config.yaml` - Main Configuration
**Purpose**: Production configuration with optimized settings for the complete analysis pipeline

**Key Sections**:
- **Data Sources**: Database connections, API endpoints, file paths
- **Feature Engineering**: Parameters for feature creation and transformation
- **Model Configuration**: ML model hyperparameters and training settings
- **Analysis Settings**: Statistical analysis parameters and thresholds
- **Output Configuration**: File paths, formats, and export settings
- **Visualization**: Plot styling, output formats, and dashboard settings

**Security**: Contains actual database credentials and API keys (kept private)

### `config.example.yaml` - Template Configuration
**Purpose**: Template file showing configuration structure without sensitive data

**Usage**:
- Copy to `config.yaml` for new installations
- Reference for understanding configuration options
- Safe to commit to version control (no sensitive data)
- Documentation for all available configuration parameters

## Configuration Structure

### Data Configuration
```yaml
data:
  sources:
    database:
      host: "your-database-host"
      port: 5432
      database: "basketball_analytics"
      username: "your-username"
      password: "your-password"
    
    apis:
      basketball_api:
        base_url: "https://api.basketball-reference.com"
        api_key: "your-api-key"
        rate_limit: 100  # requests per minute
    
    files:
      raw_data_dir: "data/raw"
      processed_data_dir: "data/processed"
      output_dir: "data/processed/models"
```

### Feature Engineering
```yaml
feature_engineering:
  temporal_features:
    time_windows: [7, 14, 30, 90]  # days
    aggregations: ["mean", "sum", "max", "std"]
    seasonal_features: true
  
  behavioral_features:
    engagement_metrics: ["views", "likes", "shares", "comments"]
    loyalty_indicators: ["team_consistency", "season_attendance", "merchandise"]
    churn_predictors: ["days_since_last_activity", "engagement_trend", "spend_decline"]
  
  economic_features:
    external_data: true
    inflation_adjustment: true
    regional_income_data: true
```

### Model Configuration
```yaml
models:
  churn_prediction:
    algorithm: "xgboost"
    hyperparameters:
      n_estimators: 200
      max_depth: 8
      learning_rate: 0.1
      subsample: 0.8
    
    validation:
      test_size: 0.2
      cv_folds: 5
      stratify: true
  
  survival_analysis:
    model_type: "cox_proportional_hazards"
    time_column: "days_active"
    event_column: "churned"
    
  clv_estimation:
    discount_rate: 0.1
    prediction_horizon: 365  # days
    confidence_intervals: true
```

### Analysis Settings
```yaml
analysis:
  customer_segmentation:
    method: "rfm_with_behavioral"
    n_segments: 5
    segment_names: ["champions", "potential_loyalists", "new_customers", "at_risk", "churned"]
  
  statistical_validation:
    significance_level: 0.05
    minimum_sample_size: 100
    correlation_threshold: 0.3
  
  business_metrics:
    currency: "USD"
    fiscal_year_start: "10-01"  # October 1st
    kpi_targets:
      retention_rate: 0.85
      clv_growth: 0.15
      engagement_increase: 0.20
```

### Visualization Configuration
```yaml
visualization:
  style:
    theme: "professional"
    color_palette: "viridis"
    figure_size: [12, 8]
    dpi: 300
  
  outputs:
    save_formats: ["png", "pdf", "svg"]
    interactive: true
    dashboard_auto_refresh: 3600  # seconds
  
  charts:
    survival_curves:
      confidence_bands: true
      risk_table: true
    
    clv_distributions:
      bins: 50
      outlier_percentile: 0.99
```

## Environment Management

### Development vs Production
The configuration system supports different environments:

**Development**:
- Local file-based data sources
- Reduced dataset sizes for faster iteration
- Debug logging enabled
- Interactive visualizations

**Production**:
- Database connections
- Full dataset processing
- Optimized performance settings
- Automated reporting

### Configuration Loading
```python
from src.config import load_config

# Load configuration (automatically detects environment)
config = load_config()

# Access nested configuration
database_config = config['data']['sources']['database']
model_params = config['models']['churn_prediction']['hyperparameters']
```

## Security Considerations

### Sensitive Data Handling
- **Credentials**: Database passwords, API keys stored in `config.yaml` (not versioned)
- **Example File**: `config.example.yaml` contains placeholders only
- **Environment Variables**: Support for environment variable substitution
- **Encryption**: Consider encrypting sensitive configuration values

### Best Practices
```yaml
# Use environment variables for sensitive data
database:
  password: "${DATABASE_PASSWORD}"  # Resolved from environment

# Or reference external files
api_keys:
  basketball_api: "file:///path/to/secret/api_key.txt"
```

## Configuration Validation

### Schema Validation
The configuration system includes validation to ensure:
- Required parameters are present
- Data types are correct
- Value ranges are within acceptable limits
- File paths exist and are accessible

### Example Validation
```python
from src.config import validate_config

# Validate configuration on startup
config = load_config()
validation_errors = validate_config(config)

if validation_errors:
    print("Configuration errors found:")
    for error in validation_errors:
        print(f"  - {error}")
    exit(1)
```

## Usage Examples

### Loading Specific Configuration Sections
```python
from src.config import load_config

config = load_config()

# Model configuration
model_config = config['models']['churn_prediction']
estimators = model_config['hyperparameters']['n_estimators']

# Data paths
data_config = config['data']
raw_data_path = data_config['files']['raw_data_dir']
```

### Dynamic Configuration Updates
```python
# Update configuration at runtime
config['models']['churn_prediction']['hyperparameters']['learning_rate'] = 0.05

# Save updated configuration
save_config(config, 'config/config_modified.yaml')
```

### Environment-Specific Loading
```python
# Load configuration for specific environment
dev_config = load_config(environment='development')
prod_config = load_config(environment='production')
```

## Extending Configuration

### Adding New Sections
1. Define new configuration section in `config.example.yaml`
2. Add validation rules in `src/config.py`
3. Document the new section in this README
4. Update dependent code to use new configuration

### Custom Configuration Loaders
```python
def load_custom_config(config_path, merge_with_defaults=True):
    """Load custom configuration file with optional defaults merging."""
    custom_config = yaml.safe_load(open(config_path))
    
    if merge_with_defaults:
        default_config = load_config()
        return merge_configs(default_config, custom_config)
    
    return custom_config
```

## Configuration Templates

### Minimal Configuration
For quick start scenarios, a minimal configuration includes:
```yaml
data:
  files:
    raw_data_dir: "data/raw"
    processed_data_dir: "data/processed"

models:
  churn_prediction:
    algorithm: "xgboost"

analysis:
  customer_segmentation:
    n_segments: 5
```

### Full Production Configuration
See `config.example.yaml` for complete production-ready configuration template with all available options documented.

## Troubleshooting

### Common Configuration Issues
1. **Missing API Keys**: Check that all required API keys are provided
2. **Invalid Paths**: Ensure all file paths exist and are accessible
3. **Database Connections**: Verify database credentials and network connectivity
4. **YAML Syntax**: Validate YAML syntax using online validators
5. **Environment Variables**: Confirm environment variables are properly set

### Configuration Debugging
```python
# Debug configuration loading
import logging
logging.basicConfig(level=logging.DEBUG)

config = load_config()  # Will show detailed loading information
```

For more information on configuration management patterns, see [../docs/configuration_guide.md](../docs/configuration_guide.md)
