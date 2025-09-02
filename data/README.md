# Data Directory

This directory contains all data sources, processed datasets, and analysis outputs for the basketball fan retention project.

## Directory Structure

```
data/
├── raw/                         # Source data (never modify directly)
│   ├── api/                     # NBA API data cache
│   ├── bbref/                   # Basketball Reference scraped data
│   └── synth/                   # Generated synthetic datasets
│
└── processed/                   # Processed datasets and outputs
    ├── engineered_features.csv  # Feature engineering results
    ├── final_train_features.csv # Training dataset
    ├── final_test_features.csv  # Test dataset
    ├── models/                  # Trained ML models
    ├── figures/                 # Analysis visualizations
    ├── clv/                     # Customer lifetime value results
    ├── ltv/                     # Legacy LTV outputs
    └── offer_optimization/      # Campaign optimization results
```

## Data Sources

### 1. Synthetic Customer Data (`raw/synth/`)
- **enhanced_customers.csv**: Customer profiles with demographics and behavioral segments
- **enhanced_customer_interactions.csv**: Monthly engagement and transaction data
- **enhanced_team_performance.csv**: NBA team performance and characteristics

**Generation**: Ultra-realistic synthetic data with 50,000 customers across 4 behavioral segments

### 2. NBA API Data (`raw/api/`)
- **teams.json**: NBA team information and metadata
- **players.json**: Player rosters and statistics
- **games_{season}.json**: Game results and box scores

**Source**: Ball Don't Lie API (https://www.balldontlie.io)

### 3. Basketball Reference Data (`raw/bbref/`)
- **attendance_data.csv**: Historical attendance figures
- **team_game_logs.csv**: Detailed game performance data

**Source**: Basketball Reference (respectful scraping with rate limits)

## Key Datasets

### Customer Features
- **Demographics**: Age, region, income level, tech adoption
- **Behavioral Segments**: Casual, regular, avid, super_fan
- **Subscription**: Tier, price, tenure, payment history
- **Engagement**: Viewing minutes, app usage, social media activity

### Interaction Data
- **Monthly Records**: 48 months of interaction data (2021-2024)
- **RFM Features**: Recency, frequency, monetary analysis
- **Trends**: Rolling averages, trend slopes, seasonality
- **External Factors**: Economic conditions, team performance

## Data Quality

### Validation Checks
- **Schema Validation**: Column types, required fields
- **Business Rules**: Realistic ranges, logical relationships
- **Completeness**: Missing value analysis
- **Consistency**: Cross-dataset integrity

### Quality Metrics
- **Completeness**: >95% for critical fields
- **Accuracy**: Statistical distributions match industry benchmarks
- **Consistency**: No logical contradictions
- **Timeliness**: Data freshness within acceptable ranges

## Usage

### Loading Data
```python
import pandas as pd

# Load customer data
customers = pd.read_csv('data/raw/synth/enhanced_customers.csv')
interactions = pd.read_csv('data/raw/synth/enhanced_customer_interactions.csv')

# Load processed features
train_features = pd.read_csv('data/processed/final_train_features.csv')
test_features = pd.read_csv('data/processed/final_test_features.csv')
```

### Data Generation
```bash
# Generate new synthetic data
python scripts/generate_realistic_data.py

# Validate data quality
python scripts/validate_data.py --all --report data/processed/validation/
```

## Data Pipeline

1. **Generation**: Create synthetic customer and interaction data
2. **Collection**: Gather external NBA data from APIs
3. **Integration**: Combine all data sources with common keys
4. **Validation**: Quality checks and business rule validation
5. **Feature Engineering**: Create modeling features and transformations
6. **Splitting**: Train/test splits with temporal considerations

## Important Notes

- **Never modify raw data directly** - always work with processed copies
- **Version control**: Raw data generation is reproducible with random seeds
- **Privacy**: All customer data is synthetic - no real PII
- **Scalability**: Data generation scripts can create datasets of any size
- **Documentation**: Each dataset includes metadata and column descriptions

## Data Dictionary

### Customer Schema
- `customer_id`: Unique identifier
- `age_group`: Demographic segment
- `region`: Geographic location
- `favorite_team`: Primary team affiliation
- `segment`: Behavioral classification
- `price`: Monthly subscription price
- `team_loyalty_score`: Loyalty metric (0-1)

### Interaction Schema
- `customer_id`: Links to customer data
- `month`: Interaction period
- `engagement_level`: Overall engagement score
- `minutes_watched`: Video consumption
- `tickets_purchased`: Ticket buying behavior
- `merch_spend`: Merchandise purchases
- `app_logins`: Mobile app usage
- `social_media_interactions`: Social engagement

For complete data dictionary, see [data_dictionary.md](data_dictionary.md)
