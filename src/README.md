# Source Code Directory

This directory contains the core Python modules and utilities that power the basketball fan retention analysis system. These modules provide reusable functions and classes for data collection, synthetic data generation, and configuration management.

## Module Overview

### `__init__.py`
**Purpose**: Package initialization and module exports

Makes the `src` directory a Python package and defines public API:
```python
from src.config import load_config
from src.data_collection import collect_basketball_data
from src.synthetic_data_generator import generate_realistic_customers
```

### `config.py`
**Purpose**: Configuration management and validation

**Key Functions**:
- `load_config(config_path=None)` - Load and validate configuration from YAML
- `validate_config(config_dict)` - Validate configuration structure and values
- `get_database_config()` - Extract database connection parameters
- `get_model_config(model_name)` - Get model-specific configuration
- `save_config(config_dict, output_path)` - Save configuration to file

**Features**:
- Environment variable substitution
- Configuration schema validation
- Default value handling
- Type checking and range validation

**Usage**:
```python
from src.config import load_config, validate_config

# Load configuration
config = load_config('config/config.yaml')

# Validate before use
errors = validate_config(config)
if not errors:
    db_config = config['data']['sources']['database']
```

### `data_collection.py`
**Purpose**: Real-world data collection from external APIs and web sources

**Key Classes**:
- `BasketballAPICollector` - NBA statistics API integration
- `BasketballReferenceCollector` - Basketball-Reference.com web scraping
- `DataIntegrator` - Combine multiple data sources with deduplication

**Key Functions**:
- `collect_team_data(teams, seasons)` - Team performance and roster data
- `collect_player_stats(players, seasons)` - Individual player statistics
- `collect_game_results(date_range)` - Game outcomes and box scores
- `validate_collected_data(data)` - Data quality checks and validation

**API Integration**:
- NBA Stats API (stats.nba.com)
- Basketball-Reference.com scraping
- Rate limiting and retry logic
- Data caching for efficiency

**Usage**:
```python
from src.data_collection import BasketballAPICollector

collector = BasketballAPICollector(api_key="your_key")
team_data = collector.collect_team_data(['LAL', 'BOS'], ['2022-23', '2023-24'])
```

### `synthetic_data_generator.py`
**Purpose**: Advanced synthetic customer and interaction data generation

**Key Classes**:
- `CustomerGenerator` - Generate realistic customer profiles
- `InteractionGenerator` - Generate customer interaction patterns
- `EconomicSimulator` - Simulate economic factors and their impacts
- `BehavioralModelSimulator` - Generate psychologically realistic behaviors

**Customer Generation Features**:
- **Demographics**: Age, income, location with realistic distributions
- **Psychographics**: Fan engagement levels, team loyalty, spending patterns
- **Behavioral Archetypes**: Casual, regular, avid, super_fan with distinct characteristics
- **Economic Sensitivity**: Income-based price sensitivity and recession impacts

**Interaction Simulation**:
- **Multi-Channel**: App usage, website visits, social media, merchandise, tickets
- **Temporal Patterns**: Seasonality, game schedules, championship effects
- **Engagement Evolution**: Loyalty progression and churn patterns
- **Economic Responses**: Spending changes during economic downturns

**Realism Validation**:
- Industry benchmark comparisons
- Statistical validation of distributions
- Correlation structure preservation
- Business logic consistency checks

**Usage**:
```python
from src.synthetic_data_generator import CustomerGenerator, InteractionGenerator

# Generate customer base
customer_gen = CustomerGenerator(config=config)
customers = customer_gen.generate_customers(n_customers=10000)

# Generate interactions for customers
interaction_gen = InteractionGenerator(config=config)
interactions = interaction_gen.generate_interactions(customers, n_months=24)
```

### `synthetic_data.py` (Legacy)
**Purpose**: Basic synthetic data generation (superseded by `synthetic_data_generator.py`)

**Status**: Maintained for backward compatibility
**Recommendation**: Use `synthetic_data_generator.py` for new development

Contains simpler data generation functions:
- `generate_basic_customers()` - Simple customer profiles
- `generate_basic_interactions()` - Basic interaction patterns
- Legacy utility functions

## Core Functionality

### Configuration Management
The configuration system provides centralized management of all system parameters:

```python
# Load configuration with validation
config = load_config()

# Access nested configuration values
api_key = config['data']['sources']['apis']['basketball_api']['api_key']
model_params = config['models']['churn_prediction']['hyperparameters']

# Environment-specific configurations
dev_config = load_config(environment='development')
```

### Data Collection Pipeline
Real-world data collection with robust error handling:

```python
from src.data_collection import BasketballAPICollector, DataIntegrator

# Initialize collectors
api_collector = BasketballAPICollector(config=config)
web_collector = BasketballReferenceCollector(config=config)

# Collect data from multiple sources
team_data = api_collector.collect_team_data(['all'], ['2023-24'])
player_data = web_collector.collect_player_stats(['all'], ['2023-24'])

# Integrate and deduplicate
integrator = DataIntegrator()
combined_data = integrator.integrate_sources([team_data, player_data])
```

### Synthetic Data Generation
Ultra-realistic synthetic data with psychological and economic modeling:

```python
from src.synthetic_data_generator import CustomerGenerator, InteractionGenerator

# Configure generation parameters
gen_config = {
    'n_customers': 50000,
    'time_period_months': 36,
    'economic_scenario': 'post_pandemic_recovery',
    'team_performance_scenario': 'championship_run'
}

# Generate customer base
customer_gen = CustomerGenerator(config=config)
customers = customer_gen.generate_customers(**gen_config)

# Generate interaction history
interaction_gen = InteractionGenerator(config=config)
interactions = interaction_gen.generate_interactions(
    customers=customers,
    include_economic_effects=True,
    include_team_performance_effects=True
)
```

## Advanced Features

### Economic Modeling
Sophisticated economic factor simulation:

- **Recession Impacts**: Reduced spending, delayed purchases, increased price sensitivity
- **Inflation Effects**: Adjusted pricing, spending pattern changes
- **Regional Economics**: Geographic income variations and local economic conditions
- **Recovery Patterns**: Post-recession behavioral changes and spending recovery

### Behavioral Psychology
Realistic customer behavior modeling:

- **Fan Archetypes**: Distinct psychological profiles with different motivations
- **Loyalty Evolution**: Dynamic loyalty changes based on team performance
- **Social Influences**: Peer effects and social media viral moments
- **Generational Differences**: Age-based technology adoption and engagement patterns

### Data Quality Assurance
Comprehensive validation and quality checks:

- **Statistical Validation**: Distribution checks, correlation preservation
- **Business Logic Validation**: Realistic patterns, constraint satisfaction
- **Benchmark Comparisons**: Industry standard comparisons
- **Realism Scoring**: Quantitative realism assessment

## Performance Optimization

### Efficient Data Generation
- **Vectorized Operations**: NumPy-based computations for large datasets
- **Memory Management**: Chunked processing for memory efficiency
- **Parallel Processing**: Multi-core utilization for independent operations
- **Caching**: Intermediate result caching for repeated operations

### Scalability Features
- **Configurable Scale**: Generate from 1K to 1M+ customers
- **Progressive Generation**: Build datasets incrementally
- **Memory Monitoring**: Track and optimize memory usage
- **Performance Profiling**: Built-in timing and performance metrics

## Error Handling and Logging

### Robust Error Management
```python
import logging
from src.config import setup_logging

# Configure logging
setup_logging(level='INFO', log_file='data_generation.log')

try:
    customers = generate_customers(n_customers=100000)
except DataGenerationError as e:
    logging.error(f"Customer generation failed: {e}")
    # Fallback to smaller dataset
    customers = generate_customers(n_customers=10000)
```

### Comprehensive Logging
- **Progress Tracking**: Real-time generation progress updates
- **Performance Metrics**: Timing and memory usage tracking
- **Quality Metrics**: Data quality assessment logging
- **Error Details**: Detailed error reporting with context

## Testing and Validation

### Unit Testing
Each module includes comprehensive unit tests:
```bash
# Run all source code tests
python -m pytest src/tests/ -v

# Test specific module
python -m pytest src/tests/test_synthetic_data_generator.py -v
```

### Integration Testing
End-to-end testing of complete workflows:
```python
# Test complete data generation pipeline
from src.tests.integration_tests import test_full_pipeline

test_results = test_full_pipeline(
    n_customers=1000,
    validate_quality=True,
    save_outputs=False
)
```

## API Documentation

### Complete Function Reference
All functions include comprehensive docstrings:

```python
def generate_customers(n_customers: int, 
                      config: dict, 
                      seed: int = None) -> pd.DataFrame:
    """
    Generate realistic customer profiles with psychological and economic modeling.
    
    Parameters:
    -----------
    n_customers : int
        Number of customers to generate (1 to 1,000,000+)
    config : dict
        Configuration dictionary with generation parameters
    seed : int, optional
        Random seed for reproducible generation
        
    Returns:
    --------
    pd.DataFrame
        Customer profiles with demographics, psychographics, and behavioral indicators
        
    Raises:
    -------
    DataGenerationError
        If configuration is invalid or generation fails
    """
```

### Type Hints and Validation
All functions use proper type hints and input validation:

```python
from typing import List, Dict, Optional, Union
import pandas as pd

def validate_customer_data(customers: pd.DataFrame) -> List[str]:
    """Validate customer data quality and return list of issues."""
    issues = []
    
    # Check required columns
    required_cols = ['customer_id', 'age', 'income', 'fan_segment']
    missing_cols = set(required_cols) - set(customers.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    return issues
```

## Extension Points

### Custom Data Generators
Extend the system with custom generators:

```python
from src.synthetic_data_generator import BaseGenerator

class CustomFanGenerator(BaseGenerator):
    """Custom fan behavior generator for specific use cases."""
    
    def generate_custom_behaviors(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Generate custom behavioral patterns."""
        # Your custom logic here
        return enhanced_customers
```

### Plugin Architecture
Add new data sources through plugin interface:

```python
from src.data_collection import BaseDataCollector

class CustomAPICollector(BaseDataCollector):
    """Custom API integration following standard interface."""
    
    def collect_data(self, parameters: dict) -> pd.DataFrame:
        """Collect data from custom API source."""
        # Your collection logic here
        return collected_data
```

For detailed API documentation and advanced usage examples, see [../docs/api_reference.md](../docs/api_reference.md)
