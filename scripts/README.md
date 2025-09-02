# Scripts Directory

This directory contains automation scripts for data collection, processing, validation, and pipeline execution.

## Available Scripts

### Data Generation & Collection

#### `generate_realistic_data.py`
**Purpose**: Generate ultra-realistic synthetic basketball fan data

```bash
python scripts/generate_realistic_data.py [options]
```

**Features**:
- 50,000 synthetic customers across 4 behavioral segments
- 48 months of interaction data (2021-2024)
- Advanced behavioral modeling with economic factors
- Team performance integration and seasonality effects

**Options**:
- `--customers N`: Number of customers to generate (default: 50000)
- `--output-dir DIR`: Output directory (default: data/raw/synth)
- `--seed N`: Random seed for reproducibility (default: 42)

#### `collect_api_data.py`
**Purpose**: Collect NBA data from Ball Don't Lie API

```bash
python scripts/collect_api_data.py --seasons 2023 2024 [options]
```

**Features**:
- NBA teams, players, and game data collection
- Rate limiting and error handling
- Data validation and caching
- Comprehensive logging and progress tracking

**Options**:
- `--seasons YEAR...`: Target seasons to collect
- `--teams`: Collect team data
- `--players`: Collect player data  
- `--games`: Collect game data
- `--all`: Collect all data types
- `--verbose`: Enable detailed logging

#### `scrape_bbref.py`
**Purpose**: Scrape Basketball Reference for attendance and historical data

```bash
python scripts/scrape_bbref.py [options]
```

**Features**:
- Respectful scraping with rate limits
- Attendance figures and game logs
- Data cleaning and validation
- Error handling and retry logic

### Data Processing

#### `integrate_data.py` 
**Purpose**: Integrate and combine data from multiple sources

```bash
python scripts/integrate_data.py --output data/processed/integrated.csv [options]
```

**Features**:
- Merge synthetic, API, and scraped data
- Handle data quality issues and missing values
- Create unified customer interaction timeline
- Generate modeling-ready datasets

**Options**:
- `--output FILE`: Output file path
- `--include-external`: Include API and scraped data
- `--all`: Process all available data sources

#### `validate_data.py`
**Purpose**: Comprehensive data quality validation

```bash
python scripts/validate_data.py --all --report output/ [options]
```

**Features**:
- Schema validation (types, constraints, required fields)
- Business rule validation (realistic ranges, relationships)
- Data quality metrics (completeness, accuracy, consistency)
- Detailed validation reports with recommendations

**Options**:
- `--all`: Validate all datasets
- `--dataset NAME`: Validate specific dataset
- `--report DIR`: Generate HTML validation reports
- `--strict`: Fail on warnings

## Usage Examples

### Complete Data Pipeline
```bash
# 1. Generate synthetic data
python scripts/generate_realistic_data.py --customers 50000

# 2. Collect external data
python scripts/collect_api_data.py --seasons 2023 2024 --all

# 3. Integrate all sources
python scripts/integrate_data.py --all --output data/processed/

# 4. Validate final datasets
python scripts/validate_data.py --all --report data/processed/validation/
```

### Custom Data Generation
```bash
# Small dataset for testing
python scripts/generate_realistic_data.py --customers 1000 --seed 123

# Large dataset for production
python scripts/generate_realistic_data.py --customers 100000 --output-dir data/production/
```

### Incremental Data Collection
```bash
# Collect only current season
python scripts/collect_api_data.py --seasons 2024 --games --verbose

# Update team and player data
python scripts/collect_api_data.py --teams --players
```

## Script Architecture

### Common Patterns
- **Configuration-driven**: All scripts use `config/config.yaml`
- **Comprehensive logging**: Structured logging with multiple levels
- **Error handling**: Graceful failures with detailed error messages
- **Progress tracking**: Progress bars and status updates
- **Validation**: Built-in data quality checks

### Dependencies
- **Core**: pandas, numpy, requests, pyyaml
- **Validation**: jsonschema, great_expectations
- **Utilities**: tqdm, pathlib, argparse
- **NBA API**: Custom API client with rate limiting

## Configuration

Scripts use centralized configuration in `config/config.yaml`:

```yaml
data:
  synthetic:
    n_customers: 50000
    date_range:
      start: "2021-01-01" 
      end: "2024-12-31"
    
  api:
    base_url: "https://www.balldontlie.io/api/v1"
    rate_limit: 60  # requests per minute
    timeout: 30     # seconds
    
  validation:
    strict_mode: false
    required_completeness: 0.95
```

## Error Handling

### Common Issues
1. **API Rate Limits**: Scripts include exponential backoff and retry logic
2. **Network Connectivity**: Timeout handling and offline mode support
3. **Data Quality**: Validation errors with clear remediation steps
4. **File Permissions**: Proper error messages for access issues

### Debugging
- Use `--verbose` flag for detailed logging
- Check log files in `logs/` directory
- Validation reports include specific error locations
- Use `--dry-run` mode where available

## Performance Considerations

### Optimization Features
- **Parallel Processing**: Multi-threading for API requests
- **Caching**: API responses cached to avoid redundant calls
- **Memory Management**: Chunked processing for large datasets
- **Efficient I/O**: Optimized file reading/writing

### Resource Usage
- **Memory**: ~2GB RAM for 50k customers
- **Storage**: ~500MB for complete dataset
- **Network**: Respectful API usage with rate limiting
- **CPU**: Multi-core utilization for data processing

## Maintenance

### Regular Tasks
- **Data Refresh**: Re-generate synthetic data monthly
- **API Updates**: Check for API changes and deprecations
- **Validation Rules**: Update business rules as requirements change
- **Performance Monitoring**: Track execution times and resource usage

### Troubleshooting
- **Log Analysis**: Structured logging with correlation IDs
- **Data Lineage**: Track data transformations and sources
- **Version Control**: All scripts are version controlled
- **Testing**: Unit tests for critical functions

For detailed API documentation, see [../docs/api_reference.md](../docs/api_reference.md)
