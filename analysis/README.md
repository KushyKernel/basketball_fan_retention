# Analysis Directory

This directory contains advanced analysis modules and specialized analytics functions for basketball fan retention modeling.

## Module Overview

### `data_analysis.py`
**Purpose**: Core data analysis functions and utilities

**Key Functions**:
- `calculate_rfm_scores()` - RFM (Recency, Frequency, Monetary) analysis
- `analyze_customer_segments()` - Customer segmentation analysis
- `calculate_engagement_trends()` - Temporal engagement pattern analysis
- `validate_data_quality()` - Data quality assessment utilities
- `generate_summary_statistics()` - Comprehensive dataset summaries

**Usage**:
```python
from analysis.data_analysis import calculate_rfm_scores, analyze_customer_segments

# Calculate RFM scores
rfm_data = calculate_rfm_scores(customer_interactions)

# Analyze customer segments
segment_profiles = analyze_customer_segments(customers, interactions)
```

### `behavioral_analysis.py`
**Purpose**: Advanced behavioral analysis with detailed modeling insights

**Key Features**:
- **Psychological Profiles**: Analysis of customer behavioral archetypes
- **Economic Impact**: Recession and inflation effects on engagement
- **Team Performance Correlations**: Championship and dynasty effects
- **Seasonal Pattern Analysis**: NBA calendar impact on fan behavior
- **Churn Risk Assessment**: Advanced churn prediction modeling
- **Regional Analysis**: Geographic patterns in fan loyalty
- **Technology Adoption**: Multi-generational tech usage patterns

**Outputs**:
- Comprehensive analysis reports with 10+ detailed sections
- Advanced visualizations (9-panel analysis dashboard)
- Statistical validation and realism metrics
- Business insights and recommendations

**Usage**:
```python
# Run complete behavioral analysis
python -c "from analysis.behavioral_analysis import main; main()"

# Or import specific functions
from analysis.behavioral_analysis import analyze_economic_impact
economic_insights = analyze_economic_impact(interactions_data)
```

## Analysis Features

### Customer Behavioral Modeling
- **Fan Archetypes**: Casual, regular, avid, super_fan with distinct psychological profiles
- **Loyalty Metrics**: Team loyalty scoring with geographic and demographic factors
- **Engagement Patterns**: Multi-dimensional engagement scoring across channels
- **Spending Behavior**: Price sensitivity analysis with economic context

### Advanced Statistical Analysis
- **Correlation Analysis**: Behavioral correlation matrices and interaction effects
- **Trend Analysis**: Rolling averages, seasonal decomposition, slope calculations
- **Segmentation**: Data-driven customer segmentation with validation
- **Outlier Detection**: Statistical anomaly identification and handling

### Temporal Analysis
- **Seasonality**: NBA calendar effects (regular season, playoffs, offseason)
- **Economic Cycles**: Recession and inflation impact modeling
- **Team Performance**: Win/loss streaks, championship effects, dynasty periods
- **Viral Moments**: Social media and cultural event impact analysis

## Visualization Capabilities

### Interactive Dashboards
- **Economic Impact**: Year-over-year engagement trends with economic context
- **Championship Effects**: Team performance vs fan engagement correlations
- **Social Media Analysis**: Age-based social media activity patterns
- **Seasonal Patterns**: Monthly engagement cycles with NBA calendar overlay

### Advanced Plots
- **Survival Curves**: Customer retention probability over time
- **CLV Distributions**: Customer lifetime value segmentation
- **Churn Risk Heatmaps**: Risk scoring across customer dimensions
- **Campaign Effectiveness**: Marketing simulation results

### Statistical Validation
- **Realism Metrics**: Validation against industry benchmarks
- **Model Performance**: Cross-validation and holdout testing
- **Business Logic**: Consistency checks and sanity testing
- **Interpretability**: SHAP explanations and feature importance

## Usage Examples

### Basic Analysis Workflow
```python
from analysis.data_analysis import *
from analysis.behavioral_analysis import main as run_behavioral_analysis

# 1. Load and validate data
customers, interactions = load_and_validate_data()

# 2. Run core analysis
rfm_analysis = calculate_rfm_scores(interactions)
segment_profiles = analyze_customer_segments(customers, interactions)

# 3. Generate advanced insights
run_behavioral_analysis()  # Comprehensive analysis with visualizations
```

### Custom Analysis
```python
from analysis.data_analysis import calculate_engagement_trends

# Analyze specific customer segment
avid_fans = customers[customers['segment'] == 'avid']
avid_trends = calculate_engagement_trends(avid_fans, interactions)

# Generate custom visualizations
plot_engagement_trends(avid_trends, title="Avid Fan Engagement Patterns")
```

### Integration with ML Pipeline
```python
from analysis.data_analysis import prepare_modeling_features

# Prepare features for machine learning
modeling_data = prepare_modeling_features(
    customers=customers,
    interactions=interactions,
    feature_engineering=True,
    include_trends=True
)
```

## Output Files

### Analysis Reports
- **behavioral_analysis_report.txt** - Detailed text analysis report
- **customer_segment_profiles.csv** - Customer segmentation results
- **engagement_trend_analysis.csv** - Temporal pattern analysis
- **economic_impact_analysis.csv** - Economic factor correlations

### Visualizations
- **behavioral_analysis_dashboard.png** - 9-panel comprehensive dashboard
- **customer_segment_distribution.png** - Segment breakdown visualizations
- **engagement_trends_by_segment.png** - Temporal engagement patterns
- **economic_correlation_heatmap.png** - Economic factor impact analysis

## Configuration

Analysis modules use configuration from `config/config.yaml`:

```yaml
analysis:
  output_dir: "data/processed/figures"
  validation_thresholds:
    engagement_correlation: 0.3
    loyalty_score_range: [0.0, 1.0]
    realistic_churn_rate: [0.05, 0.25]
  
  visualization:
    figure_size: [20, 16]
    color_palette: "viridis"
    save_high_res: true
```

## Performance Considerations

### Optimization Features
- **Vectorized Operations**: NumPy and pandas optimizations
- **Memory Efficiency**: Chunked processing for large datasets
- **Caching**: Expensive calculations cached between runs
- **Parallel Processing**: Multi-core utilization where applicable

### Scalability
- **Dataset Size**: Tested with 50k+ customers and 2M+ interactions
- **Memory Usage**: ~4GB RAM for complete analysis pipeline
- **Execution Time**: ~5-10 minutes for full ultra-realistic analysis
- **Storage**: ~100MB for all analysis outputs

## Extending the Analysis

### Adding New Analysis Modules
1. Create new `.py` file following existing patterns
2. Implement core analysis functions with proper documentation
3. Add configuration options to `config.yaml`
4. Include unit tests and validation checks
5. Update this README with module description

### Custom Visualizations
```python
def create_custom_analysis_plot(data, title="Custom Analysis"):
    """Create custom visualization following project patterns."""
    fig, ax = plt.subplots(figsize=(12, 8))
    # Your visualization code here
    ax.set_title(title)
    plt.tight_layout()
    return fig
```

### Integration Points
- **Notebook Integration**: Import functions into Jupyter notebooks
- **Pipeline Integration**: Use with automated data processing
- **API Integration**: Expose analysis functions via REST API
- **Reporting Integration**: Generate automated business reports

For detailed API documentation, see [../docs/api_reference.md](../docs/api_reference.md)
