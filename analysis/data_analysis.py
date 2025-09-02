"""
Basketball Fan Retention - Core Data Analysis Module

This module provides essential data analysis functions for the basketball fan retention project.
It includes utilities for loading data, basic statistical analysis, and visualization helpers.

Key Functions:
- load_and_validate_data(): Load datasets with validation
- calculate_basic_stats(): Generate summary statistics  
- create_visualization_helpers(): Common plotting utilities
- export_analysis_results(): Save results to files

Usage:
    from analysis.data_analysis import load_and_validate_data, calculate_basic_stats
    
    # Load and analyze data
    data = load_and_validate_data('path/to/data.csv')
    stats = calculate_basic_stats(data)
    
Author: Basketball Fan Retention Analysis Team
Created: 2024
Last Modified: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List, Any, Union
import warnings

# Configure plotting style for consistency
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)

# Set up module logger
logger = logging.getLogger(__name__)


def load_and_validate_data(file_path: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load dataset from file and perform basic validation.
    
    Args:
        file_path (str): Path to the data file (CSV format)
        required_columns (List[str], optional): List of required column names
        
    Returns:
        pd.DataFrame: Loaded and validated dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If required columns are missing
        
    Example:
        >>> data = load_and_validate_data('customers.csv', ['customer_id', 'segment'])
        >>> print(f"Loaded {len(data)} records")
    """
    try:
        # Load the data
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        
        # Validate required columns if specified
        if required_columns:
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
        # Basic data quality checks
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Missing values: {data.isnull().sum().sum()}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise


def calculate_basic_stats(data: pd.DataFrame, group_by_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics for a dataset.
    
    Args:
        data (pd.DataFrame): Dataset to analyze
        group_by_column (str, optional): Column to group analysis by
        
    Returns:
        Dict[str, Any]: Dictionary containing various statistical summaries
        
    Example:
        >>> stats = calculate_basic_stats(customers_data, group_by_column='segment')
        >>> print(f"Total customers: {stats['total_records']}")
    """
    stats = {
        'total_records': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'numeric_summary': data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # Analyze categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        stats['categorical_summary'][col] = {
            'unique_values': data[col].nunique(),
            'top_values': data[col].value_counts().head().to_dict(),
            'missing_count': data[col].isnull().sum()
        }
    
    # Group-by analysis if specified
    if group_by_column and group_by_column in data.columns:
        stats['group_analysis'] = {
            'group_column': group_by_column,
            'group_counts': data[group_by_column].value_counts().to_dict(),
            'group_percentages': (data[group_by_column].value_counts(normalize=True) * 100).round(2).to_dict()
        }
        
        # Numeric columns by group
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats['group_numeric_summary'] = data.groupby(group_by_column)[numeric_columns].agg(['mean', 'median', 'std']).round(3).to_dict()
    
    logger.info(f"Calculated statistics for {len(data)} records")
    return stats


def create_basic_visualizations(data: pd.DataFrame, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Create a set of basic exploratory data visualizations.
    
    Args:
        data (pd.DataFrame): Dataset to visualize
        output_dir (str, optional): Directory to save plots (if None, plots are shown)
        
    Returns:
        Dict[str, str]: Dictionary mapping plot names to file paths (if saved)
        
    Example:
        >>> plots = create_basic_visualizations(customers_data, 'output/figures/')
        >>> print(f"Created {len(plots)} visualizations")
    """
    plots_created = {}
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    # 1. Data Overview Plot
    plt.figure(figsize=(12, 8))
    
    # Missing values heatmap
    plt.subplot(2, 2, 1)
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        missing_data = missing_data[missing_data > 0]
        plt.bar(range(len(missing_data)), missing_data.values.astype(float))
        plt.xticks(range(len(missing_data)), [str(x) for x in missing_data.index], rotation=45)
        plt.title('Missing Values by Column')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Missing Values Status')
    
    # Data types distribution
    plt.subplot(2, 2, 2)
    dtype_counts = data.dtypes.value_counts()
    plt.pie(dtype_counts.values.astype(float), labels=[str(x) for x in dtype_counts.index], autopct='%1.1f%%')
    plt.title('Data Types Distribution')
    
    # Numeric columns correlation (if available)
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        plt.subplot(2, 2, (3, 4))
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Numeric Columns Correlation Matrix')
    else:
        plt.subplot(2, 2, 3)
        plt.text(0.5, 0.5, 'Insufficient Numeric\nColumns for Correlation', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Correlation Analysis')
    
    plt.tight_layout()
    
    if output_dir and output_path:
        overview_path = output_path / 'data_overview.png'
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plots_created['data_overview'] = str(overview_path)
        logger.info(f"Saved data overview plot to {overview_path}")
    else:
        plt.show()
    
    plt.close()
    
    return plots_created


def export_analysis_results(stats: Dict[str, Any], output_file: str) -> None:
    """
    Export analysis results to a formatted text file.
    
    Args:
        stats (Dict[str, Any]): Statistics dictionary from calculate_basic_stats()
        output_file (str): Path to output file
        
    Example:
        >>> stats = calculate_basic_stats(data)
        >>> export_analysis_results(stats, 'analysis_report.txt')
    """
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("Basketball Fan Retention - Data Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Dataset Overview:\n")
            f.write(f"  Total Records: {stats['total_records']:,}\n")
            f.write(f"  Total Columns: {stats['total_columns']}\n")
            f.write(f"  Missing Values: {sum(stats['missing_values'].values()):,}\n\n")
            
            if stats['categorical_summary']:
                f.write("Categorical Columns Summary:\n")
                for col, summary in stats['categorical_summary'].items():
                    f.write(f"  {col}:\n")
                    f.write(f"    Unique Values: {summary['unique_values']}\n")
                    f.write(f"    Missing Count: {summary['missing_count']}\n")
                    f.write(f"    Top Values: {summary['top_values']}\n")
                f.write("\n")
            
            if 'group_analysis' in stats:
                f.write(f"Group Analysis ({stats['group_analysis']['group_column']}):\n")
                for group, count in stats['group_analysis']['group_counts'].items():
                    pct = stats['group_analysis']['group_percentages'][group]
                    f.write(f"  {group}: {count:,} records ({pct:.1f}%)\n")
                f.write("\n")
        
        logger.info(f"Analysis results exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting analysis results: {str(e)}")
        raise


# Utility functions for common data operations
def clean_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by converting to lowercase and replacing spaces with underscores.
    
    Args:
        data (pd.DataFrame): Dataset with columns to clean
        
    Returns:
        pd.DataFrame: Dataset with cleaned column names
    """
    cleaned_data = data.copy()
    cleaned_data.columns = cleaned_data.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return cleaned_data


def detect_outliers(data: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """
    Detect outliers in a numeric column using specified method.
    
    Args:
        data (pd.DataFrame): Dataset containing the column
        column (str): Name of the numeric column to analyze
        method (str): Method to use ('iqr' or 'zscore')
        
    Returns:
        pd.Series: Boolean series indicating outlier positions
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataset")
    
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Column '{column}' is not numeric")
    
    if method == 'iqr':
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data[column] < lower_bound) | (data[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        return pd.Series(z_scores > 3, index=data.index)
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")


# Main execution example
if __name__ == "__main__":
    """
    Example usage of the data analysis module.
    This will run when the module is executed directly.
    """
    print("Basketball Fan Retention - Data Analysis Module")
    print("=" * 60)
    print("This module provides core data analysis functionality.")
    print("\nExample usage:")
    print("  from analysis.data_analysis import load_and_validate_data")
    print("  data = load_and_validate_data('customers.csv')")
    print("  stats = calculate_basic_stats(data)")
    print("\nFor more examples, see the project notebooks.")