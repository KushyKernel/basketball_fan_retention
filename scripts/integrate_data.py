#!/usr/bin/env python3
"""
Basketball Fan Retention - Data Integration Script

This script integrates data from multiple sources to create a unified dataset for analysis.
It combines synthetic data, real NBA data (if available), and creates enhanced feature sets
for machine learning models.

Integration Tasks:
1. Merge customer profiles with interaction history
2. Add team performance metrics to customer interactions
3. Create temporal features (seasonality, trends)
4. Calculate customer lifetime value features
5. Generate behavioral features (engagement patterns, churn indicators)
6. Create final training and test datasets

Data Sources:
- Synthetic customer data (customers.csv)
- Customer interaction history (customer_interactions.csv)
- Team performance data (team_performance.csv)
- External NBA data (if available from Basketball Reference)

Output:
- Integrated customer dataset with all features
- Train/test splits ready for modeling
- Feature engineering pipeline results

Usage:
    python integrate_data.py --all                  # Integrate all data sources
    python integrate_data.py --synthetic-only       # Use only synthetic data
    python integrate_data.py --include-external     # Include external NBA data
    python integrate_data.py --output processed/    # Specify output directory

Author: Basketball Fan Retention Analysis Team
Created: 2024
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from config import get_data_paths, load_config, setup_logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataIntegrator:
    """
    Data integration class for combining multiple data sources into unified datasets.
    
    Handles merging customer profiles, interactions, team performance, and external data
    to create feature-rich datasets for machine learning models.
    """
    
    def __init__(self, include_external: bool = False):
        """
        Initialize the data integrator.
        
        Args:
            include_external (bool): Whether to include external NBA data sources
        """
        self.config = load_config()
        self.data_paths = get_data_paths()
        self.logger = setup_logging()
        self.include_external = include_external
        
        # Cache for loaded datasets
        self._data_cache = {}
    
    def integrate_all_data(self, output_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Integrate all available data sources into unified datasets.
        
        Args:
            output_dir (Optional[str]): Directory to save integrated datasets
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of integrated datasets
        """
        self.logger.info("Starting comprehensive data integration")
        
        # Load base datasets
        customers_df = self._load_customers_data()
        interactions_df = self._load_interactions_data()
        team_performance_df = self._load_team_performance_data()
        
        # External data integration (if requested)
        external_data = {}
        if self.include_external:
            external_data = self._load_external_data()
        
        # Core integration steps
        integrated_df = self._merge_core_datasets(
            customers_df, interactions_df, team_performance_df
        )
        
        # Add external data if available
        if external_data:
            integrated_df = self._add_external_features(integrated_df, external_data)
        
        # Feature engineering
        engineered_df = self._engineer_features(integrated_df)
        
        # Create train/test splits
        train_df, test_df = self._create_train_test_splits(engineered_df)
        
        # Prepare final datasets
        datasets = {
            'integrated': integrated_df,
            'engineered_features': engineered_df,
            'train_features': train_df,
            'test_features': test_df
        }
        
        # Save datasets if output directory specified
        if output_dir:
            self._save_datasets(datasets, output_dir)
        
        self.logger.info("Data integration completed successfully")
        return datasets
    
    def _load_customers_data(self) -> pd.DataFrame:
        """Load customer data with fallback options."""
        if 'customers' in self._data_cache:
            return self._data_cache['customers']
        
        # Try multiple sources in order of preference
        data_sources = [
            ('enhanced_customers', self.data_paths['raw_synth'] / 'enhanced_customers.csv'),
            ('customers', self.data_paths['raw_synth'] / 'customers.csv'),
        ]
        
        customers_df = None
        source_used = None
        
        for source_name, source_path in data_sources:
            if source_path.exists():
                customers_df = pd.read_csv(source_path)
                source_used = source_name
                break
        
        if customers_df is None:
            raise FileNotFoundError("No customer data found in any expected location")
        
        self.logger.info(f"Loaded {len(customers_df)} customers from {source_used}")
        
        # Basic data cleaning
        customers_df = self._clean_customers_data(customers_df)
        
        self._data_cache['customers'] = customers_df
        return customers_df
    
    def _load_interactions_data(self) -> pd.DataFrame:
        """Load customer interaction data."""
        if 'interactions' in self._data_cache:
            return self._data_cache['interactions']
        
        interactions_sources = [
            ('enhanced_interactions', self.data_paths['raw_synth'] / 'enhanced_customer_interactions.csv'),
            ('interactions', self.data_paths['raw_synth'] / 'customer_interactions.csv'),
        ]
        
        interactions_df = None
        source_used = None
        
        for source_name, source_path in interactions_sources:
            if source_path.exists():
                interactions_df = pd.read_csv(source_path)
                source_used = source_name
                break
        
        if interactions_df is None:
            raise FileNotFoundError("No interaction data found")
        
        self.logger.info(f"Loaded {len(interactions_df)} interactions from {source_used}")
        
        # Basic data cleaning
        interactions_df = self._clean_interactions_data(interactions_df)
        
        self._data_cache['interactions'] = interactions_df
        return interactions_df
    
    def _load_team_performance_data(self) -> pd.DataFrame:
        """Load team performance data."""
        if 'team_performance' in self._data_cache:
            return self._data_cache['team_performance']
        
        team_sources = [
            ('enhanced_team_performance', self.data_paths['raw_synth'] / 'enhanced_team_performance.csv'),
            ('team_performance', self.data_paths['raw_synth'] / 'team_performance.csv'),
        ]
        
        team_df = None
        source_used = None
        
        for source_name, source_path in team_sources:
            if source_path.exists():
                team_df = pd.read_csv(source_path)
                source_used = source_name
                break
        
        if team_df is None:
            self.logger.warning("No team performance data found, will create placeholder")
            team_df = self._create_placeholder_team_data()
            source_used = "placeholder"
        
        self.logger.info(f"Loaded {len(team_df)} team performance records from {source_used}")
        
        self._data_cache['team_performance'] = team_df
        return team_df
    
    def _load_external_data(self) -> Dict[str, pd.DataFrame]:
        """Load external NBA data sources."""
        external_data = {}
        
        # Check for Basketball Reference scraped data
        bbref_dir = self.data_paths['raw_bbref']
        if bbref_dir.exists():
            for file_path in bbref_dir.glob('*.csv'):
                dataset_name = file_path.stem
                try:
                    df = pd.read_csv(file_path)
                    external_data[dataset_name] = df
                    self.logger.info(f"Loaded external dataset: {dataset_name} ({len(df)} records)")
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {str(e)}")
        
        # Check for API data
        api_dir = self.data_paths['raw_api']
        if api_dir.exists():
            for file_path in api_dir.glob('*.csv'):
                dataset_name = f"api_{file_path.stem}"
                try:
                    df = pd.read_csv(file_path)
                    external_data[dataset_name] = df
                    self.logger.info(f"Loaded API dataset: {dataset_name} ({len(df)} records)")
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {str(e)}")
        
        return external_data
    
    def _clean_customers_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize customer data."""
        df = df.copy()
        
        # Ensure customer_id is integer
        if 'customer_id' in df.columns:
            df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
        
        # Standardize categorical columns
        categorical_columns = ['segment', 'age_group', 'region', 'favorite_team']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['customer_id'])
        if len(df) < original_len:
            self.logger.warning(f"Removed {original_len - len(df)} duplicate customers")
        
        return df
    
    def _clean_interactions_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize interactions data."""
        df = df.copy()
        
        # Ensure customer_id is integer
        if 'customer_id' in df.columns:
            df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
        
        # Parse month column if it exists
        if 'month' in df.columns:
            df['month'] = pd.to_datetime(df['month'], errors='coerce')
        
        # Remove rows with missing critical data
        critical_columns = ['customer_id', 'month']
        for col in critical_columns:
            if col in df.columns:
                before_len = len(df)
                df = df.dropna(subset=[col])
                if len(df) < before_len:
                    self.logger.warning(f"Removed {before_len - len(df)} rows with missing {col}")
        
        return df
    
    def _create_placeholder_team_data(self) -> pd.DataFrame:
        """Create placeholder team performance data."""
        teams = [
            'ATL', 'BOS', 'BRK', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
        ]
        
        # Create placeholder data for 2021-2024
        placeholder_data = []
        for year in [2021, 2022, 2023, 2024]:
            for month in range(1, 13):
                for team in teams:
                    # Random but consistent win rates
                    np.random.seed(hash(f"{team}-{year}-{month}") % 2**32)
                    win_rate = max(0.2, min(0.8, np.random.normal(0.5, 0.15)))
                    
                    placeholder_data.append({
                        'team': team,
                        'month': f"{year}-{month:02d}",
                        'win_rate': win_rate,
                        'wins': int(win_rate * 20),  # Approximate games per month
                        'losses': int((1 - win_rate) * 20)
                    })
        
        return pd.DataFrame(placeholder_data)
    
    def _merge_core_datasets(self, customers_df: pd.DataFrame, 
                           interactions_df: pd.DataFrame, 
                           team_performance_df: pd.DataFrame) -> pd.DataFrame:
        """Merge the core datasets into a unified dataset."""
        self.logger.info("Merging core datasets...")
        
        # Start with interactions as the base (transaction-level data)
        merged_df = interactions_df.copy()
        
        # Add customer information
        merged_df = merged_df.merge(
            customers_df, 
            on='customer_id', 
            how='left',
            suffixes=('', '_customer')
        )
        
        # Add team performance data
        if 'favorite_team' in merged_df.columns and 'team' in team_performance_df.columns:
            # Create month column for merging if needed
            if 'month' in merged_df.columns and 'month' in team_performance_df.columns:
                merged_df['month_str'] = merged_df['month'].dt.strftime('%Y-%m')
                team_performance_df['month_str'] = pd.to_datetime(team_performance_df['month']).dt.strftime('%Y-%m')
                
                merged_df = merged_df.merge(
                    team_performance_df,
                    left_on=['favorite_team', 'month_str'],
                    right_on=['team', 'month_str'],
                    how='left',
                    suffixes=('', '_team')
                )
                
                # Clean up temporary columns
                merged_df = merged_df.drop(['month_str', 'team'], axis=1, errors='ignore')
        
        self.logger.info(f"Merged dataset created with {len(merged_df)} records")
        return merged_df
    
    def _add_external_features(self, df: pd.DataFrame, 
                             external_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Add features from external data sources."""
        self.logger.info("Adding external data features...")
        
        enhanced_df = df.copy()
        
        # Add features from each external dataset
        for dataset_name, external_df in external_data.items():
            try:
                # Implement specific merging logic based on dataset type
                if 'attendance' in dataset_name.lower():
                    enhanced_df = self._merge_attendance_data(enhanced_df, external_df)
                elif 'game_log' in dataset_name.lower():
                    enhanced_df = self._merge_game_log_data(enhanced_df, external_df)
                # Add more external data types as needed
                
            except Exception as e:
                self.logger.warning(f"Failed to merge {dataset_name}: {str(e)}")
        
        return enhanced_df
    
    def _merge_attendance_data(self, df: pd.DataFrame, attendance_df: pd.DataFrame) -> pd.DataFrame:
        """Merge attendance data with main dataset."""
        # Implement attendance data merging logic
        return df  # Placeholder
    
    def _merge_game_log_data(self, df: pd.DataFrame, game_log_df: pd.DataFrame) -> pd.DataFrame:
        """Merge game log data with main dataset."""
        # Implement game log data merging logic
        return df  # Placeholder
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for machine learning."""
        self.logger.info("Engineering additional features...")
        
        feature_df = df.copy()
        
        # Temporal features
        if 'month' in feature_df.columns:
            feature_df['year'] = feature_df['month'].dt.year
            feature_df['month_of_year'] = feature_df['month'].dt.month
            feature_df['quarter'] = feature_df['month'].dt.quarter
            
            # NBA season features
            feature_df['nba_season'] = feature_df.apply(self._determine_nba_season, axis=1)
            feature_df['is_playoffs'] = feature_df['month_of_year'].isin([4, 5, 6])
            feature_df['is_offseason'] = feature_df['month_of_year'].isin([7, 8, 9])
        
        # Customer lifetime features
        if 'customer_id' in feature_df.columns and 'month' in feature_df.columns:
            feature_df = self._add_customer_lifetime_features(feature_df)
        
        # Engagement pattern features
        if 'engagement_level' in feature_df.columns:
            feature_df = self._add_engagement_features(feature_df)
        
        # Team performance features
        if 'win_rate' in feature_df.columns:
            feature_df = self._add_team_performance_features(feature_df)
        
        self.logger.info(f"Feature engineering completed, {len(feature_df.columns)} total features")
        return feature_df
    
    def _determine_nba_season(self, row) -> str:
        """Determine NBA season based on month."""
        if pd.isna(row['month']):
            return 'unknown'
        
        year = row['month'].year
        month = row['month'].month
        
        # NBA season runs from October to June
        if month >= 10:
            return f"{year}-{year+1}"
        else:
            return f"{year-1}-{year}"
    
    def _add_customer_lifetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add customer lifetime value and tenure features."""
        # Calculate months since first interaction
        customer_first_month = df.groupby('customer_id')['month'].min().reset_index()
        customer_first_month.columns = ['customer_id', 'first_interaction_month']
        
        df = df.merge(customer_first_month, on='customer_id', how='left')
        df['months_since_first_interaction'] = (df['month'] - df['first_interaction_month']).dt.days / 30.44
        
        # Calculate total interactions per customer
        customer_interaction_counts = df.groupby('customer_id').size().reset_index(name='total_interactions')
        df = df.merge(customer_interaction_counts, on='customer_id', how='left')
        
        return df
    
    def _add_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engagement pattern features."""
        # Rolling averages of engagement
        if 'customer_id' in df.columns and 'month' in df.columns:
            df = df.sort_values(['customer_id', 'month'])
            
            # 3-month rolling average
            df['engagement_3m_avg'] = df.groupby('customer_id')['engagement_level'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            # Engagement trend (current vs 3-month average)
            df['engagement_trend'] = df['engagement_level'] - df['engagement_3m_avg']
        
        return df
    
    def _add_team_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team performance derived features."""
        # Team performance categories
        df['team_performance_category'] = pd.cut(
            df['win_rate'], 
            bins=[0, 0.35, 0.65, 1.0], 
            labels=['Poor', 'Average', 'Good']
        )
        
        # Team performance trend (if multiple months available)
        if 'customer_id' in df.columns and 'month' in df.columns:
            df = df.sort_values(['favorite_team', 'month'])
            df['team_win_rate_3m_avg'] = df.groupby('favorite_team')['win_rate'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
        
        return df
    
    def _create_train_test_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train and test splits based on temporal splitting."""
        if 'month' in df.columns:
            # Use last 20% of time period for testing
            max_date = df['month'].max()
            min_date = df['month'].min()
            date_range = (max_date - min_date).days
            
            split_date = max_date - timedelta(days=int(date_range * 0.2))
            
            train_df = df[df['month'] < split_date].copy()
            test_df = df[df['month'] >= split_date].copy()
            
            self.logger.info(f"Created temporal split: {len(train_df)} train, {len(test_df)} test records")
        else:
            # Random split if no temporal data
            train_df = df.sample(frac=0.8, random_state=42)
            test_df = df.drop(train_df.index)
            
            self.logger.info(f"Created random split: {len(train_df)} train, {len(test_df)} test records")
        
        return train_df, test_df
    
    def _save_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str) -> None:
        """Save integrated datasets to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, df in datasets.items():
            file_path = output_path / f"{dataset_name}.csv"
            df.to_csv(file_path, index=False)
            self.logger.info(f"Saved {dataset_name}: {file_path} ({len(df)} records)")
        
        # Save integration metadata
        metadata = {
            'integration_timestamp': datetime.now().isoformat(),
            'datasets_created': list(datasets.keys()),
            'record_counts': {name: len(df) for name, df in datasets.items()},
            'feature_counts': {name: len(df.columns) for name, df in datasets.items()},
            'include_external_data': self.include_external
        }
        
        metadata_path = output_path / 'integration_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved integration metadata: {metadata_path}")


def main():
    """Main function to run data integration."""
    parser = argparse.ArgumentParser(
        description="Integrate basketball fan retention data sources"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Integrate all available data sources (default)"
    )
    parser.add_argument(
        "--synthetic-only", 
        action="store_true",
        help="Use only synthetic data sources"
    )
    parser.add_argument(
        "--include-external", 
        action="store_true",
        help="Include external NBA data sources"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="data/processed",
        help="Output directory for integrated datasets"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine integration mode
    include_external = args.include_external and not args.synthetic_only
    
    logger.info("Starting basketball fan retention data integration")
    logger.info(f"Include external data: {include_external}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        integrator = DataIntegrator(include_external=include_external)
        datasets = integrator.integrate_all_data(output_dir=args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA INTEGRATION SUMMARY")
        print("=" * 60)
        
        for dataset_name, df in datasets.items():
            print(f"{dataset_name.upper()}:")
            print(f"  Records: {len(df):,}")
            print(f"  Features: {len(df.columns):,}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        print(f"\nSUCCESS: Data integration completed successfully!")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Critical error during data integration: {str(e)}")
        print(f"\nERROR: Critical error during data integration: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()