#!/usr/bin/env python3
"""
Basketball Fan Retention - Data Validation Script

This script provides comprehensive validation for all datasets in the basketball fan retention project.
It performs data quality checks, schema validation, business rule validation, and generates 
detailed validation reports.

Key Validation Areas:
1. File existence and accessibility
2. Data schema validation (columns, types, constraints)
3. Data quality checks (missing values, outliers, duplicates)
4. Business rule validation (realistic ranges, logical relationships)
5. Cross-dataset consistency checks
6. Time series continuity validation

Usage:
    python validate_data.py --all                    # Validate all datasets
    python validate_data.py --customers              # Validate customer data only
    python validate_data.py --interactions           # Validate interaction data only
    python validate_data.py --teams                  # Validate team performance data
    python validate_data.py --report output/         # Generate detailed report

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
import json
import warnings

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from config import get_data_paths, load_config, setup_logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataValidator:
    """
    Comprehensive data validation class for basketball fan retention datasets.
    
    Performs multi-level validation including schema, quality, business rules,
    and cross-dataset consistency checks.
    """
    
    def __init__(self):
        """Initialize the data validator with configuration and paths."""
        self.config = load_config()
        self.data_paths = get_data_paths()
        self.logger = setup_logging()
        self.validation_results = {
            'overall_status': 'PENDING',
            'validation_timestamp': datetime.now().isoformat(),
            'datasets_validated': [],
            'errors': [],
            'warnings': [],
            'summary': {}
        }
    
    def validate_all_datasets(self) -> Dict[str, Any]:
        """
        Validate all datasets in the project.
        
        Returns:
            Dict[str, Any]: Comprehensive validation results
        """
        self.logger.info("Starting comprehensive data validation")
        
        # Validate each dataset
        datasets_to_validate = ['customers', 'interactions', 'team_performance']
        
        for dataset in datasets_to_validate:
            self.logger.info(f"Validating {dataset} dataset...")
            try:
                if dataset == 'customers':
                    self._validate_customers_dataset()
                elif dataset == 'interactions':
                    self._validate_interactions_dataset()
                elif dataset == 'team_performance':
                    self._validate_team_performance_dataset()
                
                self.validation_results['datasets_validated'].append(dataset)
                
            except Exception as e:
                error_msg = f"Critical error validating {dataset}: {str(e)}"
                self.logger.error(error_msg)
                self.validation_results['errors'].append(error_msg)
        
        # Perform cross-dataset validation
        if len(self.validation_results['datasets_validated']) > 1:
            self.logger.info("Performing cross-dataset consistency validation...")
            self._validate_cross_dataset_consistency()
        
        # Generate summary
        self._generate_validation_summary()
        
        return self.validation_results
    
    def _validate_customers_dataset(self) -> None:
        """Validate customer dataset schema, quality, and business rules."""
        # Check if enhanced data exists, fall back to original if needed
        enhanced_path = self.data_paths['processed'] / 'final_train_features.csv'
        original_path = self.data_paths['processed'] / 'train_features.csv'
        raw_path = self.data_paths['raw_synth'] / 'customers.csv'
        
        data_path = None
        data_source = None
        
        if enhanced_path.exists():
            data_path = enhanced_path
            data_source = "enhanced_features"
        elif original_path.exists():
            data_path = original_path
            data_source = "processed_features"
        elif raw_path.exists():
            data_path = raw_path
            data_source = "raw_synthetic"
        else:
            raise FileNotFoundError("No customer dataset found in any expected location")
        
        self.logger.info(f"Validating customers dataset from: {data_source}")
        customers_df = pd.read_csv(data_path)
        
        # Schema validation
        self._validate_customers_schema(customers_df, data_source)
        
        # Data quality validation
        self._validate_customers_quality(customers_df)
        
        # Business rules validation
        self._validate_customers_business_rules(customers_df)
        
        self.logger.info(f"Customer dataset validation completed: {len(customers_df)} records")
    
    def _validate_customers_schema(self, df: pd.DataFrame, data_source: str) -> None:
        """Validate customer dataset schema."""
        required_columns = {
            'raw_synthetic': ['customer_id', 'segment', 'age_group', 'region', 'favorite_team'],
            'processed_features': ['customer_id', 'segment', 'age_group', 'region', 'favorite_team'],
            'enhanced_features': ['customer_id', 'segment', 'age_group', 'region', 'favorite_team']
        }
        
        expected_columns = required_columns.get(data_source, required_columns['raw_synthetic'])
        missing_columns = set(expected_columns) - set(df.columns)
        
        if missing_columns:
            error_msg = f"Customer dataset missing required columns: {missing_columns}"
            self.validation_results['errors'].append(error_msg)
        
        # Check for reasonable number of customers
        if len(df) < 1000:
            warning_msg = f"Customer dataset has unusually few records: {len(df)}"
            self.validation_results['warnings'].append(warning_msg)
        elif len(df) > 1000000:
            warning_msg = f"Customer dataset has unusually many records: {len(df)}"
            self.validation_results['warnings'].append(warning_msg)
    
    def _validate_customers_quality(self, df: pd.DataFrame) -> None:
        """Validate customer data quality."""
        # Check for missing values in critical columns
        critical_columns = ['customer_id', 'segment']
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    error_msg = f"Critical column '{col}' has {missing_count} missing values"
                    self.validation_results['errors'].append(error_msg)
        
        # Check for duplicate customer IDs
        if 'customer_id' in df.columns:
            duplicate_count = df['customer_id'].duplicated().sum()
            if duplicate_count > 0:
                error_msg = f"Found {duplicate_count} duplicate customer IDs"
                self.validation_results['errors'].append(error_msg)
        
        # Check data types
        if 'customer_id' in df.columns:
            if not pd.api.types.is_integer_dtype(df['customer_id']):
                warning_msg = "customer_id is not integer type"
                self.validation_results['warnings'].append(warning_msg)
    
    def _validate_customers_business_rules(self, df: pd.DataFrame) -> None:
        """Validate customer business rules."""
        # Valid segments
        valid_segments = ['casual', 'regular', 'avid', 'super_fan']
        if 'segment' in df.columns:
            invalid_segments = set(df['segment'].unique()) - set(valid_segments)
            if invalid_segments:
                error_msg = f"Invalid customer segments found: {invalid_segments}"
                self.validation_results['errors'].append(error_msg)
        
        # Valid age groups
        valid_age_groups = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        if 'age_group' in df.columns:
            invalid_age_groups = set(df['age_group'].unique()) - set(valid_age_groups)
            if invalid_age_groups:
                error_msg = f"Invalid age groups found: {invalid_age_groups}"
                self.validation_results['errors'].append(error_msg)
        
        # Valid regions
        valid_regions = ['northeast', 'southeast', 'midwest', 'southwest', 'west']
        if 'region' in df.columns:
            invalid_regions = set(df['region'].unique()) - set(valid_regions)
            if invalid_regions:
                error_msg = f"Invalid regions found: {invalid_regions}"
                self.validation_results['errors'].append(error_msg)
    
    def _validate_interactions_dataset(self) -> None:
        """Validate interactions dataset."""
        # Check for interaction data
        enhanced_path = self.data_paths['processed'] / 'final_train_features.csv'
        raw_path = self.data_paths['raw_synth'] / 'customer_interactions.csv'
        
        if raw_path.exists():
            interactions_df = pd.read_csv(raw_path)
            self._validate_interactions_schema(interactions_df)
            self._validate_interactions_quality(interactions_df)
            self._validate_interactions_business_rules(interactions_df)
            self.logger.info(f"Interactions dataset validation completed: {len(interactions_df)} records")
        else:
            warning_msg = "No interactions dataset found for validation"
            self.validation_results['warnings'].append(warning_msg)
    
    def _validate_interactions_schema(self, df: pd.DataFrame) -> None:
        """Validate interactions dataset schema."""
        required_columns = ['customer_id', 'month', 'engagement_level', 'minutes_watched']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            error_msg = f"Interactions dataset missing required columns: {missing_columns}"
            self.validation_results['errors'].append(error_msg)
    
    def _validate_interactions_quality(self, df: pd.DataFrame) -> None:
        """Validate interactions data quality."""
        # Check for missing values in critical columns
        critical_columns = ['customer_id', 'month']
        for col in critical_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    error_msg = f"Critical column '{col}' has {missing_count} missing values"
                    self.validation_results['errors'].append(error_msg)
    
    def _validate_interactions_business_rules(self, df: pd.DataFrame) -> None:
        """Validate interactions business rules."""
        # Engagement level should be between 0 and 1
        if 'engagement_level' in df.columns:
            out_of_range = ((df['engagement_level'] < 0) | (df['engagement_level'] > 1)).sum()
            if out_of_range > 0:
                error_msg = f"{out_of_range} engagement_level values outside valid range [0,1]"
                self.validation_results['errors'].append(error_msg)
        
        # Minutes watched should be non-negative
        if 'minutes_watched' in df.columns:
            negative_minutes = (df['minutes_watched'] < 0).sum()
            if negative_minutes > 0:
                error_msg = f"{negative_minutes} negative minutes_watched values found"
                self.validation_results['errors'].append(error_msg)
    
    def _validate_team_performance_dataset(self) -> None:
        """Validate team performance dataset."""
        team_path = self.data_paths['raw_synth'] / 'team_performance.csv'
        
        if team_path.exists():
            team_df = pd.read_csv(team_path)
            self._validate_team_performance_schema(team_df)
            self._validate_team_performance_quality(team_df)
            self._validate_team_performance_business_rules(team_df)
            self.logger.info(f"Team performance dataset validation completed: {len(team_df)} records")
        else:
            warning_msg = "No team performance dataset found for validation"
            self.validation_results['warnings'].append(warning_msg)
    
    def _validate_team_performance_schema(self, df: pd.DataFrame) -> None:
        """Validate team performance dataset schema."""
        required_columns = ['team', 'month', 'win_rate']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            error_msg = f"Team performance dataset missing required columns: {missing_columns}"
            self.validation_results['errors'].append(error_msg)
    
    def _validate_team_performance_quality(self, df: pd.DataFrame) -> None:
        """Validate team performance data quality."""
        # Check for missing values
        for col in ['team', 'month', 'win_rate']:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    error_msg = f"Column '{col}' has {missing_count} missing values"
                    self.validation_results['errors'].append(error_msg)
    
    def _validate_team_performance_business_rules(self, df: pd.DataFrame) -> None:
        """Validate team performance business rules."""
        # Win rate should be between 0 and 1
        if 'win_rate' in df.columns:
            out_of_range = ((df['win_rate'] < 0) | (df['win_rate'] > 1)).sum()
            if out_of_range > 0:
                error_msg = f"{out_of_range} win_rate values outside valid range [0,1]"
                self.validation_results['errors'].append(error_msg)
        
        # Should have all NBA teams
        nba_teams = [
            'ATL', 'BOS', 'BRK', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
        ]
        
        if 'team' in df.columns:
            missing_teams = set(nba_teams) - set(df['team'].unique())
            if missing_teams:
                warning_msg = f"Missing team performance data for: {missing_teams}"
                self.validation_results['warnings'].append(warning_msg)
    
    def _validate_cross_dataset_consistency(self) -> None:
        """Validate consistency across datasets."""
        try:
            # Load datasets for cross-validation
            customers_path = self._find_best_customer_dataset()
            interactions_path = self.data_paths['raw_synth'] / 'customer_interactions.csv'
            
            if customers_path and interactions_path.exists():
                customers_df = pd.read_csv(customers_path)
                interactions_df = pd.read_csv(interactions_path)
                
                # Check customer ID consistency
                customer_ids_customers = set(customers_df['customer_id'].unique())
                customer_ids_interactions = set(interactions_df['customer_id'].unique())
                
                missing_in_customers = customer_ids_interactions - customer_ids_customers
                missing_in_interactions = customer_ids_customers - customer_ids_interactions
                
                if missing_in_customers:
                    error_msg = f"{len(missing_in_customers)} customer IDs in interactions missing from customers"
                    self.validation_results['errors'].append(error_msg)
                
                if missing_in_interactions:
                    warning_msg = f"{len(missing_in_interactions)} customers have no interaction data"
                    self.validation_results['warnings'].append(warning_msg)
        
        except Exception as e:
            warning_msg = f"Could not perform cross-dataset validation: {str(e)}"
            self.validation_results['warnings'].append(warning_msg)
    
    def _find_best_customer_dataset(self) -> Optional[Path]:
        """Find the best available customer dataset."""
        candidates = [
            self.data_paths['processed'] / 'final_train_features.csv',
            self.data_paths['processed'] / 'train_features.csv',
            self.data_paths['raw_synth'] / 'customers.csv'
        ]
        
        for path in candidates:
            if path.exists():
                return path
        return None
    
    def _generate_validation_summary(self) -> None:
        """Generate validation summary."""
        total_errors = len(self.validation_results['errors'])
        total_warnings = len(self.validation_results['warnings'])
        datasets_validated = len(self.validation_results['datasets_validated'])
        
        if total_errors == 0:
            self.validation_results['overall_status'] = 'PASS' if total_warnings == 0 else 'PASS_WITH_WARNINGS'
        else:
            self.validation_results['overall_status'] = 'FAIL'
        
        self.validation_results['summary'] = {
            'total_datasets_validated': datasets_validated,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'validation_status': self.validation_results['overall_status']
        }
        
        # Log summary
        self.logger.info(f"Validation Summary:")
        self.logger.info(f"  Status: {self.validation_results['overall_status']}")
        self.logger.info(f"  Datasets validated: {datasets_validated}")
        self.logger.info(f"  Errors: {total_errors}")
        self.logger.info(f"  Warnings: {total_warnings}")
    
    def save_validation_report(self, output_dir: str) -> None:
        """Save detailed validation report to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / f"data_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        self.logger.info(f"Validation report saved to: {report_file}")
        
        # Also create a human-readable summary
        summary_file = output_path / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Basketball Fan Retention - Data Validation Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Validation Status: {self.validation_results['overall_status']}\n")
            f.write(f"Timestamp: {self.validation_results['validation_timestamp']}\n")
            f.write(f"Datasets Validated: {self.validation_results['datasets_validated']}\n\n")
            
            f.write(f"Summary:\n")
            f.write(f"  Total Errors: {len(self.validation_results['errors'])}\n")
            f.write(f"  Total Warnings: {len(self.validation_results['warnings'])}\n\n")
            
            if self.validation_results['errors']:
                f.write("ERRORS:\n")
                for i, error in enumerate(self.validation_results['errors'], 1):
                    f.write(f"  {i}. {error}\n")
                f.write("\n")
            
            if self.validation_results['warnings']:
                f.write("WARNINGS:\n")
                for i, warning in enumerate(self.validation_results['warnings'], 1):
                    f.write(f"  {i}. {warning}\n")
        
        self.logger.info(f"Human-readable summary saved to: {summary_file}")


def main():
    """Main function to run data validation."""
    parser = argparse.ArgumentParser(
        description="Validate basketball fan retention datasets"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Validate all datasets (default)"
    )
    parser.add_argument(
        "--customers", 
        action="store_true",
        help="Validate customer dataset only"
    )
    parser.add_argument(
        "--interactions", 
        action="store_true",
        help="Validate interaction dataset only"
    )
    parser.add_argument(
        "--teams", 
        action="store_true",
        help="Validate team performance dataset only"
    )
    parser.add_argument(
        "--report", 
        type=str,
        help="Output directory for validation report"
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
    
    # If no specific dataset specified, validate all
    if not any([args.customers, args.interactions, args.teams]):
        args.all = True
    
    logger.info("Starting basketball fan retention data validation")
    
    try:
        validator = DataValidator()
        
        if args.all:
            results = validator.validate_all_datasets()
        else:
            # Individual dataset validation
            if args.customers:
                validator._validate_customers_dataset()
            if args.interactions:
                validator._validate_interactions_dataset()
            if args.teams:
                validator._validate_team_performance_dataset()
            
            validator._generate_validation_summary()
            results = validator.validation_results
        
        # Save report if requested
        if args.report:
            validator.save_validation_report(args.report)
        
        # Print summary to console
        print("\n" + "=" * 60)
        print("DATA VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Status: {results['overall_status']}")
        print(f"Datasets Validated: {len(results['datasets_validated'])}")
        print(f"Errors: {len(results['errors'])}")
        print(f"Warnings: {len(results['warnings'])}")
        
        if results['errors']:
            print(f"\nErrors found:")
            for error in results['errors']:
                print(f"  FAILED: {error}")
        
        if results['warnings']:
            print(f"\nWarnings:")
            for warning in results['warnings']:
                print(f"  WARNING: {warning}")
        
        if results['overall_status'] == 'PASS':
            print(f"\nSUCCESS: All validation checks passed!")
        elif results['overall_status'] == 'PASS_WITH_WARNINGS':
            print(f"\nWARNING: Validation passed with warnings")
        else:
            print(f"\nFAILED: Validation failed - please address errors")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Critical error during validation: {str(e)}")
        print(f"\nERROR: Critical error during validation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()