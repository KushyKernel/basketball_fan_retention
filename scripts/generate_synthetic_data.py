#!/usr/bin/env python3
"""
Generate realistic synthetic basketball fan data.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from synthetic_data import UltraRealisticSyntheticDataGenerator
from config import get_data_paths, load_config


def main():
    """Generate realistic synthetic data and save to files."""
    print("Generating realistic synthetic basketball fan data...")
    
    # Initialize generator
    generator = UltraRealisticSyntheticDataGenerator(random_seed=42)
    
    # Generate all datasets
    data = generator.generate_all_data()
    customers = data['customers']
    interactions = data['interactions']
    team_performance = data['team_performance']
    
    # Get output paths
    data_paths = get_data_paths()
    synth_path = data_paths['raw_synth']
    synth_path.mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    print("Saving datasets...")
    customers.to_csv(synth_path / 'customers.csv', index=False)
    interactions.to_csv(synth_path / 'customer_interactions.csv', index=False)
    team_performance.to_csv(synth_path / 'team_performance.csv', index=False)
    
    # Create simple summary statistics
    summary_stats = {
        'generation_date': datetime.now().isoformat(),
        'num_customers': len(customers),
        'num_interactions': len(interactions),
        'num_teams': len(team_performance),
        'data_type': 'realistic_synthetic'
    }
    
    # Save summary statistics
    with open(synth_path / 'summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Realistic synthetic data generated successfully!")
    print(f"Files saved to: {synth_path}")
    print(f"{len(customers):,} customers")
    print(f"{len(interactions):,} interaction records")
    print(f"{len(team_performance):,} team performance records")


if __name__ == "__main__":
    main()
