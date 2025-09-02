"""
Configuration utilities for the basketball fan retention project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = str(project_root / "config" / "config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_paths() -> Dict[str, Path]:
    """
    Get standardized data paths.
    
    Returns:
        Dictionary of data paths.
    """
    project_root = Path(__file__).parent.parent
    
    paths = {
        'raw_api': project_root / "data" / "raw" / "api",
        'raw_bbref': project_root / "data" / "raw" / "bbref", 
        'raw_synth': project_root / "data" / "raw" / "synth",
        'processed_models': project_root / "data" / "processed" / "models",
        'processed_figures': project_root / "data" / "processed" / "figures",
        'processed_ltv': project_root / "data" / "processed" / "ltv",
        'processed_assignments': project_root / "data" / "processed" / "assignments"
    }
    
    # Ensure directories exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def setup_logging():
    """
    Setup logging configuration.
    """
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('basketball_analytics.log')
        ]
    )
    
    return logging.getLogger(__name__)
