#!/usr/bin/env python3
"""
Basketball Fan Retention Project Setup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Setup the basketball fan retention project environment."""
    print("Setting up Basketball Fan Retention & Revenue Optimization Project")
    print("=" * 70)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"PASSED: Python {sys.version} detected")
    
    # Create data directories
    data_dirs = [
        "data/raw/api",
        "data/raw/bbref", 
        "data/raw/synth",
        "data/processed",
        "data/models",
        "reports"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep files
        gitkeep_file = Path(dir_path) / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    print("PASSED: Data directories created")
    
    # Install requirements
    try:
        print("INSTALLING: Installing requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("PASSED: Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install requirements")
        sys.exit(1)
    
    # Run setup command
    try:
        print("SETUP: Setting up project in development mode...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], check=True)
        print("PASSED: Project setup complete")
    except subprocess.CalledProcessError:
        print("ERROR: Failed to setup project")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("COMPLETE: Setup Complete!")
    print("\nNext steps:")
    print("1. Copy config/config.example.yaml to config/config.yaml")
    print("2. Update config.yaml with your API keys")
    print("3. Run: make synthetic-data")
    print("4. Run: make run-notebooks")
    print("\nFor help, run: make help")

if __name__ == "__main__":
    main()
