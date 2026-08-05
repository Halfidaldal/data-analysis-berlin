#!/usr/bin/env python
"""
Script 01: Download and clean raw story data from Firestore.

This script:
1. Downloads stories from Firestore
2. Filters to keep only complete stories (≥min_interactions)
3. Saves raw data to data/<experiment>/raw/

Usage:
    python scripts/01_download_stories.py
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.cleaning import init_firestore, download_stories_from_firestore
from nes.io import save_csv, get_project_root, load_config, get_active_experiment, get_experiment_config, get_shared_config


def main():
    
    # Load config
    config = load_config()
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    
    print(f"Active experiment: {experiment}")
    
    # Initialize Firestore
    print("Initializing Firestore client...")
    credentials_path = get_project_root() / shared_config['firestore']['credentials_path']
    db = init_firestore(str(credentials_path))
    
    # Get Firestore settings for this experiment
    firestore_config = exp_config['firestore']
    
    # Download stories
    print(f"\nDownloading stories from collection: {firestore_config['collection_name']}")
    print(f"Min interactions: {firestore_config['min_interactions']}")
    print(f"Schema: {firestore_config['schema']}")
    
    df_stories = download_stories_from_firestore(
        db,
        collection_name=firestore_config['collection_name'],
        min_interactions=firestore_config['min_interactions'],
        schema=firestore_config['schema']
    )
    
    # Save to raw data
    output_filename = "finished_stories_raw.csv"
    save_csv(df_stories, output_filename, stage="raw")
    
    print(f"\n✓ Downloaded {len(df_stories)} interaction rows")
    print(f"✓ Saved to {exp_config['raw_dir']}/{output_filename}")
    
    print("\n✅ Script 01 complete!")


if __name__ == "__main__":
    main()
