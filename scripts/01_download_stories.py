#!/usr/bin/env python
"""
Script 01: Download raw story data from Firestore.

This script:
1. Downloads stories from Firestore using the experiment's document schema
2. Applies the completeness filter, where the schema defines one
3. Saves raw data to data/<experiment>/raw/

The berlin schema applies no completeness filter: sessions end whenever the
visitor walks away, so a short conversation is data rather than a fragment.
Selection happens in nes.berlin_pov, where it is explicit and reported.

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
    
    # Get Firestore settings for this experiment
    firestore_config = exp_config['firestore']

    # Initialize Firestore. The service-account file is optional; without it the
    # client authenticates with Application Default Credentials.
    print("Initializing Firestore client...")
    configured = shared_config.get('firestore', {}).get('credentials_path')
    credentials_path = str(get_project_root() / configured) if configured else None
    db = init_firestore(
        credentials_path=credentials_path,
        project_id=firestore_config.get('project_id'),
    )
    
    # Download stories. `min_interactions` is a completeness filter used by the
    # fixed-length corpora; the berlin schema does not take one and does not set
    # the key, because selection there happens later in nes.berlin_pov.
    schema = firestore_config['schema']
    print(f"\nDownloading stories from collection: {firestore_config['collection_name']}")
    print(f"Schema: {schema}")

    kwargs = {}
    if 'min_interactions' in firestore_config:
        kwargs['min_interactions'] = firestore_config['min_interactions']
        print(f"Min interactions: {firestore_config['min_interactions']}")
    for key in ('collection_start', 'collection_end', 'timezone'):
        if key in firestore_config:
            kwargs[key] = firestore_config[key]
    if 'collection_start' in kwargs or 'collection_end' in kwargs:
        print(f"Collection window: {kwargs.get('collection_start')} .. "
              f"{kwargs.get('collection_end')} ({kwargs.get('timezone', 'UTC')})")

    df_stories = download_stories_from_firestore(
        db,
        collection_name=firestore_config['collection_name'],
        schema=schema,
        **kwargs,
    )
    
    # Save to raw data
    output_filename = "finished_stories_raw.csv"
    save_csv(df_stories, output_filename, stage="raw")
    
    print(f"\n✓ Downloaded {len(df_stories)} interaction rows")
    print(f"✓ Saved to {exp_config['raw_dir']}/{output_filename}")
    
    print("\n✅ Script 01 complete!")


if __name__ == "__main__":
    main()
