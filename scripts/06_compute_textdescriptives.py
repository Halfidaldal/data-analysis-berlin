#!/usr/bin/env python
"""
Script 06: Compute text descriptives for stories.

Usage:
    python scripts/06_compute_textdescriptives.py
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.io import load_csv, save_parquet, get_active_experiment, get_experiment_config, get_shared_config
from nes.surface_metrics import get_descriptive_metrics_dual_full_long, get_descriptive_metrics_dual_inter_long


def main():
    parser = argparse.ArgumentParser(description="Compute text descriptives for stories")
    parser.add_argument("--full", default=None, help="Path to full stories CSV")
    parser.add_argument("--interaction", default=None, help="Path to interaction level stories CSV")
    args = parser.parse_args()
    
    # Load config
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    simulated = shared_config['cleaning'].get('simulated', False)
    
    print(f"Active experiment: {experiment}")

    full_input = args.full or ("stories_full_text_filtered_simulated.csv" if simulated else "stories_full_text_filtered.csv")
    interaction_input = args.interaction or ("interaction_level_stories_filtered_simulated.csv" if simulated else "interaction_level_stories_filtered.csv")
    full_output = "full_story_surface_metrics_simulated.parquet" if simulated else "full_story_surface_metrics.parquet"
    interaction_output = "interaction_level_surface_metrics_simulated.parquet" if simulated else "interaction_level_surface_metrics.parquet"
    
    df_full = load_csv(full_input, stage="interim")
    print(f"Loaded {len(df_full)} full stories")
    df_inter = load_csv(interaction_input, stage="interim")
    print(f"Loaded {len(df_inter)} interaction level stories")
    
    spacy_mdl = shared_config['surface_metrics']['spacy_mdl']
    batch_size = shared_config['surface_metrics']['batch_size']
    n_process = shared_config['surface_metrics']['n_process']

    print(f"Computing Text Descriptives for: {experiment}")

    df_descriptives_full = get_descriptive_metrics_dual_full_long(
        df_full,
        spacy_mdl=spacy_mdl, 
        batch_size=batch_size,
        n_process=n_process
    )
    save_parquet(df=df_descriptives_full, filename=full_output)
    print('\nFinished computing for full stories\n')

    df_descriptives_inter = get_descriptive_metrics_dual_inter_long(
        df_inter,
        spacy_mdl=spacy_mdl, 
        batch_size=batch_size,
        n_process=n_process
    )
    print('\nFinished computing for interaction level stories\n')

    save_parquet(df=df_descriptives_inter, filename=interaction_output)
    
    print(f"✓ Saved to {exp_config['processed_dir']}/")
    print("\n✅ Script 06 complete!")


if __name__ == "__main__":
    main()
