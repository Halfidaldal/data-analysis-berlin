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

from nes.berlin_pov import ANALYSIS_LANGUAGE, analysis_set_ids, language_suffix
from nes.cleaning import normalize_columns
from nes.io import load_csv, save_parquet, get_active_experiment, get_experiment_config, get_shared_config, get_file_suffix
from nes.surface_metrics import get_descriptive_metrics_dual_full_long, get_descriptive_metrics_dual_inter_long


def main():
    parser = argparse.ArgumentParser(description="Compute text descriptives for stories")
    parser.add_argument("--full", default=None, help="Path to full stories CSV")
    parser.add_argument("--interaction", default=None, help="Path to interaction level stories CSV")
    parser.add_argument("--language", default=ANALYSIS_LANGUAGE, choices=["de", "en"],
                        help="'de' is the primary analysis set; 'en' is the validation subset")
    args = parser.parse_args()
    
    # Load config
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    simulated = shared_config['cleaning'].get('simulated', False)
    
    print(f"Active experiment: {experiment}")

    suffix = get_file_suffix()
    full_input = args.full or f"stories_full_text_filtered{suffix}.csv"
    interaction_input = args.interaction or f"interaction_level_stories_filtered{suffix}.csv"
    lang_sfx = language_suffix(args.language)
    full_output = f"full_story_surface_metrics{'_simulated' if simulated else lang_sfx}.parquet"
    interaction_output = f"interaction_level_surface_metrics{'_simulated' if simulated else lang_sfx}.parquet"

    df_full = load_csv(full_input, stage="interim")
    df_inter = load_csv(interaction_input, stage="interim")

    # Same analysis set as every other stage; see scripts/03 for why script 02
    # deliberately keeps more than this.
    keep = analysis_set_ids(args.language)
    df_full = df_full[df_full["conversation_id"].isin(keep)].reset_index(drop=True)
    df_inter = df_inter[df_inter["conversation_id"].isin(keep)].reset_index(drop=True)
    print(f"Analysis set (language={args.language}): {len(df_full)} stories, "
          f"{len(df_inter)} interaction rows")
    
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
