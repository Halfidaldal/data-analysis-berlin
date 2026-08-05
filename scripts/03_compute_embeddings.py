#!/usr/bin/env python
"""
Script 03: Compute embeddings for story data.

This script:
1. Loads cleaned story data
2. Computes embeddings for full_story, full_author_1, full_author_2
3. Saves embeddings as both:
   - Parquet file (with embeddings as list columns)
   - Separate .npy files for each embedding type

Usage:
    python scripts/03_compute_embeddings.py
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.embeddings import compute_story_embeddings_full_stories, embed_story_columns
from nes.io import load_csv, save_parquet, save_npy, get_project_root, load_config, get_active_experiment, get_experiment_config, get_shared_config


def main():
    # Load config
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    embeddings_config = shared_config['embeddings']
    simulated = shared_config['cleaning'].get('simulated', False)
    
    print(f"Active experiment: {experiment}")
    
    df_full = load_csv("stories_full_text_filtered_simulated.csv" if simulated else "stories_full_text_filtered.csv", stage="interim")
    df_interactions = load_csv("interaction_level_stories_filtered_simulated.csv" if simulated else "interaction_level_stories_filtered.csv", stage="interim")
    print(f"Loaded {len(df_full)} full stories and {len(df_interactions)} interaction-level stories")
    
    # Compute embeddings using standardized column names (author_1, author_2)
    print(f"\nComputing embeddings using {embeddings_config['model_name']}...")
    df_embedded, story_emb, author_1_emb, author_2_emb = compute_story_embeddings_full_stories(
        df_full,
        model_name=embeddings_config['model_name'],
        batch_size=embeddings_config['batch_size'],
        active_dataset=experiment
    )
    
    df_embedded_interaction, embeddings_interaction_dict = embed_story_columns(
        df_interactions,
        ['author_1', 'author_2'],
        model_name=embeddings_config['model_name'],
        batch_size=embeddings_config['batch_size'],
        active_dataset=experiment
    )
    
    # Save parquet with embeddings as list columns
    print("\nSaving embeddings...")
    save_parquet(df_embedded, "story_embeddings_full_simulated.parquet" if simulated else "story_embeddings_full.parquet", stage="processed")
    save_parquet(df_embedded_interaction, "story_embeddings_interaction_level_simulated.parquet" if simulated else "story_embeddings_interaction_level.parquet", stage="processed") 
    
    # Save individual .npy files for numpy arrays
    save_npy(story_emb, "story_embeddings_full_simulated.npy" if simulated else "story_embeddings_full.npy",  stage="processed")
    save_npy(author_1_emb, "story_author_1_embeddings_full_simulated.npy" if simulated else "story_author_1_embeddings_full.npy", stage="processed")
    save_npy(author_2_emb, "story_author_2_embeddings_full_simulated.npy" if simulated else "story_author_2_embeddings_full.npy", stage="processed")
        
    print(f"\n✓ Computed embeddings for {len(df_embedded)} stories")
    print(f"✓ Embedding dimension: {story_emb.shape[1]}")
    print(f"✓ Saved to {exp_config['processed_dir']}/")
    print("\n✅ Script 03 complete!")


if __name__ == "__main__":
    main()
