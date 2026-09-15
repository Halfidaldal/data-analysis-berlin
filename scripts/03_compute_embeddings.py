#!/usr/bin/env python
"""
Script 03: Compute embeddings.

Three outputs, all with the encoder named in config.yaml (shared.embeddings):

  story_embeddings_full.parquet              per story: whole text and each side
  story_embeddings_interaction_level.parquet per turn-pair: each side separately

Both are restricted to the analysis set for `--language` (German by default, the
52-story English subset with --language en), so every stage of the pipeline
carries the same n. Script 02 deliberately keeps more than this -- both languages
and the abandoned sessions -- because the abandonment analysis needs them; the
narrowing happens here and is the same `nes.berlin_pov` selection every later
stage uses.

Each row carries conversation_id and workshop_id, so vectors are never matched to
conditions by position.

Turns are encoded plain, with no instruction or query prefix. The configured
encoder offers a retrieval prompt format; this corpus needs symmetric
turn-to-turn similarity, and a prefix would change what the cosine distances in
scripts/09_build_metric_tables.py mean.

Usage:
    PYTHONPATH=src python scripts/03_compute_embeddings.py
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.embeddings import compute_story_embeddings_full_stories, embed_story_columns
from nes.cleaning import normalize_columns
from nes.berlin_pov import ANALYSIS_LANGUAGE, analysis_set_ids, author_labelled_stream, build_frames, language_suffix
from nes.io import get_file_suffix, load_csv, save_parquet, save_npy, get_project_root, load_config, get_active_experiment, get_experiment_config, get_shared_config



def main(language: str = ANALYSIS_LANGUAGE):
    # Load config
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    embeddings_config = shared_config['embeddings']
    simulated = shared_config['cleaning'].get('simulated', False)
    
    print(f"Active experiment: {experiment}")
    
    suffix = get_file_suffix()
    df_full = load_csv(f"stories_full_text_filtered{suffix}.csv", stage="interim")
    df_interactions = load_csv(f"interaction_level_stories_filtered{suffix}.csv", stage="interim")
    # Berlin has a fixed agent per slot, so normalize user/ai to author_1/author_2
    # before anything downstream addresses the columns.
    df_full = normalize_columns(df_full, experiment)
    df_interactions = normalize_columns(df_interactions, experiment)
    print(f"Loaded {len(df_full)} full stories and {len(df_interactions)} interaction-level stories")

    # Restrict to the analysis set so every stage of the pipeline carries the
    # same n. Script 02 keeps both languages and the abandoned sessions, because
    # the abandonment analysis needs them; from here on the corpus is the
    # analysis set for `language`.
    keep = analysis_set_ids(language)
    df_full = df_full[df_full["conversation_id"].isin(keep)].reset_index(drop=True)
    df_interactions = df_interactions[df_interactions["conversation_id"].isin(keep)].reset_index(drop=True)
    lang_sfx = language_suffix(language)
    print(f"Analysis set (language={language}): {len(df_full)} stories, "
          f"{len(df_interactions)} interaction rows")
    
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
    save_parquet(df_embedded, f"story_embeddings_full{'_simulated' if simulated else lang_sfx}.parquet", stage="processed")
    save_parquet(df_embedded_interaction, f"story_embeddings_interaction_level{'_simulated' if simulated else lang_sfx}.parquet", stage="processed") 
    
    # Save individual .npy files for numpy arrays
    save_npy(story_emb, f"story_embeddings_full{'_simulated' if simulated else lang_sfx}.npy",  stage="processed")
    save_npy(author_1_emb, f"story_author_1_embeddings_full{'_simulated' if simulated else lang_sfx}.npy", stage="processed")
    save_npy(author_2_emb, f"story_author_2_embeddings_full{'_simulated' if simulated else lang_sfx}.npy", stage="processed")
        
    print(f"\n✓ Computed embeddings for {len(df_embedded)} stories")
    print(f"✓ Embedding dimension: {story_emb.shape[1]}")
    print(f"✓ Saved to {exp_config['processed_dir']}/")
    print("\n✅ Script 03 complete!")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Compute embeddings for the analysis set.")
    _ap.add_argument("--language", default=ANALYSIS_LANGUAGE, choices=["de", "en"],
                     help="'de' is the primary analysis set; 'en' is the validation subset")
    _args = _ap.parse_args()
    main(language=_args.language)
