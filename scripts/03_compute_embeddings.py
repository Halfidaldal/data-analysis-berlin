#!/usr/bin/env python
"""
Script 03: Compute embeddings.

Three outputs, all with the encoder named in config.yaml (shared.embeddings):

  story_embeddings_full.parquet             per story: whole text and each side
  story_embeddings_interaction_level.parquet per turn-pair: each side separately
  analysis_story_embeddings.parquet          per story, ANALYSIS SET ONLY

The third is the one downstream code should use for anything keyed to condition.
The first two are computed over the whole interim table, which still contains
non-German stories, instrument tests, and stories the analysis set excludes; the
third applies the selection in src/nes/berlin_pov.py first and carries
conversation_id and condition on every row, so vectors cannot be misaligned to
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
from nes.berlin_pov import author_labelled_stream, build_frames
from nes.io import get_file_suffix, load_csv, save_parquet, save_npy, get_project_root, load_config, get_active_experiment, get_experiment_config, get_shared_config


def embed_analysis_set(embeddings_config, experiment, language="de"):
    """Per-story embeddings over the analysis set, keyed by conversation_id."""
    from nes.embeddings import compute_embeddings_batch

    frames = build_frames(language=language)
    stream = author_labelled_stream(frames.turns)

    def joined(df):
        return (
            df.groupby(["conversation_id", "condition"], sort=False)["text"]
            .apply(" ".join)
            .reset_index()
        )

    story = joined(stream).rename(columns={"text": "full_story"})
    for side, who in (("full_user", "human"), ("full_model", "model")):
        part = joined(stream[stream["author"] == who]).rename(columns={"text": side})
        story = story.merge(
            part[["conversation_id", side]], on="conversation_id", how="left"
        )
        story[side] = story[side].fillna("")

    for col in ("full_story", "full_user", "full_model"):
        print(f"  encoding {col} ({len(story)} stories) ...")
        vecs = compute_embeddings_batch(
            story[col].tolist(),
            model_name=embeddings_config["model_name"],
            batch_size=embeddings_config["batch_size"],
            active_dataset=experiment,
        )
        story[f"{col}_embedding"] = list(vecs)

    suffix = "" if language == "de" else f"_{language}"
    save_parquet(story, f"analysis_story_embeddings{suffix}.parquet", stage="processed")
    print(f"  -> analysis_story_embeddings{suffix}.parquet ({len(story)} stories)")


def main():
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
        
    # ---- analysis-set per-story embeddings ------------------------------
    # Built from the author-labelled stream so the visitor and model sides are
    # exactly the text the driver analyses parse, and so every vector carries
    # its conversation_id. scripts/10_condition_classifier.py reads this.
    embed_analysis_set(embeddings_config, experiment)

    print(f"\n✓ Computed embeddings for {len(df_embedded)} stories")
    print(f"✓ Embedding dimension: {story_emb.shape[1]}")
    print(f"✓ Saved to {exp_config['processed_dir']}/")
    print("\n✅ Script 03 complete!")


if __name__ == "__main__":
    main()
