#!/usr/bin/env python3
"""
Script 08: Compute semantic exploration metrics (binned embedding distances).

Computes non-overlapping semantic jumps at multiple timescales to measure
how much narratives explore semantic space over time.

Input:  data/<experiment>/processed/story_embeddings_interaction_level.parquet
Output: data/<experiment>/processed/semantic_exploration_binned.parquet
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.semantic_exploration import (
    compute_lag_exploration_metrics
)
from nes.io import (
    backfill_interaction_metadata,
    get_active_experiment,
    get_experiment_config,
    get_shared_config,
    load_parquet,
    save_parquet,
)


def main():
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    
    exploration_config = shared_config.get('exploration', {})
    simulated = shared_config['cleaning'].get('simulated', False)

    print(f"Active experiment: {experiment}")
    
    # Determine author_2 embedding column name (standardized)
    author_2_emb_col = 'author_2_embedding'

    print("Computing semantic exploration for interactions...")
    print(f"Loading embeddings from {exp_config['processed_dir']}/")
    df_interaction_level = load_parquet("story_embeddings_interaction_level_simulated.parquet" if simulated else "story_embeddings_interaction_level.parquet", stage="processed")
    df_interaction_level = backfill_interaction_metadata(df_interaction_level, simulated=simulated)
    
    try:
        if 'analysis_turn' in df_interaction_level.columns and df_interaction_level['analysis_turn'].notna().any():
            analysis_turn_min = exploration_config.get('analysis_turn_min', 1)
            analysis_turn_max = exploration_config.get('analysis_turn_max', 9)
            mask = df_interaction_level['analysis_turn'].between(analysis_turn_min, analysis_turn_max, inclusive='both')
        else:
            raw_turn_min = exploration_config.get('turn_min', 2)
            raw_turn_max = exploration_config.get('turn_max', 9)
            mask = df_interaction_level['turn'].between(raw_turn_min, raw_turn_max, inclusive='both')

        if 'complete_exchange' in df_interaction_level.columns:
            mask &= df_interaction_level['complete_exchange'].fillna(False)

        df_filtered = df_interaction_level[mask].copy()
        print(f"Retained {len(df_filtered)} interaction rows for semantic exploration")
        
        # Compute exploration metrics using standardized column names
        exploration_df = compute_lag_exploration_metrics(
            df_filtered,
            user_embedding_col="author_1_embedding",
            ai_embedding_col=author_2_emb_col,
            max_lag=exploration_config.get('max_k', 10)
        )
        
        # Save results
        output_file = "semantic_exploration_binned_simulated.parquet" if simulated else "semantic_exploration_binned.parquet"
        save_parquet(exploration_df, output_file, stage="processed")
        print(f"Saved {len(exploration_df)} exploration records to {output_file}")
        
        # Print summary
        mean_by_agent_k = exploration_df.groupby(["agent", "k"])["distance"].mean()
        print("\nMean semantic jump by window size and agent:")
        for (agent, k), dist in mean_by_agent_k.items():
            print(f"  agent={agent}, k={k} (window={k+1}): {dist:.4f}")
        
    except Exception as e:
        print(f"Error processing embeddings: {e}")
        print("Make sure embeddings are saved as parquet to preserve array columns.")
    
    print(f"\n✓ Saved to {exp_config['processed_dir']}/")
    print("\n✅ Script 08 complete!")


if __name__ == "__main__":
    main()
