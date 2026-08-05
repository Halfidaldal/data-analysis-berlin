#!/usr/bin/env python
"""
Script 04: Compute sentiment scores via semantic projection.

Two parallel projection-based sentiment signals are computed on the
interaction-level story data, both using the same encoder + concept vector
configured under shared.sentiment.projection_*:

1. Legacy turn-alone baseline (sanity check):
   compute_semantic_projection_batch on each author_1 / author_2 text in
   isolation.
   -> author_1_sentiment_projection, author_2_sentiment_projection

2. Windowed contextual marginal (primary):
   For each conversation, interleave author_1 / author_2 turns
   chronologically. For each position, project the last
   `projection_context_window` turns of context with and without the current
   turn appended, and take the difference. This bounded-context marginal
   answers "given the partner's last utterance, how much did this turn shift
   the sentiment axis?" without the magnitude collapse that an unbounded
   cumulative prefix produces with L2-normalized embeddings. We also emit a
   within-conversation z-scored variant (pooled across both author slots)
   for fair cross-condition mean comparisons.
   -> author_{1,2}_sentiment_marginal_window
   -> author_{1,2}_sentiment_marginal_window_z

Because (1) and (2) share encoder + concept vector, the only methodological
difference between the two outputs is the bounded contextual encoding -- so
the legacy column functions as a tight non-contextual sanity check on the
windowed result.

Usage:
    python scripts/04_compute_sentiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.sentiment import (
    compute_semantic_projection_batch,
    compute_dyadic_windowed_projection,
)
from nes.io import (
    backfill_interaction_metadata,
    get_active_experiment,
    get_experiment_config,
    get_shared_config,
    get_project_root,
    load_parquet,
    save_parquet,
)


def main():
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    sentiment_config = shared_config['sentiment']
    simulated = shared_config['cleaning'].get('simulated', False)

    print(f"Active experiment: {experiment}")

    print("Loading story data with embeddings...")
    df_interaction_level = load_parquet(
        "story_embeddings_interaction_level_simulated.parquet" if simulated
        else "story_embeddings_interaction_level.parquet",
        stage="processed",
    )
    df_interaction_level = backfill_interaction_metadata(
        df_interaction_level, simulated=simulated
    )
    print(f"Loaded {len(df_interaction_level)} interaction rows")
    df_dyadic = df_interaction_level.copy()

    project_root = get_project_root()
    projection_model = sentiment_config.get("projection_model_name")
    projection_vector_rel = sentiment_config.get("projection_vector_path")
    if not (projection_model and projection_vector_rel):
        raise ValueError(
            "Missing shared.sentiment.projection_model_name and/or "
            "projection_vector_path in config.yaml."
        )
    projection_vector_path = project_root / projection_vector_rel
    if not projection_vector_path.exists():
        raise FileNotFoundError(
            f"Concept vector not found at {projection_vector_path}. "
            "Either ship the file or run scripts/04b_build_concept_vector.py."
        )

    legacy_batch_size = int(sentiment_config.get("batch_size", 32))
    projection_batch_size = int(sentiment_config.get("projection_batch_size", 32))
    projection_task = sentiment_config.get("projection_task")
    context_window = int(sentiment_config.get("projection_context_window", 1))
    separator = sentiment_config.get("projection_context_separator", "\n")

    # ------------------------------------------------------------------
    # 1. Legacy turn-alone projection (sanity check).
    # ------------------------------------------------------------------
    print(
        f"\n[Legacy] Computing per-turn projection (turn embedded alone) with "
        f"{projection_model} (batch_size={legacy_batch_size})..."
    )
    print("[Legacy] Projecting author_1 turns...")
    df_dyadic['author_1_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['author_1'].tolist(),
        model_name=projection_model,
        vector_path=str(projection_vector_path),
        batch_size=legacy_batch_size,
    )
    print("[Legacy] Projecting author_2 turns...")
    df_dyadic['author_2_sentiment_projection'] = compute_semantic_projection_batch(
        df_dyadic['author_2'].tolist(),
        model_name=projection_model,
        vector_path=str(projection_vector_path),
        batch_size=legacy_batch_size,
    )

    # ------------------------------------------------------------------
    # 2. Windowed contextual marginal projection (primary).
    # ------------------------------------------------------------------
    print(
        f"\n[Contextual] Computing windowed marginal projection "
        f"(context_window={context_window}) with "
        f"{projection_model} (batch_size={projection_batch_size})..."
    )
    df_dyadic = compute_dyadic_windowed_projection(
        df_dyadic,
        model_name=projection_model,
        concept_vector_path=str(projection_vector_path),
        context_window=context_window,
        batch_size=projection_batch_size,
        task=projection_task,
        separator=separator,
    )

    output_file = (
        "dyadic_sentiment_scores_simulated.parquet" if simulated
        else "dyadic_sentiment_scores.parquet"
    )
    save_parquet(df_dyadic, output_file, stage="processed")
    print(f"\n✓ Saved sentiment for {len(df_dyadic)} turns to {output_file}")

    print("\n✅ Script 04 complete!")


if __name__ == "__main__":
    main()
