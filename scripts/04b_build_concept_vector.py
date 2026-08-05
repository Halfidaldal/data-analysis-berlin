#!/usr/bin/env python
"""
Script 04b: Build a sentiment concept vector for the context-aware projection.

Faithfully replicates the CHC procedure from
https://github.com/centre-for-humanities-computing/embedding-projection
(v = mean(positive_anchor_embeddings) - mean(negative_anchor_embeddings))
using the published Fiction4Sentiment anchor texts, but with whichever
encoder is configured under shared.sentiment.projection_model_name.

This only needs to be re-run when the encoder or anchor set changes.

Usage:
    python scripts/04b_build_concept_vector.py [--overwrite]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.sentiment import build_concept_vector_fiction4
from nes.io import get_shared_config, get_project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute the concept vector even if the output file already exists.",
    )
    args = parser.parse_args()

    shared_config = get_shared_config()
    sentiment_config = shared_config["sentiment"]

    model_name = sentiment_config.get("projection_model_name")
    if not model_name:
        raise ValueError(
            "Missing shared.sentiment.projection_model_name in config.yaml; "
            "set it to the encoder used for the contextual projection."
        )

    project_root = get_project_root()
    output_path = project_root / sentiment_config["projection_vector_path"]
    anchor_path = project_root / sentiment_config.get(
        "projection_anchor_text_path",
        "src/nes/data/Fiction4Sentiment_concept_text.csv",
    )
    batch_size = int(sentiment_config.get("projection_build_batch_size", 8))
    task = sentiment_config.get("projection_task")

    print(f"Encoder:       {model_name}")
    print(f"Anchor texts:  {anchor_path}")
    print(f"Output vector: {output_path}")

    build_concept_vector_fiction4(
        model_name=model_name,
        output_vector_path=str(output_path),
        concept_text_path=str(anchor_path),
        batch_size=batch_size,
        task=task,
        overwrite=args.overwrite,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
