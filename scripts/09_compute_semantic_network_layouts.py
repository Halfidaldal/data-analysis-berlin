#!/usr/bin/env python3
"""
Script 09: Compute semantic network layouts for comparison plots.

This script is intentionally a thin CLI wrapper. The computational functions
live in src/nes/semantic_network_layouts.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.semantic_network_layouts import compute_semantic_network_layouts  # noqa: E402


def parse_conditions(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute semantic network 2D layouts.")
    parser.add_argument("--conditions", default="human-ai,human-human,ai-ai")
    parser.add_argument("--method", choices=["pca", "umap", "auto"], default="pca")
    parser.add_argument("--turn-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analysis-turn-min", type=int, default=1)
    parser.add_argument("--analysis-turn-max", type=int, default=9)
    parser.add_argument(
        "--no-balance-stories",
        action="store_true",
        help="Use all available stories instead of sampling the same number per condition.",
    )
    parser.add_argument(
        "--stories-per-condition",
        type=int,
        default=None,
        help="Number of stories to sample per condition. Defaults to the smallest condition count.",
    )
    parser.add_argument(
        "--no-balance-edges",
        action="store_true",
        help="Use all adjacent edges from the story-balanced sample instead of equalizing edge counts.",
    )
    parser.add_argument(
        "--edges-per-condition",
        type=int,
        default=None,
        help="Number of adjacent edges to sample per condition. Defaults to the smallest condition count.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/comparison/semantic_network_layouts",
        help="Output directory relative to the project root.",
    )
    args = parser.parse_args()

    metadata = compute_semantic_network_layouts(
        conditions=parse_conditions(args.conditions),
        method=args.method,
        turn_neighbors=args.turn_neighbors,
        min_dist=args.min_dist,
        seed=args.seed,
        analysis_turn_min=args.analysis_turn_min,
        analysis_turn_max=args.analysis_turn_max,
        output_dir=args.output_dir,
        balance_stories=not args.no_balance_stories,
        stories_per_condition=args.stories_per_condition,
        balance_edges=not args.no_balance_edges,
        edges_per_condition=args.edges_per_condition,
    )

    print(f"Saved semantic network layouts to {args.output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
