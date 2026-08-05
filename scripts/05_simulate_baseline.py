#!/usr/bin/env python
"""
Script 05: Simulate AI-AI baseline conversations.

This script generates a dataset of AI-AI collaborative stories for the ai-ai 
experimental condition. It matches the actual PENPAL human-AI experiment:
- Temperature: 1.0
- Max tokens: 35
- System prompt: Collaborative storytelling with pacing info

Models configured in config.yaml:
- gpt-4o (OpenAI)
- claude-3-5-sonnet-20241022 (Anthropic)
- Llama-3.3-70B-Instruct (OpenRouter)
- Qwen2.5-72B-Instruct (OpenRouter)

Usage:
    # API keys can be in .env file or environment:
    # OPENAI_API_KEY=your_key
    # ANTHROPIC_API_KEY=your_key
    # OPENROUTER_API_KEY=your_key
    
    # Run simulation (saves to data/ai-ai/raw/):
    python scripts/05_simulate_baseline.py
    
    # Run with specific models only:
    python scripts/05_simulate_baseline.py --models gpt-4o claude-sonnet
    
    # Customize story count:
    python scripts/05_simulate_baseline.py --n-stories 5 --n-turns 8
"""

import sys
from pathlib import Path
import argparse
import os

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.simulation import simulate_ai_ai_dataset
from nes.io import save_csv, get_project_root, load_config, get_experiment_config


def main():
    parser = argparse.ArgumentParser(
        description="Simulate AI-AI collaborative stories for baseline analysis"
    )
    parser.add_argument(
        "--n-stories",
        type=int,
        default=None,
        help="Number of stories per model (default: from config, usually 10)"
    )
    parser.add_argument(
        "--n-turns",
        type=int,
        default=None,
        help="Number of turns per story (default: from config, usually 10)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: from config, 1.0 to match experiment)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens (default: from config, 35 to match experiment)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between stories (default: 2.0)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model IDs to run (default: all configured models)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running simulation"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Script 05: AI-AI Baseline Simulation")
    print("=" * 60)
    
    # Force active experiment to ai-ai for this script
    config = load_config()
    
    # Get ai-ai experiment config
    if 'ai-ai' not in config.get('experiments', {}):
        print("❌ Error: ai-ai experiment not configured in config.yaml")
        sys.exit(1)
    
    ai_ai_config = config['experiments']['ai-ai']
    sim_config = ai_ai_config.get('simulation', {})
    
    # Get parameters (with experiment-matching defaults)
    n_stories = args.n_stories or sim_config.get('n_stories_per_model', 10)
    n_turns = args.n_turns or sim_config.get('n_turns_per_story', 10)
    temperature = args.temperature or sim_config.get('temperature', 1.0)
    max_tokens = args.max_tokens or sim_config.get('max_tokens', 40)
    model_configs = sim_config.get('models', [])
    
    if not model_configs:
        print("❌ Error: No models configured in config.yaml")
        sys.exit(1)
    
    # Filter models if specified
    if args.models:
        model_configs = [m for m in model_configs if m['id'] in args.models]
        if not model_configs:
            print(f"❌ Error: No matching models found for: {args.models}")
            sys.exit(1)
    
    # Check API keys
    print("\nAPI Key Status:")
    missing_keys = []
    for model in model_configs:
        env_key = model['env_key']
        if os.environ.get(env_key):
            print(f"  ✓ {env_key}")
        else:
            print(f"  ✗ {env_key} (not set)")
            missing_keys.append(env_key)
    
    if missing_keys:
        print(f"\n⚠️  Warning: Some API keys not set. Those models will be skipped.")
    
    # Print configuration
    print(f"\nConfiguration (matching PENPAL experiment):")
    print(f"  Stories per model: {n_stories}")
    print(f"  Turns per story: {n_turns}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Models ({len(model_configs)}):")
    for m in model_configs:
        print(f"    - {m['id']}: {m['model_name']} ({m['provider']})")
    print(f"  Total stories: {n_stories * len(model_configs)}")
    print(f"  Total turns: {n_stories * len(model_configs) * n_turns}")
    print(f"  Delay between stories: {args.delay}s")
    print(f"  Rate limit handling: exponential backoff retry")
    
    if args.dry_run:
        print("\n[Dry run - no simulation performed]")
        return
    
    # Run simulation
    print("\n" + "=" * 60)
    print("Starting round-robin simulation...")
    print("=" * 60)
    
    df_simulated = simulate_ai_ai_dataset(
        model_configs=model_configs,
        n_stories_per_model=n_stories,
        n_turns_per_story=n_turns,
        temperature=temperature,
        max_tokens=max_tokens,
        delay_between_stories=args.delay
    )
    
    if df_simulated.empty:
        print("\n❌ No data generated. Check API keys and try again.")
        sys.exit(1)
    
    # Save to ai-ai raw directory
    raw_dir = Path(get_project_root()) / ai_ai_config['raw_dir']
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_dir / "simulated_stories.csv"
    df_simulated.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Simulation Summary")
    print("=" * 60)
    print(f"  Total rows: {len(df_simulated)}")
    print(f"  Unique stories: {df_simulated['story_id'].nunique()}")
    print(f"  By model:")
    for model_id, group in df_simulated.groupby('model_id'):
        n_stories_model = group['story_id'].nunique()
        print(f"    - {model_id}: {n_stories_model} stories, {len(group)} turns")
    
    print(f"\n✓ Saved to: {output_path}")
    print("\n✅ Script 05 complete!")
    print("\nNext steps:")
    print("  1. Set active_experiment to 'ai-ai' in config.yaml")
    print("  2. Run scripts 02-08 to process the simulated data")


if __name__ == "__main__":
    main()
