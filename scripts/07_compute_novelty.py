#!/usr/bin/env python3
"""
Script 07: Compute novelty and transience scores.

Uses a causal language model (Llama-8B) to compute:
- Novelty (surprise): how surprising this utterance is given prior context
- Transience: how surprising future text is given this utterance

Input:  data/<experiment>/interim/interaction_level_stories_filtered.csv
Output: data/<experiment>/processed/novelty_scores.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.io import load_config, load_csv, save_csv, get_active_experiment, get_experiment_config, get_shared_config
from nes.novelty import load_language_model, compute_novelty_scores, compute_transience_scores
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compute novelty and transience scores")
    parser.add_argument("--input", default=None, help="Input CSV path")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()
    
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    simulated = shared_config['cleaning'].get('simulated', False)
    
    print(f"Active experiment: {experiment}")
    
    model_name = shared_config['novelty'].get('model_name', 'mistral-7b')
    window_size = shared_config['novelty'].get('window_size', 128)
    input_file = args.input or ("interaction_level_stories_filtered_simulated.csv" if simulated else "interaction_level_stories_filtered.csv")
    output_file = args.output or ("novelty_scores_simulated.csv" if simulated else "novelty_scores.csv")

    print(f"Loading clean data from {input_file}")
    df = load_csv(Path(input_file).name, stage="interim")
    
    print("Loading language model...")
    tokenizer, model, device = load_language_model(model_name)
    
    print("Computing novelty scores...")
    df = compute_novelty_scores(df, tokenizer, model, window_size=window_size)
    
    print("Computing transience scores...")
    df = compute_transience_scores(df, tokenizer, model, window_size=window_size)
    
    # Drop token ID columns (not needed in output)
    df = df.drop(columns=["author_1_ids", "author_2_ids"], errors="ignore")
    
    save_csv(df, Path(output_file).name)
    print(f"Saved novelty scores to {exp_config['processed_dir']}/{output_file}")
    print("\n✅ Script 07 complete!")


if __name__ == "__main__":
    main()
