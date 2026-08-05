#!/usr/bin/env python
"""
Script 02: Filter and clean story data.

This script:
1. Loads raw story data
2. Applies deterministic text-quality QC (human experiments only)
3. Removes "This is the story of" prefix
4. Adds exchange-aligned analysis metadata
5. Builds full story text (full_story, full_author_1, full_author_2 columns)
6. Saves cleaned data to data/<experiment>/interim/

Supports all three conditions:
- human-ai: Human-AI collaborative stories
- human-human: Human-Human collaborative stories  
- ai-ai: Simulated AI-AI stories

Usage:
    python scripts/02_clean_dataset.py
"""

import sys
from pathlib import Path
import argparse

from dotenv import load_dotenv
load_dotenv()


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nes.cleaning import (
    build_full_story_text,
    build_long_format_analysis,
    filter_by_respondent_id,
    clean_user_ai_start,
    clean_ai_ai_data,
    keep_complete_conversations,
    randomize_author_assignment,
    add_exchange_aligned_metadata,
    split_repeated_conversation_ids,
    add_text_quality_qc,
    build_text_quality_story_summary,
    filter_by_text_quality_story_qc,
    apply_spell_correction,
    drop_text_qc_flagged_rows,
)
from nes.io import load_csv, save_csv, get_project_root, load_config, get_active_experiment, get_experiment_config, get_shared_config

import os


def main():
    
    parser = argparse.ArgumentParser(description="Filter and clean story data.")
    parser.add_argument(
        "--input-csv-raw",
        type=str,
        default=None,
        help="Path to input CSV file with raw story data (auto-detected based on experiment)"
    )
    parser.add_argument(
        "--skip-text-qc",
        action="store_true",
        help="Skip deterministic text-quality QC"
    )
    parser.add_argument(
        "--text-qc-story-min-rows",
        type=int,
        default=None,
        help="Exclude a full story when at least this many rows are text-quality flagged"
    )
    parser.add_argument(
        "--skip-spell-correction",
        action="store_true",
        help="Skip GPT-4o-mini spell correction on human-authored slots"
    )
    parser.add_argument(
        "--drop-flagged-rows",
        action="store_true",
        help="Drop QC-flagged rows at the row level (alternative to story-level exclusion)"
    )
    parser.add_argument(
        "--spell-correction-edit-threshold",
        type=int,
        default=70,
        help="Edit-distance threshold to flag rows where spell correction changed text excessively"
    )
    args = parser.parse_args()
    
    # Load config
    experiment = get_active_experiment()
    exp_config = get_experiment_config()
    shared_config = get_shared_config()
    
    text_qc_config = shared_config['cleaning'].get('text_quality_qc', {})
    text_qc_story_min_rows = text_qc_config.get(
        'story_exclusion_min_flagged_rows',
        2,
    )
    if args.text_qc_story_min_rows is not None:
        text_qc_story_min_rows = args.text_qc_story_min_rows
    simulated = shared_config['cleaning']['simulated']
    max_turns = exp_config['cleaning']['max_turns']
    min_interactions = None
    if exp_config.get('firestore'):
        min_interactions = exp_config['firestore'].get('min_interactions')

    print("=" * 60)
    print(f"Script 02: Clean Dataset ({experiment})")
    print("=" * 60)
    print(f"Simulated mode: {simulated}")

    # Determine input file based on experiment
    if args.input_csv_raw:
        input_file = args.input_csv_raw
    elif experiment == 'ai-ai':
        input_file = "simulated_stories.csv"
    else:
        input_file = "finished_stories_raw.csv"
    
    # Load raw data
    print(f"\nLoading raw story data from {input_file}...")
    df = load_csv(input_file, stage="raw")
    df = df.rename(columns={
        df.columns[1]: "author_1",
        df.columns[2]: "author_2"
    })
    df['condition'] = experiment
    
    print(f"Loaded {len(df)} rows")
    if experiment == 'human-ai':
        df = split_repeated_conversation_ids(
            df,
            group_col='conversation_id',
            expected_length=min_interactions,
        )
    
    # AI-AI has its own cleaning path
    if experiment == 'ai-ai':
        print("\n--- AI-AI Cleaning Pipeline ---")
        df_filtered = clean_ai_ai_data(df, max_turns=max_turns)
        
        # Randomize author assignment to match human-ai starter randomness
        print("\nRandomizing author assignment (50/50 swap)...")
        random_seed = shared_config['analysis']['random_seed']
        df_filtered = randomize_author_assignment(df_filtered, group_col='conversation_id', seed=random_seed)
        
    else:
        # Clean starter text and identify who started before any row/story filter
        # can make a later row look like the beginning of the story.
        df_filtered = clean_user_ai_start(df, max_turns=max_turns, experiment=experiment)

        # Filter by respondent_id (human-ai only) before text QC so invalid/test
        # rows do not inflate quality-control removal counts.
        if 'respondent_id' in df_filtered.columns and experiment == 'human-ai':
            print("\nFiltering by respondent ID...")
            df_filtered = filter_by_respondent_id(df_filtered, threshold=12)
            print(f"✓ Filtered to {len(df_filtered)} rows with valid respondent IDs")
        else:
            print("\nNo respondent_id filtering (not applicable for this experiment)")

        if not args.skip_text_qc and experiment in ['human-ai', 'human-human']:
            print("\nApplying deterministic text-quality QC...")
            text_qc_slots = ['author_1']
            if experiment == 'human-human':
                text_qc_slots.append('author_2')

            df_filtered = add_text_quality_qc(
                df_filtered,
                text_columns=text_qc_slots,
                min_substantive_chars=text_qc_config.get('min_substantive_chars', 3),
                min_alpha_ratio=text_qc_config.get('min_alpha_ratio', 0.45),
                max_symbol_ratio=text_qc_config.get('max_symbol_ratio', 0.35),
                max_keyboard_pattern_ratio=text_qc_config.get('max_keyboard_pattern_ratio', 0.12),
                min_word_vowel_ratio=text_qc_config.get('min_word_vowel_ratio', 0.35),
                long_token_chars=text_qc_config.get('long_token_chars', 24),
                min_word_count_for_vowel_check=text_qc_config.get('min_word_count_for_vowel_check', 3),
            )

            text_qc_audit = df_filtered[df_filtered['text_qc_flagged']].copy()
            audit_columns = [
                'conversation_id', 'source_conversation_id', 'turn',
                'interaction_count', 'starter', 'respondent_id',
                'respondent_id_u1', 'respondent_id_u2', 'llm_type',
                'timestamp', 'text_qc_reasons',
                'author_1', 'author_1_text_qc_reasons',
                'author_1_text_qc_char_count', 'author_1_text_qc_alpha_ratio',
                'author_1_text_qc_symbol_ratio', 'author_1_text_qc_keyboard_pattern_ratio',
                'author_1_text_qc_word_vowel_ratio', 'author_1_text_qc_long_no_vowel_tokens',
                'author_2', 'author_2_text_qc_reasons',
                'author_2_text_qc_char_count', 'author_2_text_qc_alpha_ratio',
                'author_2_text_qc_symbol_ratio', 'author_2_text_qc_keyboard_pattern_ratio',
                'author_2_text_qc_word_vowel_ratio', 'author_2_text_qc_long_no_vowel_tokens',
            ]
            audit_columns = [col for col in audit_columns if col in text_qc_audit.columns]
            save_csv(
                text_qc_audit[audit_columns],
                "text_quality_qc_flagged_rows.csv",
                stage="interim",
            )

            story_qc = build_text_quality_story_summary(
                df_filtered,
                story_exclusion_min_flagged_rows=text_qc_story_min_rows,
                exclude_if_first_human_flagged=text_qc_config.get(
                    'exclude_if_first_human_flagged',
                    True,
                ),
            )
            save_csv(
                story_qc,
                "text_quality_qc_story_summary.csv",
                stage="interim",
            )
            print(f"✓ Saved {len(text_qc_audit)} text-quality flagged row(s)")
            print(f"✓ Saved {len(story_qc)} text-quality story QC summary row(s)")
            df_filtered = filter_by_text_quality_story_qc(df_filtered, story_qc)
        else:
            print("\nSkipping deterministic text-quality QC")

        if not args.skip_spell_correction and experiment in ['human-ai', 'human-human']:
            spell_slots = ['author_1']
            if experiment == 'human-human':
                spell_slots.append('author_2')

            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print(
                    "\nSkipping spell correction: OPENAI_API_KEY environment variable is not set"
                )
            else:
                print(
                    f"\nApplying GPT-4o-mini spell correction to: {', '.join(spell_slots)}"
                )
                df_filtered = apply_spell_correction(
                    df_filtered,
                    text_columns=spell_slots,
                    api_key=api_key,
                    edit_distance_threshold=args.spell_correction_edit_threshold,
                )

        if args.drop_flagged_rows and 'text_qc_flagged' in df_filtered.columns:
            print("\nApplying row-level QC outlier filter (drop flagged rows)...")
            df_filtered = drop_text_qc_flagged_rows(df_filtered)

        # Randomize author assignment for human-human (to match human-ai randomness)
        if experiment == 'human-human':
            print("\nRandomizing author assignment (50/50 swap)...")
            random_seed = shared_config['analysis']['random_seed']
            df_filtered = randomize_author_assignment(df_filtered, group_col='conversation_id', seed=random_seed)

    print("\nRemoving incomplete conversation fragments...")
    df_filtered = keep_complete_conversations(df_filtered, group_col='conversation_id')

    print("\nAdding exchange-aligned analysis metadata...")
    df_filtered = add_exchange_aligned_metadata(df_filtered, experiment=experiment)

    print("\nBuilding full story text...")
    # Save filtered interaction-level data
    output_interaction = "interaction_level_stories_filtered_simulated.csv" if simulated else "interaction_level_stories_filtered.csv"
    save_csv(df_filtered, output_interaction, stage="interim")

    print("\nBuilding long-format analysis export...")
    df_long = build_long_format_analysis(df_filtered)
    output_long = "interaction_level_stories_long_filtered_simulated.csv" if simulated else "interaction_level_stories_long_filtered.csv"
    save_csv(df_long, output_long, stage="interim")
    
    df_stories = build_full_story_text(df_filtered, experiment=experiment)
    output_stories = "stories_full_text_filtered_simulated.csv" if simulated else "stories_full_text_filtered.csv"
    save_csv(df_stories, output_stories, stage="interim")
    
    print(f"\n✓ Filtered to {len(df_filtered)} interaction rows")
    print(f"✓ Built {len(df_long)} long-format contribution rows")
    print(f"✓ Built {len(df_stories)} complete stories")
    print(f"✓ Saved to {exp_config['interim_dir']}/")
    print("\n✅ Script 02 complete!")


if __name__ == "__main__":
    main()
