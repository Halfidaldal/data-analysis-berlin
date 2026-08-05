"""
Data cleaning and filtering functions.

This module handles:
- Firestore data download and filtering (both human-ai and human-human schemas)
- Edit distance filtering
- Spell-checking and rectification
- Story grouping (10 interactions = 1 story)
- Schema normalization (author_1/author_2 with condition column)
"""

from typing import Optional, List, Dict, Any
from uuid import uuid4
import re
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter


def init_firestore(credentials_path: str) -> firestore.Client:
    """
    Initialize Firestore client.
    
    Args:
        credentials_path: Path to Firebase admin SDK JSON file
        
    Returns:
        Firestore client instance
    """
    try:
        # Check if already initialized
        firebase_admin.get_app()
    except ValueError:
        # Not initialized yet
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
    
    return firestore.client()


def normalize_columns(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """
    Normalize column names to unified schema: author_1, author_2, condition.
    
    This should be called after loading/cleaning data to ensure consistent
    column names across experiments for comparative analysis.
    
    Args:
        df: Input DataFrame with experiment-specific column names
        experiment: Experiment name ('human-ai' or 'human-human')
        
    Returns:
        DataFrame with normalized columns
    """
    df = df.copy()
    
    # Add condition column
    df['condition'] = experiment
    
    # Rename author columns based on experiment
    if experiment == 'human-ai':
        # user -> author_1, ai -> author_2
        if 'user' in df.columns:
            df = df.rename(columns={'user': 'author_1'})
        if 'ai' in df.columns:
            df = df.rename(columns={'ai': 'author_2'})
        # Also handle full_user, full_ai if present
        if 'full_user' in df.columns:
            df = df.rename(columns={'full_user': 'full_author_1'})
        if 'full_ai' in df.columns:
            df = df.rename(columns={'full_ai': 'full_author_2'})
        if 'full_user_dot' in df.columns:
            df = df.rename(columns={'full_user_dot': 'full_author_1_dot'})
        if 'full_ai_dot' in df.columns:
            df = df.rename(columns={'full_ai_dot': 'full_author_2_dot'})
            
    elif experiment == 'human-human':
        # user -> author_1, user2 -> author_2
        if 'user' in df.columns:
            df = df.rename(columns={'user': 'author_1'})
        if 'user2' in df.columns:
            df = df.rename(columns={'user2': 'author_2'})
        # Also handle full_user, full_ai (used in human-human despite the name)
        if 'full_user' in df.columns:
            df = df.rename(columns={'full_user': 'full_author_1'})
        if 'full_ai' in df.columns:
            df = df.rename(columns={'full_ai': 'full_author_2'})
        if 'full_user_dot' in df.columns:
            df = df.rename(columns={'full_user_dot': 'full_author_1_dot'})
        if 'full_ai_dot' in df.columns:
            df = df.rename(columns={'full_ai_dot': 'full_author_2_dot'})
            
    elif experiment == 'ai-ai':
        # author_1 -> author_1, author_2 -> author_2
        if 'author_1' in df.columns:
            df = df.rename(columns={'author_1': 'author_1'})
        if 'author_2' in df.columns:
            df = df.rename(columns={'author_2': 'author_2'})
        # Handle full text columns if present
        if 'full_author_1' in df.columns:
            df = df.rename(columns={'full_author_1': 'full_author_1'})
        if 'full_author_2' in df.columns:
            df = df.rename(columns={'full_author_2': 'full_author_2'})
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    
    print(f"Normalized columns for {experiment} experiment")
    return df


def download_stories_from_firestore(
    db: firestore.Client,
    collection_name: str = "story_data_TEXT",
    min_interactions: int = 9,
    schema: str = "flat"
) -> pd.DataFrame:
    """
    Download stories from Firestore, keeping only complete stories.
    
    Supports two Firestore schemas:
    - "flat": Human-AI data (all interactions in single collection)
    - "nested": Human-Human data (sessions with nested turns subcollection)
    
    Args:
        db: Firestore client
        collection_name: Name of the Firestore collection
        min_interactions: Minimum number of interactions to count as a story
        schema: Firestore schema type ("flat" or "nested")
        
    Returns:
        DataFrame with story interaction rows (only complete stories)
    """
    if schema == "flat":
        return _download_flat_schema(db, collection_name, min_interactions)
    elif schema == "nested":
        return _download_nested_schema(db, collection_name, min_interactions)
    else:
        raise ValueError(f"Unknown schema: {schema}. Must be 'flat' or 'nested'")


def _download_flat_schema(
    db: firestore.Client,
    collection_name: str,
    min_interactions: int
) -> pd.DataFrame:
    """Download from flat collection schema (human-ai)."""
    story_data_ref = db.collection(collection_name)
    
    count = {}
    out_rows = []
    full_story_data = []
    current_conv = None
    conv_docs = []

    docs = story_data_ref.order_by("conversation_id").order_by("timestamp").stream()
    
    for doc in docs:
        doc_conv = doc.get("conversation_id")

        # When conversation changes, process the accumulated conv_docs
        if current_conv is not None and doc_conv != current_conv:
            n = len(conv_docs)
            num_full = n // min_interactions
            if num_full:
                for i in range(num_full):
                    story_slice = conv_docs[i * min_interactions:(i + 1) * min_interactions]
                    full_story_data.append(story_slice)
                    out_rows.extend(story_slice)
                count[current_conv] = count.get(current_conv, 0) + num_full
            conv_docs = []
            current_conv = doc_conv
        else:
            if current_conv is None:
                current_conv = doc_conv

        doc_dictionary = doc.to_dict()
        row = {
            "timestamp": doc_dictionary.get("timestamp"),
            "user": doc_dictionary.get("user"),
            "ai": doc_dictionary.get("ai"),
            "conversation_id": doc_conv,
            "respondent_id": doc_dictionary.get("respondent_id"),
            "interaction_count": doc_dictionary.get("interaction_count"),
            "llm_type": doc_dictionary.get("llm_type"),
        }
        conv_docs.append(row)

    # Process the final conversation
    if current_conv is not None and conv_docs:
        n = len(conv_docs)
        num_full = n // min_interactions
        if num_full:
            for i in range(num_full):
                story_slice = conv_docs[i * min_interactions:(i + 1) * min_interactions]
                full_story_data.append(story_slice)
                out_rows.extend(story_slice)
            count[current_conv] = count.get(current_conv, 0) + num_full

    total_stories = sum(count.values()) if count else 0
    print(f"Downloaded {len(out_rows)} interactions from {total_stories} complete stories (flat schema)")
    
    return pd.DataFrame(out_rows)


def _download_nested_schema(
    db: firestore.Client,
    collection_name: str,
    min_interactions: int
) -> pd.DataFrame:
    """Download from nested collection schema (human-human)."""
    story_data_ref = db.collection(collection_name)
    
    count = {}
    out_rows = []

    docs = story_data_ref.stream()
    for story_doc in docs:
        story_meta = story_doc.to_dict() or {}
        doc_conv = story_doc.id
        participants = story_meta.get("participants", [])
        if not isinstance(participants, list) or len(participants) != 2:
            print(f"Skipping conversation {doc_conv} due to invalid participants field")
            continue

        turn_docs = list(
            story_doc.reference.collection("turns").order_by("timestamp").stream()
        )
        turn_dicts = [t.to_dict() or {} for t in turn_docs]

        n_turns = len(turn_dicts)
        n_interactions = n_turns // 2
        print(f"{doc_conv}: turns={n_turns}, interactions={n_interactions}, needed={min_interactions}")

        # Skip if odd number of turns or incorrect participants
        if len(turn_dicts) % 2 != 0 or len(story_meta.get("participants", [])) != 2:
            print(f"Skipping conversation {doc_conv} due to odd number of turns or incorrect participants")
            continue

        conv_docs = []
        for i in range(0, len(turn_dicts) - 1, 2):
            t1 = turn_dicts[i]
            t2 = turn_dicts[i + 1]

            row = {
                "timestamp": t2.get("timestamp") or t1.get("timestamp"),
                "user": t1.get("text"),
                "user2": t2.get("text"),
                "conversation_id": doc_conv,
                "respondent_id_u1": t1.get("userId"),
                "respondent_id_u2": t2.get("userId"),
                "interaction_count": (i // 2) + 1,
            }
            conv_docs.append(row)

        # Keep the entire conversation if it meets the threshold
        if conv_docs and len(conv_docs) >= min_interactions:
            out_rows.extend(conv_docs)
            count[doc_conv] = 1

    total_stories = sum(count.values()) if count else 0
    print(f"Downloaded {len(out_rows)} interactions from {total_stories} complete stories (nested schema)")
        
    return pd.DataFrame(out_rows)

def delete_incomplete_stories_from_firestore(
    db: firestore.Client,
    story_data_collection: str = "story_data_TEXT",
    stories_collection: str = "stories_TEXT",
    min_interactions: int = 10
) -> int:
    """
    Delete stories from Firestore that have fewer than min_interactions.
    
    Args:
        db: Firestore client
        story_data_collection: Name of the story_data collection
        stories_collection: Name of the stories collection
        min_interactions: Minimum interactions required
        
    Returns:
        Number of stories deleted
    """
    story_data_ref = db.collection(story_data_collection)
    stories_ref = db.collection(stories_collection)
    
    counts = {}
    docs = story_data_ref.order_by("conversation_id").order_by("timestamp").stream()
    conversation_id = None
    deleted_count = 0
    
    for doc in docs:
        # Only consider deletion when we have a previous conversation_id
        if (conversation_id is not None and 
            conversation_id != doc.get("conversation_id") and 
            counts.get(conversation_id, 0) < min_interactions):
            print(f"Deleting story with conversation_id: {conversation_id}")
            story_query = stories_ref.where(
                FieldFilter("conversation_id", "==", conversation_id)
            )
            story_docs = story_query.stream()
            for story_doc in story_docs:
                print(f" - Deleting story document ID: {story_doc.id}")
                stories_ref.document(story_doc.id).delete()
                deleted_count += 1
            
        conversation_id = doc.get("conversation_id")
        if conversation_id is None:
            continue
        counts[conversation_id] = counts.get(conversation_id, 0) + 1

    # Final check for the last conversation after streaming all documents
    if (conversation_id is not None and 
        counts.get(conversation_id, 0) < min_interactions):
        print(f"Deleting story with conversation_id: {conversation_id}")
        story_query = stories_ref.where(
            FieldFilter("conversation_id", "==", conversation_id)
        )
        story_docs = story_query.stream()
        for story_doc in story_docs:
            print(f" - Deleting story document ID: {story_doc.id}")
            stories_ref.document(story_doc.id).delete()
            deleted_count += 1

    story_count = sum(cnt // min_interactions for cnt in counts.values())
    print(f"Number of complete stories remaining: {story_count}")
    print(f"Total story documents deleted: {deleted_count}")
    
    return deleted_count


def apply_spell_correction(
    df: pd.DataFrame,
    text_columns: List[str],
    api_key: str,
    preserve_raw_suffix: str = "_raw",
    edit_distance_suffix: str = "_edit_distance",
    edit_distance_threshold: Optional[int] = 70,
    skip_if_qc_flagged: bool = True,
) -> pd.DataFrame:
    """
    Apply LLM spell-correction to selected text columns.

    For each row in each target column the text is sent to GPT-4o-mini via
    `nes.process_spelling_openai.correct_spelling`. The original text is
    preserved in a `<column><preserve_raw_suffix>` column, the Levenshtein
    distance between original and corrected text is stored in
    `<column><edit_distance_suffix>`, and the column itself is replaced with
    the corrected text.

    Rows with text-quality QC flags on a given slot are not sent to the
    correction API (they are typically gibberish that the corrector cannot
    repair); their original text is retained and edit distance is set to 0.

    If `edit_distance_threshold` is provided, rows whose edit distance on any
    target column exceeds the threshold are marked in
    `spell_correction_excessive_edit` (boolean). They are not dropped here -
    downstream code or `filter_by_edit_distance` decides whether to drop them.

    Args:
        df: Input DataFrame.
        text_columns: Columns to spell-correct (e.g., ['author_1'] for HA,
            ['author_1', 'author_2'] for HH).
        api_key: OpenAI API key.
        preserve_raw_suffix: Suffix for the preserved original text column.
        edit_distance_suffix: Suffix for the per-row edit distance column.
        edit_distance_threshold: Optional Levenshtein cutoff for the audit flag.
        skip_if_qc_flagged: If True, skip API calls on rows where the slot's
            text_qc_flagged is True.

    Returns:
        DataFrame with corrected text + audit columns.
    """
    from nes.process_spelling_openai import correct_spelling, compute_edit_distance_values

    if not text_columns:
        return df.copy()
    missing = [c for c in text_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns for spell correction: {missing}")
    if not api_key:
        raise ValueError("api_key is required for spell correction")

    df = df.copy()
    excessive = pd.Series(False, index=df.index)

    for column in text_columns:
        raw_col = f"{column}{preserve_raw_suffix}"
        ed_col = f"{column}{edit_distance_suffix}"
        df[raw_col] = df[column]

        flag_col = f"{column}_text_qc_flagged"
        skip_mask = (
            df[flag_col].fillna(False).astype(bool)
            if (skip_if_qc_flagged and flag_col in df.columns)
            else pd.Series(False, index=df.index)
        )

        corrected = []
        n_calls = 0
        n_skipped = 0
        for idx, value in df[column].items():
            if skip_mask.loc[idx]:
                corrected.append(value)
                n_skipped += 1
                continue
            corrected.append(correct_spelling(value, api_key=api_key))
            n_calls += 1
        df[column] = corrected

        df[ed_col] = [
            compute_edit_distance_values(orig, corr)
            for orig, corr in zip(df[raw_col], df[column])
        ]

        if edit_distance_threshold is not None:
            excessive = excessive | (df[ed_col].fillna(0) > edit_distance_threshold)

        print(
            f"  Spell correction on '{column}': "
            f"{n_calls} API call(s), {n_skipped} skipped (qc-flagged), "
            f"mean edit distance = {df[ed_col].mean():.2f}"
        )

    df["spell_correction_excessive_edit"] = excessive
    if edit_distance_threshold is not None:
        n_excessive = int(excessive.sum())
        print(
            f"  Rows with edit distance > {edit_distance_threshold} flagged: "
            f"{n_excessive}/{len(df)} ({100*n_excessive/max(len(df),1):.1f}%)"
        )
    return df


def drop_text_qc_flagged_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop turn-level rows where text-quality QC has flagged any author slot.

    Used as a row-level outlier filter (alternative to story-level exclusion).
    Requires the `text_qc_flagged` column produced by `add_text_quality_qc`.
    """
    if "text_qc_flagged" not in df.columns:
        return df.copy()
    n_before = len(df)
    out = df[~df["text_qc_flagged"].fillna(False).astype(bool)].copy()
    n_after = len(out)
    print(
        f"Row-level QC filtering: dropped {n_before - n_after} flagged row(s) "
        f"({100*(n_before - n_after)/max(n_before, 1):.1f}%)"
    )
    return out


def filter_by_edit_distance(
    df: pd.DataFrame,
    threshold: int,
    column: str = "edit_distance"
) -> pd.DataFrame:
    """
    Filter rows based on edit distance threshold.
    
    Args:
        df: Input DataFrame
        threshold: Maximum edit distance to keep
        column: Name of the edit distance column
        
    Returns:
        Filtered DataFrame
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    n_before = len(df)
    filtered_df = df[df[column] <= threshold].copy()
    n_after = len(filtered_df)
    n_removed = n_before - n_after
    
    print(f"Edit distance filtering (threshold={threshold}):")
    print(f"  Before: {n_before} rows")
    print(f"  After: {n_after} rows")
    print(f"  Removed: {n_removed} rows ({100*n_removed/n_before:.1f}%)")
    
    return filtered_df


def split_repeated_conversation_ids(
    df: pd.DataFrame,
    group_col: str = "conversation_id",
    expected_length: Optional[int] = None,
    source_col: str = "source_conversation_id"
) -> pd.DataFrame:
    """
    Split repeated Firestore conversation IDs into analysis story IDs.

    The flat Firestore downloader can emit multiple complete story slices with
    the same conversation_id when a source conversation contains 2x, 3x, ...
    the expected number of rows. Downstream analysis treats conversation_id as
    the story key, so repeated IDs need stable per-slice suffixes.
    """
    if expected_length is None:
        return df.copy()
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")

    df = df.copy()
    df[source_col] = df[group_col]

    split_count = 0
    split_story_count = 0

    for group_value, group_index in df[df[group_col].notna()].groupby(group_col, sort=False).groups.items():
        indices = list(group_index)
        group_size = len(indices)
        if group_size <= expected_length or group_size % expected_length != 0:
            continue

        n_slices = group_size // expected_length
        split_count += 1
        split_story_count += n_slices

        for offset, idx in enumerate(indices):
            story_number = (offset // expected_length) + 1
            df.at[idx, group_col] = f"{group_value}__story_{story_number:02d}"

    if split_count:
        print(
            f"Split {split_count} repeated conversation_id value(s) into "
            f"{split_story_count} story instance IDs"
        )

    return df


_LETTER_RE = re.compile(r'[^\W\d_]', re.UNICODE)
_WORD_RE = re.compile(r'[A-Za-zÀ-ÖØ-öø-ÿ]+')
_KEYBOARD_PATTERN_RE = re.compile(
    r'asdf|sdf|dfg|fgh|jkl|klj|qwer|wert|zxc|xcv|lkj|dflk|fkl|kldf|dfjk|jkas|ælk|ølk',
    re.IGNORECASE
)
_VOWELS = set('aeiouyæøå')


def _text_quality_metrics(value: Any) -> Dict[str, Any]:
    text = '' if pd.isna(value) else str(value)
    stripped = text.strip()
    no_space = re.sub(r'\s+', '', stripped)
    letters = _LETTER_RE.findall(stripped)
    words = _WORD_RE.findall(stripped.lower())
    keyboard_pattern_chars = sum(
        len(match.group(0)) for match in _KEYBOARD_PATTERN_RE.finditer(stripped)
    )

    nonspace_chars = len(no_space)
    letter_chars = len(letters)
    word_chars = sum(len(word) for word in words)
    word_count = len(words)
    symbol_chars = sum(1 for char in no_space if not char.isalnum())
    long_no_vowel_tokens = sum(
        len(word) >= 8 and not any(char in _VOWELS for char in word)
        for word in words
    )
    longest_token_chars = max((len(word) for word in words), default=0)
    vowel_word_count = sum(any(char in _VOWELS for char in word) for word in words)

    return {
        'char_count': len(stripped),
        'nonspace_chars': nonspace_chars,
        'letter_chars': letter_chars,
        'word_chars': word_chars,
        'word_count': word_count,
        'alpha_ratio': letter_chars / max(nonspace_chars, 1),
        'symbol_ratio': symbol_chars / max(nonspace_chars, 1),
        'keyboard_pattern_ratio': keyboard_pattern_chars / max(letter_chars, 1),
        'word_vowel_ratio': vowel_word_count / max(word_count, 1),
        'long_no_vowel_tokens': long_no_vowel_tokens,
        'longest_token_chars': longest_token_chars,
    }


def add_text_quality_qc(
    df: pd.DataFrame,
    text_columns: List[str],
    group_col: str = 'conversation_id',
    min_substantive_chars: int = 3,
    min_alpha_ratio: float = 0.45,
    max_symbol_ratio: float = 0.35,
    max_keyboard_pattern_ratio: float = 0.12,
    min_word_vowel_ratio: float = 0.35,
    long_token_chars: int = 24,
    min_word_count_for_vowel_check: int = 3,
) -> pd.DataFrame:
    """
    Add deterministic text-quality QC columns without changing text.

    The goal is to flag likely invalid/test/gibberish inputs for audit and
    story-level exclusion. This is deliberately not a spell-correction step.
    Ordinary spelling mistakes remain in the analysis text.
    """
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")
    missing = [column for column in text_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing text columns for QC: {missing}")

    df = df.copy()
    slot_flag_columns = []
    slot_reason_columns = []

    for column in text_columns:
        metrics = df[column].map(_text_quality_metrics).apply(pd.Series)
        prefix = f'{column}_text_qc_'
        for metric_column in metrics.columns:
            df[f'{prefix}{metric_column}'] = metrics[metric_column]

        protocol_empty = pd.Series(False, index=df.index)
        if column == 'author_1' and {'starter', 'turn'}.issubset(df.columns):
            protocol_empty = (
                df['starter'].eq('author_2')
                & df['turn'].eq(1)
                & df[f'{prefix}word_chars'].lt(min_substantive_chars)
            )

        non_substantive = (
            df[f'{prefix}word_chars'].lt(min_substantive_chars)
            & ~protocol_empty
        )
        low_alpha = (
            df[f'{prefix}nonspace_chars'].ge(min_substantive_chars)
            & df[f'{prefix}alpha_ratio'].lt(min_alpha_ratio)
        )
        high_symbols = (
            df[f'{prefix}nonspace_chars'].ge(min_substantive_chars)
            & df[f'{prefix}symbol_ratio'].gt(max_symbol_ratio)
        )
        keyboard_smash = (
            df[f'{prefix}keyboard_pattern_ratio'].gt(max_keyboard_pattern_ratio)
            | df[f'{prefix}long_no_vowel_tokens'].ge(2)
            | (
                df[f'{prefix}longest_token_chars'].ge(long_token_chars)
                & df[f'{prefix}keyboard_pattern_ratio'].gt(0)
            )
        )
        low_vowels = (
            df[f'{prefix}word_count'].ge(min_word_count_for_vowel_check)
            & df[f'{prefix}word_vowel_ratio'].lt(min_word_vowel_ratio)
        )

        flag_column = f'{prefix}flagged'
        reason_column = f'{prefix}reasons'
        df[flag_column] = non_substantive | low_alpha | high_symbols | keyboard_smash | low_vowels

        reasons = pd.Series('', index=df.index, dtype='object')
        for mask, reason in [
            (non_substantive, 'non_substantive'),
            (low_alpha, 'low_alpha_ratio'),
            (high_symbols, 'high_symbol_ratio'),
            (keyboard_smash, 'keyboard_smash_pattern'),
            (low_vowels, 'low_word_vowel_ratio'),
        ]:
            reasons = reasons.mask(mask, reasons + reason + ';')
        reasons = reasons.str.rstrip(';')
        reasons = reasons.mask(protocol_empty & reasons.eq(''), 'protocol_empty')
        df[reason_column] = reasons

        slot_flag_columns.append(flag_column)
        slot_reason_columns.append(reason_column)

    df['text_qc_flagged'] = df[slot_flag_columns].any(axis=1)

    def combine_reasons(row: pd.Series) -> str:
        parts = []
        for column, reason_column in zip(text_columns, slot_reason_columns):
            reason = row[reason_column]
            if isinstance(reason, str) and reason and reason != 'protocol_empty':
                parts.append(f'{column}:{reason}')
        return '|'.join(parts)

    df['text_qc_reasons'] = df.apply(combine_reasons, axis=1)
    return df


def build_text_quality_story_summary(
    df: pd.DataFrame,
    group_col: str = 'conversation_id',
    story_exclusion_min_flagged_rows: int = 2,
    exclude_if_first_human_flagged: bool = True,
) -> pd.DataFrame:
    """Summarize deterministic text-quality QC at story level."""
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")
    if 'text_qc_flagged' not in df.columns:
        raise ValueError("Expected 'text_qc_flagged' column in DataFrame")

    agg = {
        'n_rows': (group_col, 'size'),
        'n_text_qc_flagged': ('text_qc_flagged', 'sum'),
        'starter': ('starter', 'first'),
    }
    for optional_col in [
        'respondent_id', 'respondent_id_u1', 'respondent_id_u2',
        'source_conversation_id', 'condition'
    ]:
        if optional_col in df.columns:
            agg[optional_col] = (optional_col, 'first')

    summary = df.groupby(group_col, dropna=False).agg(**agg).reset_index()

    first_human_flagged = pd.Series(False, index=summary.index)
    if exclude_if_first_human_flagged and {'starter', 'turn', 'author_1_text_qc_flagged'}.issubset(df.columns):
        first_human_stories = set(
            df[
                df['starter'].eq('author_1')
                & df['turn'].eq(1)
                & df['author_1_text_qc_flagged']
            ][group_col]
        )
        first_human_flagged = summary[group_col].isin(first_human_stories)

    summary['exclude_first_human_qc_flagged'] = first_human_flagged
    summary['exclude_for_text_qc'] = (
        summary['n_text_qc_flagged'].ge(story_exclusion_min_flagged_rows)
        | summary['exclude_first_human_qc_flagged']
    )
    return summary


def filter_by_text_quality_story_qc(
    df: pd.DataFrame,
    story_summary: pd.DataFrame,
    group_col: str = 'conversation_id',
) -> pd.DataFrame:
    """Remove full stories marked for exclusion by text-quality QC summary."""
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")
    if group_col not in story_summary.columns or 'exclude_for_text_qc' not in story_summary.columns:
        raise ValueError("Story summary must include conversation IDs and exclude_for_text_qc")

    n_before = len(df)
    excluded_ids = set(story_summary.loc[story_summary['exclude_for_text_qc'], group_col])
    filtered = df[~df[group_col].isin(excluded_ids)].copy()
    n_after = len(filtered)
    n_removed = n_before - n_after
    print("Text-quality story filtering:")
    print(f"  Excluding: {len(excluded_ids)} conversation(s)")
    print(f"  Before: {n_before} rows")
    print(f"  After: {n_after} rows")
    print(f"  Removed: {n_removed} rows ({100*n_removed/max(n_before, 1):.1f}%)")
    return filtered


def append_turn_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append turn numbers within each conversation_id.
    
    Args:
        df: Input DataFrame with 'conversation_id' column
        
    Returns:
        DataFrame with new 'turn' column
    """
    df_out = df.copy()
    df_out['turn'] = df_out.groupby('conversation_id').cumcount() + 1
    return df_out


def filter_by_respondent_id(
    df: pd.DataFrame,
    threshold: int = 12,
    column: str = "respondent_id"
) -> pd.DataFrame:
    """
    Filter DataFrame by respondent_id length.
    
    Args:
        df: Input DataFrame
        column: Name of the respondent ID column
        threshold: Required length of respondent_id string
        
    Returns:
        Filtered DataFrame (keeping only rows where len(respondent_id) == threshold)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    n_before = len(df)
    respondent_ids = df[column].astype(str)
    filtered_df = df[respondent_ids.str.len() == threshold].copy()

    # removing remaining test_ids
    filtered_df = filtered_df[~filtered_df[column].astype(str).str.startswith("test-")]

    n_after = len(filtered_df)
    n_removed = n_before - n_after
    
    print(f"Filtering by respondent_id length={threshold}:")
    print(f"  Before: {n_before} rows")
    print(f"  After: {n_after} rows")
    print(f"  Removed: {n_removed} rows ({100*n_removed/n_before:.1f}%)")
    
    return filtered_df

def build_full_story_text(df: pd.DataFrame, experiment: str = "human-ai") -> pd.DataFrame:
    """
    Build full_story, full_author_1, and full_author_2 columns by concatenating
    text within each conversation_id.
    
    Uses standardized column names: author_1, author_2, conversation_id.
    
    Args:
        df: DataFrame with author_1, author_2, conversation_id columns
        experiment: Experiment type, used as a fallback when no condition column is present
        
    Returns:
        DataFrame grouped by conversation_id with concatenated text columns
    """
    group_col = 'conversation_id'

    # Validate columns
    if 'author_1' not in df.columns:
        raise ValueError("Expected 'author_1' column in DataFrame")
    if 'author_2' not in df.columns:
        raise ValueError("Expected 'author_2' column in DataFrame")
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")
    
    def _normalize_story_text(value: Any) -> str:
        """Collapse embedded newlines so story text is built as a single line."""
        if pd.isna(value):
            return ''
        return ' '.join(str(value).split())

    def _concat_series(series: pd.Series) -> str:
        return ' '.join(
            text for text in series.map(_normalize_story_text).tolist() if text
        )

    def _concat_series_with_period(series: pd.Series) -> str:
        return ' '.join(
            f"{text}." for text in series.map(_normalize_story_text).tolist() if text
        )

    sort_columns = [group_col]
    if 'turn' in df.columns:
        sort_columns.append('turn')
    elif 'interaction_count' in df.columns:
        sort_columns.append('interaction_count')
    if 'timestamp' in df.columns:
        sort_columns.append('timestamp')

    ordered_df = df.sort_values(sort_columns, kind='stable').copy()

    metadata_columns = [
        'condition', 'language', 'client_id', 'workshop_id', 'timestamp',
        'respondent_id', 'respondent_id_u1', 'respondent_id_u2',
        'source_conversation_id',
        'interaction_count', 'starter', 'starter_side', 'starter_type',
        'author_1_type', 'author_2_type', 'llm_type', 'model_id', 'turn'
    ]

    def _row_slot_order(row: pd.Series) -> List[str]:
        condition = row.get('condition', experiment)
        condition = str(condition) if not pd.isna(condition) else experiment
        starter = row.get('starter_side', row.get('starter', 'author_1'))

        # Human-AI rows are stored as human input -> AI response for every
        # complete exchange. If the AI starts, the primer is an incomplete row
        # with only author_2 populated; appending non-empty slots in this order
        # still preserves chronology.
        if condition == 'human-ai':
            return ['author_1', 'author_2']

        if starter == 'author_2':
            return ['author_2', 'author_1']
        return ['author_1', 'author_2']

    def _build_chronological_story(grp: pd.DataFrame) -> str:
        parts = []
        for _, row in grp.iterrows():
            for slot in _row_slot_order(row):
                text = _normalize_story_text(row[slot])
                if text:
                    parts.append(text)
        return ' '.join(parts)

    records = []
    for conversation_id, grp in ordered_df.groupby(group_col, sort=False):
        record = {
            group_col: conversation_id,
            'full_author_1': _concat_series(grp['author_1']),
            'full_author_2': _concat_series(grp['author_2']),
        }

        for col in metadata_columns:
            if col in grp.columns:
                record[col] = grp[col].iloc[0]

        record['full_author_1_dot'] = _concat_series_with_period(grp['author_1'])
        record['full_author_2_dot'] = _concat_series_with_period(grp['author_2'])
        record['full_story'] = _build_chronological_story(grp)
        records.append(record)

    story_df = pd.DataFrame.from_records(records)

    print(f"Built full story text for {len(story_df)} stories")

    return story_df


def clean_user_ai_start(df: pd.DataFrame, interaction_count: bool = True, max_turns: int = 10, experiment: str = "human-ai") -> pd.DataFrame: 
    """
    Clean starter text and identify which author started.
    
    For human-ai experiment:
    - If author_1's first turn is ONLY "This is the story of" (no additional content),
      then author_2 (AI) started the story
    - If author_1's first turn has "This is the story of" + additional content,
      then author_1 (human) started the story
    
    Uses standardized column names: author_1, author_2.
    
    Args:
        df: Input DataFrame with author_1, author_2 columns
        interaction_count: Whether to filter by interaction_count column
        max_turns: Maximum turns to keep
        experiment: Experiment type ('human-ai' or 'human-human')
        
    Returns:
        Cleaned DataFrame with 'starter' column
    """
    BASELINE = "This is the story of"
    BASELINE_PATTERN = re.compile(r'^\s*this\s+is\s+the\s+story\s+of\b[\s:,\-.]*', re.IGNORECASE)
    MIN_CONTENT_LENGTH = 10  # Minimum chars beyond baseline to count as "started"
    
    df = df.copy()
    
    # Story-level logic must use the conversation identifier.
    # Participant identifiers can be missing or reused and are therefore not
    # reliable story boundaries for starter detection or turn numbering.
    group_col = 'conversation_id'
    
    # Add turn numbers within each group
    df['turn'] = df.groupby(group_col).cumcount() + 1

    def strip_baseline(text: Any) -> Any:
        """Remove a leading story-prefix placeholder while preserving missing values."""
        if pd.isna(text):
            return text
        return BASELINE_PATTERN.sub('', str(text), count=1).strip()
    
    # Identify starters based on first turn content
    def detect_starter(group):
        """Detect who started based on first turn's author_1 content."""
        first_row = group[group['turn'] == 1]
        if first_row.empty:
            return 'author_1'  # Default
        
        author_1_text = str(first_row['author_1'].iloc[0]) if pd.notna(first_row['author_1'].iloc[0]) else ''
        
        # Remove baseline and check remaining content
        remainder = strip_baseline(author_1_text)
        
        # If author_1 only had the baseline placeholder (no real content), author_2 started
        if len(remainder) < MIN_CONTENT_LENGTH:
            return 'author_2'
        else:
            return 'author_1'
    
    # Build starter map
    starter_map = df.groupby(group_col, group_keys=False).apply(detect_starter, include_groups=False)
    df['starter'] = df[group_col].map(starter_map)
    
    # Report starter distribution
    author_1_started = (starter_map == 'author_1').sum()
    author_2_started = (starter_map == 'author_2').sum()
    print(f"Starter detection: {author_1_started} author_1-started, {author_2_started} author_2-started")
    
    # Filter by max_turns
    if interaction_count and 'interaction_count' in df.columns:
        df = df[df['interaction_count'] <= max_turns].copy()
    else:
        df = df[df['turn'] <= max_turns].copy()
    
    # Remove leading baseline text from both columns after starter detection.
    df['author_1'] = df['author_1'].map(strip_baseline)
    df['author_2'] = df['author_2'].map(strip_baseline)

    return df 


def clean_ai_ai_data(df: pd.DataFrame, max_turns: int = 10) -> pd.DataFrame:
    """
    Clean AI-AI simulated data.
    
    Handles:
    - Removing "This is the story of" prefix from first turn
    - Filtering to max_turns
    - Adding metadata columns for compatibility with other conditions
    
    Args:
        df: Raw AI-AI simulation data with columns:
            turn, author_1, author_2, story_id, model_id, timestamp
        max_turns: Maximum turns per story (default 10)
        
    Returns:
        Cleaned DataFrame with standardized structure
    """
    df = df.copy()
    
    # Story prefix used in simulation
    STORY_PREFIX = "This is the story of"
    
    print(f"Cleaning AI-AI data: {len(df)} rows, {df['story_id'].nunique()} stories")
    
    # Remove "This is the story of" prefix from author_1's first turn
    # The prefix is prepended with \n in simulation, so handle both cases
    df['author_1'] = df['author_1'].str.replace(f"{STORY_PREFIX}\n", "", regex=False)
    df['author_1'] = df['author_1'].str.replace(STORY_PREFIX, "", regex=False).str.strip()
    
    # Also clean author_2 just in case (shouldn't have it, but for safety)
    df['author_2'] = df['author_2'].str.replace(STORY_PREFIX, "", regex=False).str.strip()
    
    # Filter to max_turns
    if 'turn' in df.columns:
        df = df[df['turn'] <= max_turns].copy()
    
    # Replace model-revealing story IDs with opaque conversation IDs.
    story_ids = df['story_id'].dropna().unique()
    conversation_id_map = {
        story_id: f"conv_{uuid4().hex}"
        for story_id in story_ids
    }
    df['conversation_id'] = df['story_id'].map(conversation_id_map)
    
    # Add starter column (author_1 always starts in AI-AI)
    df['starter'] = 'author_1'
    
    # Add interaction_count column for compatibility
    df['interaction_count'] = df['turn']
    
    # Count prefix removals
    n_prefixes_removed = len(df[df['turn'] == 1])
    print(f"Removed '{STORY_PREFIX}' prefix from {n_prefixes_removed} first turns")
    print(f"Generated opaque conversation IDs for {len(conversation_id_map)} stories")
    print(f"After cleaning: {len(df)} rows")
    
    return df


def _is_substantive_text(
    series: pd.Series,
    min_substantive_chars: int = 2
) -> pd.Series:
    """
    Identify rows with substantive text after removing whitespace and punctuation.

    A minimum of two remaining word characters treats ".", "," or other
    punctuation-only cells as empty while still counting very short real content.
    """
    normalized = (
        series.fillna('')
        .astype(str)
        .str.strip()
        .str.replace(r'[\W_]+', '', regex=True)
    )
    return normalized.str.len() >= min_substantive_chars


def add_exchange_aligned_metadata(
    df: pd.DataFrame,
    experiment: Optional[str] = None,
    group_col: str = 'conversation_id',
    min_substantive_chars: int = 2
) -> pd.DataFrame:
    """
    Add analysis-ready exchange metadata without changing stored author slots.

    Added columns:
    - author_1_type / author_2_type
    - complete_exchange
    - analysis_turn
    - starter_side
    - starter_type
    """
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")
    if 'starter' not in df.columns:
        raise ValueError("Expected 'starter' column in DataFrame")

    df = df.copy()

    if 'condition' not in df.columns:
        if experiment is None:
            raise ValueError("Either provide 'experiment' or include a 'condition' column")
        df['condition'] = experiment

    if 'turn' not in df.columns:
        df['turn'] = df.groupby(group_col).cumcount() + 1

    author_1_type_map = {
        'human-ai': 'human',
        'human-human': 'human',
        'ai-ai': 'ai',
    }
    author_2_type_map = {
        'human-ai': 'ai',
        'human-human': 'human',
        'ai-ai': 'ai',
    }

    unknown_conditions = sorted(set(df['condition'].dropna().unique()) - set(author_1_type_map))
    if unknown_conditions:
        raise ValueError(f"Unknown condition(s): {unknown_conditions}")

    df['author_1_type'] = df['condition'].map(author_1_type_map)
    df['author_2_type'] = df['condition'].map(author_2_type_map)

    df['starter_side'] = df['starter']
    invalid_starters = sorted(set(df['starter_side'].dropna().unique()) - {'author_1', 'author_2'})
    if invalid_starters:
        raise ValueError(f"Invalid starter values: {invalid_starters}")

    df['starter_type'] = np.where(
        df['starter_side'].eq('author_1'),
        df['author_1_type'],
        df['author_2_type']
    )

    author_1_substantive = _is_substantive_text(df['author_1'], min_substantive_chars=min_substantive_chars)
    author_2_substantive = _is_substantive_text(df['author_2'], min_substantive_chars=min_substantive_chars)
    df['complete_exchange'] = author_1_substantive & author_2_substantive

    # Symmetric opener handling across conditions.
    #
    # The opener of a story (its first chronological turn) has no prior dialog
    # context, so any context-dependent metric is structurally degenerate on
    # that turn. To remove this confound:
    #
    #   - In HA when the AI starts, the AI primer is already isolated in an
    #     incomplete row (author_1 is empty), so it is excluded via
    #     complete_exchange == False.
    #   - In HA human-starter, HH, and AA, the opener is part of the first
    #     complete row (turn == 1). That whole row is marked pre-exchange
    #     (analysis_turn = NA). The row is preserved; it just falls outside
    #     the analysis window.
    #
    # The previous implementation restricted this to starter_side == 'author_1',
    # which silently left the first complete row included for HH/AA stories
    # randomized to starter_side == 'author_2'. That created an asymmetry across
    # the randomization arm: starter='author_2' stories carried one extra early
    # row through the analysis_turn window relative to starter='author_1'
    # stories, on top of the chronology mis-handling those rows received in
    # novelty/transience scoring.
    ordered = df.sort_values([group_col, 'turn'], kind='stable').copy()
    pre_exchange_opener_row = (
        ordered['turn'].eq(1) & ordered['complete_exchange']
    )
    ordered['_contextualized_exchange'] = (
        ordered['complete_exchange'] & ~pre_exchange_opener_row
    )

    ordered['analysis_turn'] = (
        ordered.groupby(group_col)['_contextualized_exchange'].cumsum()
        .where(ordered['_contextualized_exchange'], pd.NA)
        .astype('Int64')
    )
    df['analysis_turn'] = ordered.sort_index()['analysis_turn']

    n_opener_rows = int(pre_exchange_opener_row.sum())
    complete_count = int(df['complete_exchange'].sum())
    incomplete_count = int((~df['complete_exchange']).sum())
    print(
        "Exchange alignment metadata added: "
        f"{complete_count} complete exchanges, {incomplete_count} incomplete rows, "
        f"{n_opener_rows} opener rows marked as pre-exchange (analysis_turn=NA)"
    )

    return df


def build_long_format_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dyadic interaction rows into a long-format analysis export.

    Each interaction row becomes two contribution rows, one for each author slot.
    """
    required = [
        'conversation_id', 'condition', 'turn', 'analysis_turn',
        'author_1', 'author_2', 'author_1_type', 'author_2_type',
        'complete_exchange', 'starter_side', 'starter_type'
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for long-format export: {missing}")

    shared_columns = [
        'conversation_id', 'condition', 'turn', 'analysis_turn',
        'complete_exchange', 'starter_side', 'starter_type'
    ]

    frames = []
    slot_specs = [
        ('author_1', 'author_1_type', 'author_2_type'),
        ('author_2', 'author_2_type', 'author_1_type'),
    ]

    for speaker_slot, speaker_type_col, partner_type_col in slot_specs:
        frame = df[shared_columns].copy()
        frame['speaker_slot'] = speaker_slot
        frame['speaker_type'] = df[speaker_type_col]
        frame['speaker_is_starter'] = df['starter_side'].eq(speaker_slot)
        frame['partner_type'] = df[partner_type_col]
        frame['text'] = df[speaker_slot]
        frames.append(frame)

    df_long = pd.concat(frames, ignore_index=True)
    df_long = df_long.sort_values(
        ['conversation_id', 'turn', 'speaker_slot'],
        kind='stable'
    ).reset_index(drop=True)

    print(f"Built long-format analysis export with {len(df_long)} rows")

    return df_long


def keep_complete_conversations(
    df: pd.DataFrame,
    group_col: str = 'conversation_id',
    expected_length: Optional[int] = None
) -> pd.DataFrame:
    """
    Remove partial conversation fragments after row-level filtering.

    By default, the expected conversation length is inferred as the modal number
    of rows per conversation in the current DataFrame. This works well when a
    small number of corrupted or partially filtered stories remain alongside a
    dominant complete-story length.

    Args:
        df: Input DataFrame with repeated interaction rows per conversation
        group_col: Story identifier column
        expected_length: Required number of rows per conversation. If None,
            infer from the modal conversation length.

    Returns:
        DataFrame containing only complete conversations
    """
    if group_col not in df.columns:
        raise ValueError(f"Expected '{group_col}' column in DataFrame")

    df = df.copy()
    conversation_sizes = df.groupby(group_col).size()

    if conversation_sizes.empty:
        return df

    if expected_length is None:
        expected_length = int(conversation_sizes.mode().iloc[0])

    complete_ids = conversation_sizes[conversation_sizes == expected_length].index
    removed = conversation_sizes[conversation_sizes != expected_length]

    if len(removed) > 0:
        print(
            f"Dropping {len(removed)} incomplete conversation(s) with sizes "
            f"{removed.value_counts().sort_index().to_dict()} (expected {expected_length} rows)"
        )

    return df[df[group_col].isin(complete_ids)].copy()


def randomize_author_assignment(
    df: pd.DataFrame,
    group_col: str = 'conversation_id',
    seed: int = 42,
    swap_probability: float = 0.5
) -> pd.DataFrame:
    """
    Randomly swap author_1 and author_2 for each story to remove starter confound.
    
    In conditions where author_1 always starts (human-human, ai-ai), this introduces
    randomness matching the human-ai condition where starter was randomly assigned.
    
    When swapped:
    - author_1 <-> author_2 columns are swapped
    - starter column is updated to reflect who now has turn 1 content
    
    Args:
        df: DataFrame with author_1, author_2, conversation_id, starter columns
        group_col: Column to group stories by
        seed: Random seed for reproducibility
        swap_probability: Probability of swapping each story (default 0.5)
        
    Returns:
        DataFrame with randomized author assignment
    """
    import numpy as np
    
    df = df.copy()
    rng = np.random.default_rng(seed)
    
    # Get unique stories
    story_ids = df[group_col].unique()
    
    # Decide which stories to swap
    swap_mask = rng.random(len(story_ids)) < swap_probability
    stories_to_swap = set(story_ids[swap_mask])
    
    n_swapped = len(stories_to_swap)
    n_total = len(story_ids)
    print(f"Author randomization: swapping {n_swapped}/{n_total} stories ({100*n_swapped/n_total:.1f}%)")
    
    # Swap author columns for selected stories
    swap_rows = df[group_col].isin(stories_to_swap)
    
    # Swap author_1 and author_2
    df.loc[swap_rows, ['author_1', 'author_2']] = df.loc[swap_rows, ['author_2', 'author_1']].values
    
    # Update starter column: if we swapped, the original author_1 (now author_2) was the starter
    # So starter becomes 'author_2' for swapped stories
    if 'starter' in df.columns:
        # For swapped stories, flip the starter label
        original_starter = df.loc[swap_rows, 'starter'].copy()
        df.loc[swap_rows, 'starter'] = original_starter.map({
            'author_1': 'author_2',
            'author_2': 'author_1'
        })
    
    return df
