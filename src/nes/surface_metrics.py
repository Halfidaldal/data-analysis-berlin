import pandas as pd 
import textdescriptives as td
import spacy


_MISSING_TEXT_VALUES = {"", "nan", "none", "null"}


def _normalize_metric_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in _MISSING_TEXT_VALUES else text


def _is_substantive_text(series: pd.Series, min_chars: int = 2) -> pd.Series:
    normalized = series.apply(_normalize_metric_text)
    return normalized.str.len().ge(min_chars)


def _attach_slot_metadata(metrics_df: pd.DataFrame, df: pd.DataFrame, slot: str) -> pd.DataFrame:
    other_slot = "author_2" if slot == "author_1" else "author_1"
    metadata_columns = [
        "conversation_id",
        "condition",
        "turn",
        "analysis_turn",
        "interaction_count",
        "starter",
        "starter_side",
        "starter_type",
        "complete_exchange",
        "llm_type",
        "timestamp",
        "respondent_id",
        "respondent_id_u1",
        "respondent_id_u2",
        "model_id",
    ]

    metrics_df["type"] = slot
    metrics_df["speaker_slot"] = slot

    for col in metadata_columns:
        if col in df.columns:
            metrics_df[col] = df[col]

    speaker_type_col = f"{slot}_type"
    partner_type_col = f"{other_slot}_type"
    if speaker_type_col in df.columns:
        metrics_df["speaker_type"] = df[speaker_type_col]
    if partner_type_col in df.columns:
        metrics_df["partner_type"] = df[partner_type_col]

    starter_col = "starter_side" if "starter_side" in df.columns else ("starter" if "starter" in df.columns else None)
    if starter_col is not None:
        metrics_df["speaker_is_starter"] = df[starter_col].eq(slot)

    return metrics_df


def _mask_non_substantive_rows(metrics_df: pd.DataFrame, substantive_mask: pd.Series) -> pd.DataFrame:
    missing_mask = ~substantive_mask.fillna(False)
    if not missing_mask.any():
        return metrics_df

    metadata_columns = {
        "type",
        "speaker_slot",
        "speaker_type",
        "partner_type",
        "speaker_is_starter",
        "conversation_id",
        "condition",
        "turn",
        "analysis_turn",
        "interaction_count",
        "starter",
        "starter_side",
        "starter_type",
        "complete_exchange",
        "llm_type",
        "timestamp",
        "respondent_id",
        "respondent_id_u1",
        "respondent_id_u2",
        "model_id",
    }
    metric_columns = [col for col in metrics_df.columns if col not in metadata_columns]

    # textdescriptives emits a mix of float, int, bool, and object columns.
    # Newer pandas versions reject assigning pd.NA into numpy bool/int columns,
    # so convert metric columns to nullable dtypes before masking.
    for col in metric_columns:
        dtype = metrics_df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            metrics_df[col] = metrics_df[col].astype("boolean")
        elif pd.api.types.is_integer_dtype(dtype):
            metrics_df[col] = metrics_df[col].astype("Int64")
        elif pd.api.types.is_float_dtype(dtype):
            metrics_df[col] = metrics_df[col].astype("Float64")
        elif pd.api.types.is_categorical_dtype(dtype):
            metrics_df[col] = metrics_df[col].astype("object")

    metrics_df.loc[missing_mask, metric_columns] = pd.NA
    return metrics_df

def get_descriptive_metrics_dual_full_long(
        df: pd.DataFrame,
        author_1_col: str = "full_author_1_dot",
        author_2_col: str = "full_author_2_dot",
        spacy_mdl: str = "en_core_web_md",
        batch_size: int = 10,
        n_process: int = 5):
    """
    Compute text descriptives for full story texts.
    
    Uses standardized column names: author_1, author_2.
    """
    print(f"[INFO:] Loading spaCy model '{spacy_mdl}'...")
    nlp = spacy.load(spacy_mdl)
    try:
        nlp.add_pipe("textdescriptives/all")
    except Exception as e:
        if "typing.Tuple" in str(e) or "textdescriptives/quality" in str(e):
            print("[WARNING:] Caught config validation error with 'textdescriptives/all'. Adding pipes manually...")
            for comp in [
                "textdescriptives/descriptive_stats",
                "textdescriptives/readability",
                "textdescriptives/dependency_distance",
                "textdescriptives/pos_proportions",
                "textdescriptives/coherence",
                "textdescriptives/information_theory",
            ]:
                if not nlp.has_pipe(comp):
                    nlp.add_pipe(comp)
            if not nlp.has_pipe("textdescriptives/quality"):
                nlp.add_pipe(
                    "textdescriptives/quality",
                    config={
                        "top_ngram_range": (2, 4),
                        "duplicate_n_gram_fraction_range": (5, 10)
                    }
                )
        else:
            raise e

    # ----- AUTHOR 1 -----
    print(f"[INFO:] Extracting Author 1 metrics...")
    author_1_substantive = _is_substantive_text(df[author_1_col])
    docs_author_1 = nlp.pipe(df[author_1_col].apply(_normalize_metric_text), batch_size=batch_size, n_process=n_process)
    author_1_metrics = td.extract_df(docs_author_1, include_text=True)
    author_1_metrics.index = df.index
    author_1_metrics = _attach_slot_metadata(author_1_metrics, df, "author_1")
    author_1_metrics = _mask_non_substantive_rows(author_1_metrics, author_1_substantive)

    # ----- AUTHOR 2 -----
    print(f"[INFO:] Extracting Author 2 metrics...")
    author_2_substantive = _is_substantive_text(df[author_2_col])
    docs_author_2 = nlp.pipe(df[author_2_col].apply(_normalize_metric_text), batch_size=batch_size, n_process=n_process)
    author_2_metrics = td.extract_df(docs_author_2, include_text=True)
    author_2_metrics.index = df.index
    author_2_metrics = _attach_slot_metadata(author_2_metrics, df, "author_2")
    author_2_metrics = _mask_non_substantive_rows(author_2_metrics, author_2_substantive)

    # ----- STACK LONG -----
    print("[INFO:] Combining metrics (long format)...")
    metrics_long = pd.concat([author_1_metrics, author_2_metrics], axis=0)

    return metrics_long

def get_descriptive_metrics_dual_inter_long(
        df: pd.DataFrame,
        author_1_col: str = "author_1",
        author_2_col: str = "author_2",
        spacy_mdl: str = "en_core_web_md",
        batch_size: int = 10,
        n_process: int = 5):
    """
    Compute text descriptives for interaction-level texts.
    
    Uses standardized column names: author_1, author_2.
    """
    print(f"[INFO:] Loading spaCy model '{spacy_mdl}'...")
    nlp = spacy.load(spacy_mdl)
    try:
        nlp.add_pipe("textdescriptives/all")
    except Exception as e:
        if "typing.Tuple" in str(e) or "textdescriptives/quality" in str(e):
            print("[WARNING:] Caught config validation error with 'textdescriptives/all'. Adding pipes manually...")
            for comp in [
                "textdescriptives/descriptive_stats",
                "textdescriptives/readability",
                "textdescriptives/dependency_distance",
                "textdescriptives/pos_proportions",
                "textdescriptives/coherence",
                "textdescriptives/information_theory",
            ]:
                if not nlp.has_pipe(comp):
                    nlp.add_pipe(comp)
            if not nlp.has_pipe("textdescriptives/quality"):
                nlp.add_pipe(
                    "textdescriptives/quality",
                    config={
                        "top_ngram_range": (2, 4),
                        "duplicate_n_gram_fraction_range": (5, 10)
                    }
                )
        else:
            raise e
    
    author_1_substantive = _is_substantive_text(df[author_1_col])
    author_2_substantive = _is_substantive_text(df[author_2_col])
    author_1_text = df[author_1_col].apply(_normalize_metric_text)
    author_2_text = df[author_2_col].apply(_normalize_metric_text)

    # ----- AUTHOR 1 -----
    print(f"[INFO:] Extracting Author 1 metrics...")
    
    docs_author_1 = nlp.pipe(author_1_text, batch_size=batch_size, n_process=n_process)
    author_1_metrics = td.extract_df(docs_author_1, include_text=True)
    author_1_metrics.index = df.index
    author_1_metrics = _attach_slot_metadata(author_1_metrics, df, "author_1")
    author_1_metrics = _mask_non_substantive_rows(author_1_metrics, author_1_substantive)

    # ----- AUTHOR 2 -----
    print(f"[INFO:] Extracting Author 2 metrics...")
    docs_author_2 = nlp.pipe(author_2_text, batch_size=batch_size, n_process=n_process)
    author_2_metrics = td.extract_df(docs_author_2, include_text=True)
    author_2_metrics.index = df.index
    author_2_metrics = _attach_slot_metadata(author_2_metrics, df, "author_2")
    author_2_metrics = _mask_non_substantive_rows(author_2_metrics, author_2_substantive)

    # ----- STACK LONG -----
    print("[INFO:] Combining metrics (long format)...")
    metrics_long = pd.concat([author_1_metrics, author_2_metrics], axis=0)
    metrics_long = metrics_long.reset_index(drop=True)

    return metrics_long
