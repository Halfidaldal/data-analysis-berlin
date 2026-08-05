"""
Novelty, transience, and resonance computation using language models.
"""
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


_MISSING_TEXT_VALUES = {"", "nan", "none", "null"}


def _normalize_metric_text(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in _MISSING_TEXT_VALUES:
        return None
    return text


def _tokenize_metric_text(value, tokenizer):
    """Normalize scalar text-like values before tokenization."""
    text = _normalize_metric_text(value)
    if text is None:
        return []
    return tokenizer(text, add_special_tokens=False)["input_ids"]

def load_language_model(model_path, device=None):
    """
    Load a causal language model and tokenizer.
    Returns (tokenizer, model, device).
    """
    # We don’t actually use `device` directly anymore; let device_map decide.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading language model from {model_path} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Use a lighter dtype on GPU and let HF/accelerate place layers
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",        # shard/put on GPU/CPU as needed
        low_cpu_mem_usage=True,   # stream weights instead of loading all at once
    )
    model.eval()

    # Infer “device” from model params (first parameter’s device)
    device = next(model.parameters()).device
    return tokenizer, model, device

def calc_sentence_surprisal(context_ids, target_ids, model, window_size=None):
    """
    Compute average surprisal (bits/token) and total surprisal (bits)
    for target tokens given context, using dynamic context window limits.
    """
    # Dynamically extract model's maximum context length, defaulting to 128k for Maverick if undefined
    max_context = getattr(model.config, "max_position_embeddings", 131072) 
    
    # Define how much history we can safely keep (leaving room for the target sentence)
    prediction_history_limit = max_context - len(target_ids) - 1
    
    # Truncate context to fit within the model's actual limits rather than a hardcoded 1024
    context_ids = context_ids[-prediction_history_limit:]
    combined_ids = context_ids + target_ids
    
    if len(combined_ids) < 2 or not target_ids:
        return 0.0, 0.0
    
    # Convert to tensors and explicitly create an attention mask
    input_tensor = torch.tensor([combined_ids], device=model.device)
    attention_mask = torch.ones_like(input_tensor, device=model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
    
    logits = outputs.logits[:, :-1, :]  # Distributions for each next token
    
    total_nll = 0.0
    for idx, token_id in enumerate(target_ids):
        pos = len(context_ids) + idx - 1
        if pos >= logits.shape[1]:
             break 
        dist = logits[0, pos]
        log_probs = torch.nn.functional.log_softmax(dist, dim=-1)
        log2p = log_probs[token_id] / math.log(2)
        total_nll += -log2p.item()
    
    avg_nll = total_nll / len(target_ids)
    return avg_nll, total_nll


def _row_chronological_slot_order(row, has_condition, has_starter):
    """
    Chronological order of the two slots within a single row.

    The dataframe stores each interaction as a (author_1, author_2) pair, but
    the chronological order of those two slots within a row is not always
    author_1 -> author_2:

    - In `human-ai`, every complete row pairs (human input, AI reply) with
      the human in slot author_1, so chronological order is always
      author_1 -> author_2. (AI primers in AI-starter stories live in
      incomplete rows that are excluded via complete_exchange == False.)
    - In `human-human` and `ai-ai`, `randomize_author_assignment` swaps the
      column labels for ~50% of stories; for those stories starter_side
      becomes 'author_2' and every row is chronologically author_2 -> author_1.
      Stories left unswapped keep starter_side == 'author_1' and chronology
      author_1 -> author_2.

    Returns the tuple of slot names in chronological order.
    """
    condition = row.get("condition") if has_condition else None
    starter = row.get("starter_side") if has_starter else None
    if condition == "human-ai":
        return ("author_1", "author_2")
    if starter == "author_2":
        return ("author_2", "author_1")
    return ("author_1", "author_2")


def compute_novelty_scores(df, tokenizer, model, window_size=128):
    """
    Compute novelty (surprise) scores for author_1 and author_2 utterances.

    Uses standardized column names: author_1, author_2.

    Novelty = how surprising this utterance is given prior context.

    Within each row the two slots are scored in chronological order, derived
    from `condition` and `starter_side` (see
    `_row_chronological_slot_order`). Treating slot order as chronological
    order silently mis-aligned the context buffer against the actual prior
    dialog for HH/AA stories randomized to starter_side == 'author_2', so
    half of HH and half of AA were scored against the wrong past. Scoring
    in chronological order is required for the per-slot surprise values to
    be comparable across conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with author_1 and author_2 text columns (and ideally
        `condition` and `starter_side` so chronology can be derived).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer
    model : transformers.PreTrainedModel
        Causal language model
    window_size : int
        Context window size

    Returns
    -------
    pd.DataFrame
        DataFrame with added novelty columns
    """
    df = df.copy()
    sort_columns = [col for col in ["conversation_id", "turn", "timestamp"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    author_1_text = df["author_1"].apply(_normalize_metric_text)
    author_2_text = df["author_2"].apply(_normalize_metric_text)
    author_1_has_text = author_1_text.notna()
    author_2_has_text = author_2_text.notna()

    df["author_1_ids"] = author_1_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))
    df["author_2_ids"] = author_2_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))

    # Identify a safe start token for unconditional probability
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    bos_ids = [bos_token_id] if bos_token_id is not None else []
    anchor_text = "Speaker:"
    anchor_ids = tokenizer(anchor_text, add_special_tokens=False)["input_ids"]
    # base_context = unconditional baseline (BOS + anchor); reused across all turns
    base_context = bos_ids + anchor_ids
    # context_buffer grows with story tokens and resets per conversation
    context_buffer = bos_ids + anchor_ids

    has_condition = "condition" in df.columns
    has_starter = "starter_side" in df.columns

    last_client = None
    author_1_novelty, author_1_raw, author_1_entropy = [], [], []
    author_2_novelty, author_2_raw, author_2_entropy = [], [], []

    for pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Computing novelty")):
        client = row.get("conversation_id", None)

        # Reset context at new session
        if last_client is None or (client is not None and client != last_client):
            context_buffer = bos_ids + anchor_ids
            last_client = client

        slot_order = _row_chronological_slot_order(row, has_condition, has_starter)

        row_results = {
            "author_1": (np.nan, np.nan, np.nan),
            "author_2": (np.nan, np.nan, np.nan),
        }

        for slot in slot_order:
            slot_ids = row[f"{slot}_ids"]
            slot_has_text = (
                author_1_has_text.iloc[pos]
                if slot == "author_1"
                else author_2_has_text.iloc[pos]
            )
            if slot_has_text and slot_ids:
                avg_s, total_s = calc_sentence_surprisal(context_buffer, slot_ids, model, window_size)
                avg_base, _ = calc_sentence_surprisal(base_context, slot_ids, model, window_size)
                row_results[slot] = (avg_s - avg_base, avg_s, total_s)
                context_buffer.extend(slot_ids)

        a1_novelty, a1_raw, a1_total = row_results["author_1"]
        a2_novelty, a2_raw, a2_total = row_results["author_2"]
        author_1_novelty.append(a1_novelty)
        author_1_raw.append(a1_raw)
        author_1_entropy.append(a1_total)
        author_2_novelty.append(a2_novelty)
        author_2_raw.append(a2_raw)
        author_2_entropy.append(a2_total)

    df["author_1_surprise"] = author_1_novelty
    df["author_2_surprise"] = author_2_novelty
    df["author_1_surprise_raw"] = author_1_raw
    df["author_2_surprise_raw"] = author_2_raw

    df["author_1_entropy"] = author_1_entropy
    df["author_2_entropy"] = author_2_entropy

    return df


def compute_transience_scores(df, tokenizer, model, window_size=40):
    """
    Compute transience scores for author_1 and author_2 utterances using a
    counterfactual / ablation formulation:

        Transience_t = S(Future | Past + s_t) - S(Future | Past)

    This preserves the existing sign convention used downstream. Negative
    values mean s_t made the immediate next turn less surprising than the past
    alone; values near zero mean the past alone was already as informative as
    past + s_t.

    The future is the immediate next chronological turn, not necessarily the
    other slot in author_1 -> author_2 order. In HH/AA stories randomized to
    starter_side == 'author_2', each row is chronological author_2 -> author_1;
    in HA complete exchanges remain author_1 -> author_2.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (must already have author_1_ids and author_2_ids columns
        from compute_novelty_scores; if not, they are tokenized here).
    tokenizer : transformers.PreTrainedTokenizer
    model : transformers.PreTrainedModel
    window_size : int
        Kept for backward compatibility; not used here.

    Returns
    -------
    pd.DataFrame
        DataFrame with added transience columns:
            author_1_transience, author_2_transience  (counterfactual)
            author_1_transience_raw, author_2_transience_raw (S(F | Past + s_t))
            author_1_transience_baseline, author_2_transience_baseline (S(F | Past))
    """
    df = df.copy()
    sort_columns = [col for col in ["conversation_id", "turn", "timestamp"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    author_1_text = df["author_1"].apply(_normalize_metric_text)
    author_2_text = df["author_2"].apply(_normalize_metric_text)
    author_1_has_text = author_1_text.notna()
    author_2_has_text = author_2_text.notna()

    # Tokenize if not already done by compute_novelty_scores
    if "author_1_ids" not in df.columns:
        df["author_1_ids"] = author_1_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))
    if "author_2_ids" not in df.columns:
        df["author_2_ids"] = author_2_text.apply(lambda txt: _tokenize_metric_text(txt, tokenizer))

    # Anchor / BOS setup, mirroring compute_novelty_scores so the past buffer
    # is constructed identically across the two metrics.
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    bos_ids = [bos_token_id] if bos_token_id is not None else []
    anchor_text = "Speaker:"
    anchor_ids = tokenizer(anchor_text, add_special_tokens=False)["input_ids"]
    base_context = bos_ids + anchor_ids

    # We rebuild the past buffer turn by turn, exactly as in novelty.
    # This way Past_t means the same thing in both metrics.
    context_buffer = list(base_context)
    last_client = None

    has_condition = "condition" in df.columns
    has_starter = "starter_side" in df.columns

    def _slot_has_text(slot, pos):
        return author_1_has_text.iloc[pos] if slot == "author_1" else author_2_has_text.iloc[pos]

    def _gather_immediate_future_ids(pos, frame, current_conversation_id, slot, slot_order):
        slot_index = slot_order.index(slot)
        if slot_index + 1 < len(slot_order):
            return frame.iloc[pos][f"{slot_order[slot_index + 1]}_ids"]

        next_pos = pos + 1
        if next_pos >= len(frame):
            return []
        next_row = frame.iloc[next_pos]
        if next_row.get("conversation_id", None) != current_conversation_id:
            return []
        next_slot = _row_chronological_slot_order(next_row, has_condition, has_starter)[0]
        return next_row[f"{next_slot}_ids"]

    a1_trans, a1_with, a1_without = [], [], []
    a2_trans, a2_with, a2_without = [], [], []

    for pos in tqdm(range(len(df)), total=len(df), desc="Computing transience (counterfactual)"):
        row = df.iloc[pos]
        client = row.get("conversation_id", None)

        # Reset past buffer at session boundaries, same logic as novelty.
        if last_client is None or (client is not None and client != last_client):
            context_buffer = list(base_context)
            last_client = client

        slot_order = _row_chronological_slot_order(row, has_condition, has_starter)
        row_results = {
            "author_1": (np.nan, np.nan, np.nan),
            "author_2": (np.nan, np.nan, np.nan),
        }

        for slot in slot_order:
            slot_ids = row[f"{slot}_ids"]
            future_ids = _gather_immediate_future_ids(pos, df, client, slot, slot_order)

            if _slot_has_text(slot, pos) and slot_ids and future_ids:
                past = list(context_buffer)
                past_plus_st = past + slot_ids

                avg_with, _ = calc_sentence_surprisal(past_plus_st, future_ids, model, window_size)
                avg_without, _ = calc_sentence_surprisal(past, future_ids, model, window_size)

                # Preserve the existing sign convention used by downstream Rmds:
                # S(F | Past + s_t) - S(F | Past).
                row_results[slot] = (avg_with - avg_without, avg_with, avg_without)

            if _slot_has_text(slot, pos) and slot_ids:
                context_buffer.extend(slot_ids)

        a1_result, a2_result = row_results["author_1"], row_results["author_2"]
        a1_trans.append(a1_result[0])
        a1_with.append(a1_result[1])
        a1_without.append(a1_result[2])
        a2_trans.append(a2_result[0])
        a2_with.append(a2_result[1])
        a2_without.append(a2_result[2])

    df["author_1_transience"] = a1_trans
    df["author_2_transience"] = a2_trans
    df["author_1_transience_raw"] = a1_with         # S(F | Past + s_t)
    df["author_2_transience_raw"] = a2_with
    df["author_1_transience_baseline"] = a1_without # S(F | Past)
    df["author_2_transience_baseline"] = a2_without

    return df
