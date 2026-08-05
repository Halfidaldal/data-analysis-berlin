"""
Semantic exploration metrics using binned embedding distances.

Computes non-overlapping semantic jumps at different timescales to measure
how much the narrative explores semantic space over time.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_embedding(x):
    """
    Parse embedding from various formats (string, list, array, None).
    
    Parameters
    ----------
    x : various
        Embedding in any format
        
    Returns
    -------
    np.ndarray or None
        Parsed embedding as 1D array, or None if invalid
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    
    if isinstance(x, str):
        try:
            arr = np.fromstring(x.strip('[]'), sep=',')
        except Exception:
            return None
    else:
        try:
            arr = np.array(x)
        except Exception:
            return None
    
    return arr if (isinstance(arr, np.ndarray) and arr.ndim == 1) else None


def interleave_and_align(user_embs, ai_embs):
    """
    Interleave slot-ordered embeddings across complete exchanges.
    
    Parameters
    ----------
    user_embs : np.ndarray
        User embeddings, shape (T, D)
    ai_embs : np.ndarray
        AI embeddings, shape (T, D)
        
    Returns
    -------
    np.ndarray
        Interleaved embeddings, shape (2*T, D)
    """
    T, D = user_embs.shape
    E = np.empty((2 * T, D), dtype=float)
    E[0::2] = user_embs
    E[1::2] = ai_embs
    return E


def _chronological_slot_order(metadata: dict) -> tuple[str, str]:
    condition = metadata.get("condition")
    starter = metadata.get("starter_side")
    if condition == "human-ai":
        return ("author_1", "author_2")
    if starter == "author_2":
        return ("author_2", "author_1")
    return ("author_1", "author_2")


def interleave_embeddings_by_chronology(author_1_embs, author_2_embs, metadata: dict):
    """Interleave row-paired embeddings in true within-row chronological order."""
    slot_arrays = {
        "author_1": author_1_embs,
        "author_2": author_2_embs,
    }
    first_slot, second_slot = _chronological_slot_order(metadata)
    return interleave_and_align(slot_arrays[first_slot], slot_arrays[second_slot])


def _sort_story_group(grp: pd.DataFrame) -> pd.DataFrame:
    sort_columns = []
    if "analysis_turn" in grp.columns and grp["analysis_turn"].notna().any():
        sort_columns.append("analysis_turn")
    if "turn" in grp.columns:
        sort_columns.append("turn")
    if sort_columns:
        return grp.sort_values(sort_columns)
    return grp


def _conversation_metadata(grp: pd.DataFrame) -> dict:
    metadata_columns = [
        "condition",
        "starter",
        "starter_side",
        "starter_type",
        "llm_type",
        "author_1_type",
        "author_2_type",
    ]
    metadata = {}
    for col in metadata_columns:
        metadata[col] = grp[col].iloc[0] if col in grp.columns else None
    if "analysis_turn" in grp.columns:
        metadata["n_complete_exchanges"] = int(grp["analysis_turn"].notna().sum())
    elif "complete_exchange" in grp.columns:
        metadata["n_complete_exchanges"] = int(grp["complete_exchange"].fillna(False).sum())
    else:
        metadata["n_complete_exchanges"] = len(grp)
    return metadata


def _agent_record_metadata(conversation_id, agent_name: str, metadata: dict) -> dict:
    base = {
        "conversation_id": conversation_id,
        "agent": agent_name,
        "condition": metadata.get("condition"),
        "starter": metadata.get("starter"),
        "starter_side": metadata.get("starter_side"),
        "starter_type": metadata.get("starter_type"),
        "llm_type": metadata.get("llm_type"),
        "n_complete_exchanges": metadata.get("n_complete_exchanges"),
    }

    if agent_name == "interleaved":
        base.update({
            "speaker_slot": pd.NA,
            "speaker_type": pd.NA,
            "partner_type": pd.NA,
            "speaker_is_starter": pd.NA,
        })
        return base

    other_slot = "author_2" if agent_name == "author_1" else "author_1"
    base.update({
        "speaker_slot": agent_name,
        "speaker_type": metadata.get(f"{agent_name}_type"),
        "partner_type": metadata.get(f"{other_slot}_type"),
        "speaker_is_starter": metadata.get("starter_side") == agent_name if metadata.get("starter_side") is not None else pd.NA,
    })
    return base


def compute_nonoverlap_distances(embeddings, window_length):
    """
    Compute cosine distances between centroids of consecutive non-overlapping windows.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Sequence of embeddings, shape (N, D)
    window_length : int
        Number of embeddings in each window
        
    Returns
    -------
    np.ndarray
        Cosine distances between adjacent window centroids
    """
    N, D = embeddings.shape
    n_bins = N // window_length
    
    if n_bins < 2:
        return np.array([])
    
    # Reshape into non-overlapping bins
    bins = embeddings[:n_bins * window_length].reshape(n_bins, window_length, D)
    
    # Compute centroids
    centroids = bins.mean(axis=1)
    
    # Normalize
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / norms
    
    # Cosine similarity between adjacent centroids
    similarities = np.einsum('ij,ij->i', centroids_norm[:-1], centroids_norm[1:])
    similarities = np.clip(similarities, -1.0, 1.0)
    
    # Convert to distance
    distances = 1.0 - similarities
    
    return distances


def compute_semantic_exploration_metrics(
    df,
    author_1_embedding_col="author_1_embedding",
    author_2_embedding_col="author_2_embedding",
    max_k=10
):
    """
    Compute semantic exploration metrics at multiple timescales.
    
    Uses standardized column names: author_1_embedding, author_2_embedding.
    
    For each story, computes non-overlapping semantic jumps for window sizes
    from 2 to max_k+1 turns. Larger k = longer timescale.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with embedding columns
    author_1_embedding_col : str
        Name of author_1 embedding column
    author_2_embedding_col : str
        Name of author_2 embedding column
    max_k : int
        Maximum bin-width parameter (window_length = k + 1)
        
    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        - conversation_id
        - agent ('author_1' or 'author_2' for single-agent metrics, 'interleaved' for combined)
        - k (bin-width parameter)
        - bin_index (which consecutive window pair)
        - distance (cosine distance between centroids)
    """
    # Parse embeddings
    df = df.copy()
    df['author_1_emb'] = df[author_1_embedding_col].apply(parse_embedding)
    df['author_2_emb'] = df[author_2_embedding_col].apply(parse_embedding)
    
    # Filter out invalid embeddings
    before = len(df)
    df = df[df['author_1_emb'].notnull() & df['author_2_emb'].notnull()].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with invalid embeddings.")
    
    # Group by story
    story_groups = df.groupby(['conversation_id'])
    
    records = []
    for conversation_id, grp in tqdm(story_groups, desc="Computing semantic exploration"):
        grp = _sort_story_group(grp)
        author_1_list = grp['author_1_emb'].tolist()
        author_2_list = grp['author_2_emb'].tolist()
        metadata = _conversation_metadata(grp)
        
        try:
            author_1_embs = np.vstack(author_1_list)
            author_2_embs = np.vstack(author_2_list)
        except Exception:
            continue
        
        if author_1_embs.shape != author_2_embs.shape:
            continue
        
        # Interleave and align in true chronological order.
        E = interleave_embeddings_by_chronology(author_1_embs, author_2_embs, metadata)
        
        # Compute distances at different window sizes
        for k in range(1, max_k + 1):
            window_length = k + 1
            distances = compute_nonoverlap_distances(E, window_length)
            
            for idx, dist in enumerate(distances):
                record = _agent_record_metadata(conversation_id, "interleaved", metadata)
                record.update({
                    'k': k,
                    'bin_index': idx,
                    'distance': float(dist),
                })
                records.append(record)
                
        for agent_name, embs in (("author_1", author_1_embs), ("author_2", author_2_embs)):
            for k in range(1, max_k + 1):
                window_length = k + 1
                distances = compute_nonoverlap_distances(embs, window_length)
                
                for idx, dist in enumerate(distances):
                    record = _agent_record_metadata(conversation_id, agent_name, metadata)
                    record.update({
                        'k': k,
                        'bin_index': idx,
                        'distance': float(dist),
                    })
                    records.append(record)
    
    return pd.DataFrame.from_records(records)


def compute_ai_ai_semantic_exploration(
    df,
    ai1_embedding_col="ai1_embedding",
    ai2_embedding_col="ai2_embedding",
    max_k=10
):
    """
    Compute semantic exploration metrics for AI-AI baseline.
    
    Same as compute_semantic_exploration_metrics but for AI1 and AI2 columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with AI-AI embedding columns
    ai1_embedding_col : str
        Name of AI1 embedding column
    ai2_embedding_col : str
        Name of AI2 embedding column
    max_k : int
        Maximum bin-width parameter
        
    Returns
    -------
    pd.DataFrame
        Long-format dataframe with semantic exploration metrics
    """
    # Parse embeddings
    df = df.copy()
    df['user_emb'] = df[ai1_embedding_col].apply(parse_embedding)
    df['ai_emb'] = df[ai2_embedding_col].apply(parse_embedding)
    
    # Filter out invalid embeddings
    before = len(df)
    df = df[df['user_emb'].notnull() & df['ai_emb'].notnull()].reset_index(drop=True)
    print(f"Dropped {before - len(df)} rows with invalid embeddings.")
    
    # Group by story
    story_groups = df.groupby(['conversation_id'])
    
    records = []
    for conversation_id, grp in tqdm(story_groups, desc="Computing AI-AI semantic exploration"):
        grp = _sort_story_group(grp)
        user_list = grp['user_emb'].tolist()
        ai_list = grp['ai_emb'].tolist()
        metadata = _conversation_metadata(grp)
        
        try:
            user_embs = np.vstack(user_list)
            ai_embs = np.vstack(ai_list)
        except Exception:
            continue
        
        if user_embs.shape != ai_embs.shape:
            continue
        
        # Interleave and align in true chronological order when metadata is available.
        E = interleave_embeddings_by_chronology(user_embs, ai_embs, metadata)
        
        # Compute distances at different window sizes
        for k in range(1, max_k + 1):
            window_length = k + 1
            distances = compute_nonoverlap_distances(E, window_length)
            
            for idx, dist in enumerate(distances):
                records.append({
                    'conversation_id': conversation_id,
                    'k': k,
                    'bin_index': idx,
                    'distance': float(dist)
                })
    
    return pd.DataFrame.from_records(records)


def compute_lag_distances(embeddings, max_lag=None):
    """
    Compute average cosine distance between embeddings separated by lag k.
    
    Instead of binning, this uses all pairs separated by k steps.
    This is more robust for short time series (N=10).
    
    Parameters
    ----------
    embeddings : np.ndarray
        Sequence of embeddings, shape (N, D)
    max_lag : int, optional
        Maximum lag to compute. If None, uses N-1.
        
    Returns
    -------
    list of dict
        List of {'k': lag, 'distance': avg_dist}
    """
    N, D = embeddings.shape
    if max_lag is None:
        max_lag = N - 1
    
    max_lag = min(max_lag, N - 1)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1e-10
    embeddings_norm = embeddings / norms
    
    results = []
    for k in range(1, max_lag + 1):
        # Vectorized cosine distance for lag k
        # A: embeddings[:-k], B: embeddings[k:]
        # Dot product of normalized vectors = cosine similarity
        sims = np.sum(embeddings_norm[:-k] * embeddings_norm[k:], axis=1)
        # Clip for numerical stability
        sims = np.clip(sims, -1.0, 1.0)
        dists = 1.0 - sims
        
        results.append({
            'k': k,
            'distance': float(np.mean(dists)),
            'std_distance': float(np.std(dists)),
            'n_pairs': len(dists)
        })
        
    return results


def compute_lag_exploration_metrics(
    df,
    user_embedding_col="author_1_embedding",
    ai_embedding_col="author_2_embedding",
    max_lag=None
):
    """
    Compute semantic exploration using lag-based distances.
    
    Uses standardized column names: author_1_embedding, author_2_embedding.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    user_embedding_col : str
        Name of author_1 embedding column
    ai_embedding_col : str
        Name of author_2 embedding column
    max_lag : int
        Maximum lag to compute
        
    Returns
    -------
    pd.DataFrame
        Long-format dataframe with lag-based metrics
    """
    # Parse embeddings
    df = df.copy()
    df['author_1_emb'] = df[user_embedding_col].apply(parse_embedding)
    df['author_2_emb'] = df[ai_embedding_col].apply(parse_embedding)
    
    # Filter out invalid embeddings
    df = df[df['author_1_emb'].notnull() & df['author_2_emb'].notnull()].reset_index(drop=True)
    
    # Group by story
    story_groups = df.groupby(['conversation_id'])
    
    records = []
    for conversation_id, grp in tqdm(story_groups, desc="Computing lag exploration"):
        grp = _sort_story_group(grp)
        author_1_list = grp['author_1_emb'].tolist()
        author_2_list = grp['author_2_emb'].tolist()
        metadata = _conversation_metadata(grp)
        
        try:
            author_1_embs = np.vstack(author_1_list)
            author_2_embs = np.vstack(author_2_list)
        except Exception:
            continue
            
        # Interleaved
        if author_1_embs.shape == author_2_embs.shape:
            E = interleave_embeddings_by_chronology(author_1_embs, author_2_embs, metadata)
            lag_results = compute_lag_distances(E, max_lag)
            for res in lag_results:
                res.update(_agent_record_metadata(conversation_id, "interleaved", metadata))
                records.append(res)
                
        # Individual agents
        for agent_name, embs in (("author_1", author_1_embs), ("author_2", author_2_embs)):
            lag_results = compute_lag_distances(embs, max_lag)
            for res in lag_results:
                res.update(_agent_record_metadata(conversation_id, agent_name, metadata))
                records.append(res)
                
    return pd.DataFrame.from_records(records)
