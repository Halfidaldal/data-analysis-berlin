"""
Embedding computation functions.

This module handles text embedding using various models:
- Jina embeddings (jinaai/jina-embeddings-v3)
- E5 embeddings (intfloat/multilingual-e5-large-instruct)
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
import gc
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def _sanitize_text(value) -> str:
    """Convert arbitrary cell values to safe text for embedding."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _sanitize_text_batch(batch: List) -> List[str]:
    """Sanitize a batch to a list of strings for SentenceTransformer.encode."""
    return [_sanitize_text(item) for item in batch]


def _is_float_indices_runtime_error(error: RuntimeError) -> bool:
    """Detect the known embedding-layer error when token indices are float tensors."""
    message = str(error)
    return (
        "Expected tensor for argument #1 'indices'" in message
        and "FloatTensor" in message
    )


def _encode_batch_with_fallback(
    model: SentenceTransformer,
    batch: List[str],
    task: Optional[str] = None,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Encode a batch and retry with forced integer token IDs on known model dtype bug."""
    try:
        if task and hasattr(model, 'encode') and 'task' in model.encode.__code__.co_varnames:
            return model.encode(
                batch,
                task=task,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
            )
        return model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
        )
    except RuntimeError as e:
        if not _is_float_indices_runtime_error(e):
            raise

        if not getattr(model, "_float_indices_fallback_warned", False):
            print("[WARNING] Caught float token index bug; retrying batches with integer input_ids.")
            setattr(model, "_float_indices_fallback_warned", True)

        features = model.tokenize(batch)
        if "input_ids" in features and isinstance(features["input_ids"], torch.Tensor):
            features["input_ids"] = features["input_ids"].long()

        target_device = next(model.parameters()).device
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.to(target_device)

        with torch.no_grad():
            out_features = model.forward(features)
            sentence_embeddings = out_features["sentence_embedding"]
            if normalize_embeddings:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.detach().cpu().numpy()


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def compute_embeddings_batch(
    texts: List[str],
    model_name: str = "jinaai/jina-embeddings-v3",
    batch_size: int = 32,
    active_dataset: Optional[str] = None,
    task: Optional[str] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Compute embeddings for a list of texts using a SentenceTransformer model.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the SentenceTransformer model
        batch_size: Number of texts to process at once
        task: Optional task hint for the model (e.g., "text-matching")
        device: Torch device to use (auto-detected if None)
        
    Returns:
        NumPy array of shape (len(texts), embedding_dim)
    """
    if device is None:
        device = get_device()
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=str(device), tokenizer_kwargs={"padding_side": "left", "trust_remote_code": True} if active_dataset=="TEXT" else {})
    
    texts = _sanitize_text_batch(texts)
    print(f"Computing embeddings for {len(texts)} texts (batch_size={batch_size})...")
    
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = _sanitize_text_batch(texts[i:i+batch_size])
        
        embeddings = _encode_batch_with_fallback(
            model,
            batch,
            task=task,
            normalize_embeddings=True,
        )
        
        all_embeddings.append(embeddings)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    all_embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return all_embeddings


def embed_story_columns(
    df: pd.DataFrame,
    text_columns: List[str],
    model_name: str = "jinaai/jina-embeddings-v3",
    batch_size: int = 32,
    active_dataset: Optional[str] = None,
    task: Optional[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Compute embeddings for multiple text columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        text_columns: List of column names to embed
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for embedding
        task: Optional task parameter for the model
        
    Returns:
        Tuple of:
        - DataFrame with new embedding columns (as lists of floats)
        - Dictionary mapping column names to numpy arrays of embeddings
    """
    df_out = df.copy()
    embeddings_dict = {}
    
    # Load model once and reuse for all columns
    device = get_device()
    print(f"Loading model: {model_name}")
    tokenizer_kwargs = {"padding_side": "left", "trust_remote_code": True} if active_dataset == "TEXT" else {}
    model = SentenceTransformer(model_name, trust_remote_code=True, device=str(device), tokenizer_kwargs=tokenizer_kwargs)
    
    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping")
            continue
        
        print(f"\n=== Embedding column: {col} ===")
        texts = _sanitize_text_batch(df[col].tolist())
        
        # Compute embeddings directly without loading model again
        print(f"Computing embeddings for {len(texts)} texts (batch_size={batch_size})...")
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch = _sanitize_text_batch(texts[i:i+batch_size])
            
            embeddings = _encode_batch_with_fallback(
                model,
                batch,
                task=task,
                normalize_embeddings=True,
            )
            
            all_embeddings.append(embeddings)
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        embeddings = np.vstack(all_embeddings)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Store as separate arrays
        embeddings_dict[col] = embeddings
        
        # Also store in DataFrame as list column (for parquet compatibility)
        df_out[f"{col}_embedding"] = embeddings.tolist()
    
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return df_out, embeddings_dict


def compute_story_embeddings_full_stories(
    df: pd.DataFrame,
    model_name: str = "jinaai/jina-embeddings-v3",
    active_dataset: Optional[str] = None,
    batch_size: int = 32
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute embeddings for full_story, full_author_1, and full_author_2 columns.
    
    This is the "field" embedding approach where we embed the complete texts.
    
    Args:
        df: DataFrame with 'full_story', 'full_author_1', 'full_author_2' columns
        model_name: Model to use for embeddings
        batch_size: Batch size for processing
        
    Returns:
        Tuple of:
        - DataFrame with embedding columns added
        - story_embeddings (np.ndarray)
        - author_1_embeddings (np.ndarray)
        - author_2_embeddings (np.ndarray)
    """
    columns_to_embed = ['full_story', 'full_author_1', 'full_author_2']
    
    df_embedded, embeddings_dict = embed_story_columns(
        df,
        text_columns=columns_to_embed,
        model_name=model_name,
        batch_size=batch_size,
        active_dataset=active_dataset
    )
    
    story_embeddings = embeddings_dict.get('full_story')
    author_1_embeddings = embeddings_dict.get('full_author_1')
    author_2_embeddings = embeddings_dict.get('full_author_2')
    
    return df_embedded, story_embeddings, author_1_embeddings, author_2_embeddings
