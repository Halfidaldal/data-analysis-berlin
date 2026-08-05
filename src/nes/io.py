"""
I/O utilities for loading and saving data.

This module provides standardized functions for reading and writing
data files in various formats (CSV, Parquet, NumPy arrays).

The active experiment is determined by `active_experiment` in config.yaml.
"""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np
import yaml


# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = get_project_root() / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_active_experiment() -> str:
    """Get the active experiment name from config."""
    config = load_config()
    return config.get('active_experiment', 'human-ai')


def get_experiment_config(experiment: Optional[str] = None) -> dict:
    """
    Get configuration for a specific experiment.
    
    Args:
        experiment: Experiment name ('human-ai' or 'human-human'). 
                   If None, uses active_experiment from config.
    
    Returns:
        Dict with experiment-specific configuration
    """
    config = load_config()
    if experiment is None:
        experiment = config.get('active_experiment', 'human-ai')
    
    if experiment not in config.get('experiments', {}):
        raise ValueError(f"Unknown experiment: {experiment}. Must be one of {list(config['experiments'].keys())}")
    
    return config['experiments'][experiment]


def get_shared_config() -> dict:
    """Get shared configuration that applies to all experiments."""
    config = load_config()
    return config.get('shared', {})


def get_data_path(stage: str = "processed", experiment: Optional[str] = None) -> Path:
    """
    Get the path to a data directory for an experiment.
    
    Args:
        stage: One of 'raw', 'interim', 'processed'
        experiment: Experiment name. If None, uses active_experiment from config.
        
    Returns:
        Path object to the data directory
    """
    exp_config = get_experiment_config(experiment)
    path_key = f"{stage}_dir"
    
    if path_key not in exp_config:
        raise ValueError(f"Unknown stage: {stage}. Must be one of ['raw', 'interim', 'processed']")
    
    return PROJECT_ROOT / exp_config[path_key]


def load_csv(filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load a CSV file from a data directory.
    
    Args:
        filename: Name of the file (e.g., 'stories.csv')
        stage: Which data directory to load from ('raw', 'interim', 'processed')
        experiment: Experiment name. If None, uses active_experiment from config.
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with the loaded data
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Loading CSV from: {path}")
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> None:
    """
    Save a DataFrame to CSV in a data directory.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (e.g., 'stories.csv')
        stage: Which data directory to save to ('raw', 'interim', 'processed')
        experiment: Experiment name. If None, uses active_experiment from config.
        **kwargs: Additional arguments passed to df.to_csv
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Saving CSV to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def load_parquet(filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Load a Parquet file from a data directory.
    
    Args:
        filename: Name of the file (e.g., 'embeddings.parquet')
        stage: Which data directory to load from
        experiment: Experiment name. If None, uses active_experiment from config.
        **kwargs: Additional arguments passed to pd.read_parquet
        
    Returns:
        DataFrame with the loaded data
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Loading Parquet from: {path}")
    return pd.read_parquet(path, **kwargs)


def save_parquet(df: pd.DataFrame, filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> None:
    """
    Save a DataFrame to Parquet in a data directory.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (e.g., 'embeddings.parquet')
        stage: Which data directory to save to
        experiment: Experiment name. If None, uses active_experiment from config.
        **kwargs: Additional arguments passed to df.to_parquet
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Saving Parquet to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, **kwargs)


def load_npy(filename: str, stage: str = "processed", experiment: Optional[str] = None) -> np.ndarray:
    """
    Load a NumPy array from a .npy file.
    
    Args:
        filename: Name of the file (e.g., 'embeddings.npy')
        stage: Which data directory to load from
        experiment: Experiment name. If None, uses active_experiment from config.
        
    Returns:
        NumPy array
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Loading .npy from: {path}")
    return np.load(path)


def save_npy(arr: np.ndarray, filename: str, stage: str = "processed", experiment: Optional[str] = None) -> None:
    """
    Save a NumPy array to a .npy file.
    
    Args:
        arr: NumPy array to save
        filename: Name of the file (e.g., 'embeddings.npy')
        stage: Which data directory to save to
        experiment: Experiment name. If None, uses active_experiment from config.
    """
    path = get_data_path(stage, experiment) / filename
    print(f"Saving .npy to: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def backfill_interaction_metadata(
    df: pd.DataFrame,
    *,
    simulated: bool = False,
    experiment: Optional[str] = None,
    metadata_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merge current interim interaction metadata into a processed dataframe.

    This keeps downstream metric scripts compatible with processed artifacts that were
    generated before new metadata columns were added to the cleaned interaction export.
    Existing non-null values in ``df`` are preserved; missing values are filled from the
    interim interaction CSV using conversation-aligned keys.
    """
    metadata_filename = metadata_filename or (
        "interaction_level_stories_filtered_simulated.csv"
        if simulated else
        "interaction_level_stories_filtered.csv"
    )

    metadata_df = load_csv(metadata_filename, stage="interim", experiment=experiment)
    key_candidates = ["conversation_id", "turn", "interaction_count"]
    key_columns = [col for col in key_candidates if col in df.columns and col in metadata_df.columns]

    if not key_columns:
        raise ValueError(
            "Could not identify interaction metadata join keys. "
            "Expected at least one of conversation_id/turn/interaction_count in both frames."
        )

    left = df.copy()
    right = metadata_df.copy()

    for numeric_key in ["turn", "interaction_count"]:
        if numeric_key in key_columns:
            left[numeric_key] = pd.to_numeric(left[numeric_key], errors="coerce")
            right[numeric_key] = pd.to_numeric(right[numeric_key], errors="coerce")

    metadata_columns = [col for col in right.columns if col not in key_columns]
    right = right[key_columns + metadata_columns].drop_duplicates(subset=key_columns, keep="last")

    merged = left.merge(right, on=key_columns, how="left", suffixes=("", "__meta"))

    for col in metadata_columns:
        meta_col = f"{col}__meta"
        if meta_col not in merged.columns:
            continue
        if col in merged.columns:
            prefer_current_metadata = col in {
                "analysis_turn",
                "complete_exchange",
                "starter",
                "starter_side",
                "starter_type",
                "author_1_type",
                "author_2_type",
            }
            if prefer_current_metadata:
                merged[col] = merged[meta_col].where(merged[meta_col].notna(), merged[col])
            else:
                merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
        else:
            merged[col] = merged[meta_col]
        merged.drop(columns=[meta_col], inplace=True)

    return merged
