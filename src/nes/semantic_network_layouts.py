"""
Semantic network layout utilities.

This module computes pre-rendered 2D coordinates and adjacent-turn edges for
semantic-network visualizations. Scripts should call this module; analysis
notebooks should only read the output files and plot them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import backfill_interaction_metadata, get_project_root, load_parquet


CONDITION_LABELS = {
    "human-ai": "Human-AI",
    "human-human": "Human-Human",
    "ai-ai": "AI-AI",
}

DEFAULT_EXCLUDED_CONVERSATIONS = {
    "conv_ed575a06c11d42358e3eeb7826d2f959",
    "conv_a63a08273d0a4704a7638e4cd6850225",
}

PROVIDER_ERROR = "Network error while generating a response."


def parse_embedding(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        try:
            arr = np.fromstring(value.strip("[]"), sep=",")
        except Exception:
            return None
    else:
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return None
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def clean_conversation_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.json$", "", regex=True)


def load_condition_embeddings(
    condition: str,
    *,
    analysis_turn_min: int = 1,
    analysis_turn_max: int = 9,
    excluded_conversations: Iterable[str] = DEFAULT_EXCLUDED_CONVERSATIONS,
) -> pd.DataFrame:
    df = load_parquet(
        "story_embeddings_interaction_level.parquet",
        stage="processed",
        experiment=condition,
    )
    df = backfill_interaction_metadata(df, simulated=False, experiment=condition)
    df = df.copy()
    df["condition_id"] = condition
    df["condition"] = CONDITION_LABELS[condition]
    df["conversation_id"] = clean_conversation_id(df["conversation_id"])

    if "complete_exchange" in df.columns:
        df = df[df["complete_exchange"].fillna(True).astype(bool)]

    if "analysis_turn" in df.columns and df["analysis_turn"].notna().any():
        df["turn_index"] = pd.to_numeric(df["analysis_turn"], errors="coerce")
    elif "turn" in df.columns:
        df["turn_index"] = pd.to_numeric(df["turn"], errors="coerce")
    elif "interaction_count" in df.columns:
        df["turn_index"] = pd.to_numeric(df["interaction_count"], errors="coerce")
    else:
        raise ValueError(f"{condition}: no analysis_turn, turn, or interaction_count column found")

    df = df[df["turn_index"].between(analysis_turn_min, analysis_turn_max, inclusive="both")]
    df = df[~df["conversation_id"].isin(set(excluded_conversations))]

    text_cols = [col for col in ["author_1", "author_2"] if col in df.columns]
    for col in text_cols:
        df = df[df[col].astype(str) != PROVIDER_ERROR]

    return df


def balance_stories_by_condition(
    df: pd.DataFrame,
    *,
    stories_per_condition: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    story_counts = df.groupby("condition_id")["conversation_id"].nunique()
    if story_counts.empty:
        raise ValueError("Cannot balance stories: no stories found")

    target_n = stories_per_condition or int(story_counts.min())
    if target_n <= 0:
        raise ValueError("stories_per_condition must be positive")
    if target_n > int(story_counts.min()):
        raise ValueError(
            "stories_per_condition exceeds the available stories in at least one condition: "
            f"requested {target_n}, minimum available {int(story_counts.min())}"
        )

    rng = np.random.default_rng(seed)
    keep_ids: set[str] = set()

    for condition_id, condition_df in df.groupby("condition_id", sort=False):
        story_ids = np.array(sorted(condition_df["conversation_id"].unique()))
        sampled = rng.choice(story_ids, size=target_n, replace=False)
        keep_ids.update(sampled.tolist())

    return df[df["conversation_id"].isin(keep_ids)].copy(), target_n


def story_slot_order(condition_id: str, starter_side: str | None) -> tuple[str, str]:
    if condition_id == "human-ai":
        return "author_1", "author_2"
    if starter_side == "author_2":
        return "author_2", "author_1"
    return "author_1", "author_2"


def iter_turn_records(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict] = []
    embeddings: list[np.ndarray] = []
    turn_node_id = 0

    df = df.sort_values(["condition", "conversation_id", "turn_index"])

    for (condition_id, condition, conversation_id), grp in df.groupby(
        ["condition_id", "condition", "conversation_id"],
        sort=False,
    ):
        starter_values = grp.get("starter_side", pd.Series(dtype=object)).dropna().astype(str)
        starter_side = starter_values.iloc[0] if len(starter_values) else None
        first_slot, second_slot = story_slot_order(condition_id, starter_side)

        for exchange_index, (_, row) in enumerate(grp.iterrows(), start=1):
            for local_index, slot in enumerate((first_slot, second_slot)):
                emb = parse_embedding(row.get(f"{slot}_embedding"))
                if emb is None:
                    continue

                turn_node_id += 1
                embeddings.append(emb)
                records.append(
                    {
                        "turn_node_id": turn_node_id,
                        "condition": condition,
                        "condition_id": condition_id,
                        "conversation_id": conversation_id,
                        "sequence_index": 2 * exchange_index - 1 + local_index,
                        "turn_index": row["turn_index"],
                        "slot": slot,
                    }
                )

    if not records:
        raise ValueError("No valid turn embeddings found")

    mat = np.vstack(embeddings).astype(float)
    return pd.DataFrame.from_records(records), mat


def reduce_2d(
    mat: np.ndarray,
    *,
    method: str = "pca",
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> tuple[np.ndarray, str]:
    method = method.lower()

    if method in {"umap", "auto"}:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(n_neighbors, mat.shape[0] - 1),
                min_dist=min_dist,
                metric="cosine",
                random_state=seed,
            )
            return reducer.fit_transform(mat), "UMAP"
        except Exception as exc:
            if method == "umap":
                raise RuntimeError("UMAP requested but failed") from exc
            print(f"UMAP unavailable or failed ({exc}); falling back to PCA.")

    centered = mat - mat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, "PCA"


def cosine_distance_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    mat_norm = mat / norms
    sim = np.clip(mat_norm @ mat_norm.T, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.inf)
    return dist


def build_story_embeddings(turn_nodes: pd.DataFrame, turn_mat: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    story_records: list[dict] = []
    story_embeddings: list[np.ndarray] = []

    group_keys = ["condition", "condition_id", "conversation_id"]
    for story_node_id, ((condition, condition_id, conversation_id), idx) in enumerate(
        turn_nodes.groupby(group_keys, sort=False).indices.items(),
        start=1,
    ):
        story_records.append(
            {
                "story_node_id": story_node_id,
                "condition": condition,
                "condition_id": condition_id,
                "conversation_id": conversation_id,
                "n_turns": len(idx),
            }
        )
        story_embeddings.append(turn_mat[np.asarray(idx)].mean(axis=0))

    return pd.DataFrame.from_records(story_records), np.vstack(story_embeddings)


def build_knn_edges(nodes: pd.DataFrame, mat: np.ndarray, id_col: str, k: int) -> pd.DataFrame:
    if len(nodes) < 2:
        return pd.DataFrame()

    k_eff = min(k, len(nodes) - 1)
    dist = cosine_distance_matrix(mat)
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()

    ids = nodes[id_col].to_numpy()
    for i in range(len(nodes)):
        neighbors = np.argsort(dist[i])[:k_eff]
        for j in neighbors:
            a, b = sorted((int(ids[i]), int(ids[j])))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rows.append(
                {
                    "from": int(ids[i]),
                    "to": int(ids[j]),
                    "distance": float(dist[i, j]),
                    "x": float(nodes.iloc[i]["x"]),
                    "y": float(nodes.iloc[i]["y"]),
                    "xend": float(nodes.iloc[j]["x"]),
                    "yend": float(nodes.iloc[j]["y"]),
                }
            )

    return pd.DataFrame.from_records(rows)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(denom) or denom == 0:
        return np.nan
    sim = np.dot(a, b) / denom
    return float(1.0 - np.clip(sim, -1.0, 1.0))


def build_turn_edges(turn_nodes: pd.DataFrame, turn_mat: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    node_to_index = {
        int(node_id): index
        for index, node_id in enumerate(turn_nodes["turn_node_id"].to_numpy())
    }

    for (_, conversation_id), grp in turn_nodes.groupby(["condition", "conversation_id"], sort=False):
        grp = grp.sort_values("sequence_index")
        current = grp.iloc[:-1]
        nxt = grp.iloc[1:]
        for (_, row), (_, next_row) in zip(current.iterrows(), nxt.iterrows()):
            from_id = int(row["turn_node_id"])
            to_id = int(next_row["turn_node_id"])
            from_turn = float(row["turn_index"])
            to_turn = float(next_row["turn_index"])

            rows.append(
                {
                    "condition": row["condition"],
                    "condition_id": row["condition_id"],
                    "conversation_id": conversation_id,
                    "from": from_id,
                    "to": to_id,
                    "from_turn": from_turn,
                    "to_turn": to_turn,
                    "from_slot": row["slot"],
                    "to_slot": next_row["slot"],
                    "pair_type": "within_exchange" if from_turn == to_turn else "between_exchange",
                    "adjacent_distance": cosine_distance(
                        turn_mat[node_to_index[from_id]],
                        turn_mat[node_to_index[to_id]],
                    ),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "xend": float(next_row["x"]),
                    "yend": float(next_row["y"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def balance_edges_by_condition(
    turn_nodes: pd.DataFrame,
    turn_edges: pd.DataFrame,
    *,
    edges_per_condition: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    edge_counts = turn_edges.groupby("condition")["from"].size()
    if edge_counts.empty:
        raise ValueError("Cannot balance edges: no edges found")

    target_n = edges_per_condition or int(edge_counts.min())
    if target_n <= 0:
        raise ValueError("edges_per_condition must be positive")
    if target_n > int(edge_counts.min()):
        raise ValueError(
            "edges_per_condition exceeds the available adjacent edges in at least one condition: "
            f"requested {target_n}, minimum available {int(edge_counts.min())}"
        )

    sampled_edges = (
        turn_edges
        .groupby("condition", group_keys=False, sort=False)
        .sample(n=target_n, random_state=seed)
        .reset_index(drop=True)
    )
    sampled_node_ids = set(sampled_edges["from"]).union(set(sampled_edges["to"]))
    sampled_nodes = turn_nodes[turn_nodes["turn_node_id"].isin(sampled_node_ids)].copy()

    return sampled_nodes, sampled_edges, target_n


def compute_semantic_network_layouts(
    *,
    conditions: Iterable[str] = ("human-ai", "human-human", "ai-ai"),
    method: str = "pca",
    turn_neighbors: int = 30,
    min_dist: float = 0.1,
    seed: int = 42,
    analysis_turn_min: int = 1,
    analysis_turn_max: int = 9,
    output_dir: str | Path = "analysis/comparison/semantic_network_layouts",
    excluded_conversations: Iterable[str] = DEFAULT_EXCLUDED_CONVERSATIONS,
    balance_stories: bool = True,
    stories_per_condition: int | None = None,
    balance_edges: bool = True,
    edges_per_condition: int | None = None,
) -> dict:
    conditions = list(conditions)
    unknown = set(conditions) - set(CONDITION_LABELS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    frames = [
        load_condition_embeddings(
            cond,
            analysis_turn_min=analysis_turn_min,
            analysis_turn_max=analysis_turn_max,
            excluded_conversations=excluded_conversations,
        )
        for cond in conditions
    ]
    df = pd.concat(frames, ignore_index=True)

    balanced_story_n = None
    if balance_stories:
        df, balanced_story_n = balance_stories_by_condition(
            df,
            stories_per_condition=stories_per_condition,
            seed=seed,
        )

    turn_nodes, turn_mat = iter_turn_records(df)
    turn_coords, turn_method = reduce_2d(
        turn_mat,
        method=method,
        seed=seed,
        n_neighbors=turn_neighbors,
        min_dist=min_dist,
    )
    turn_nodes[["x", "y"]] = turn_coords
    turn_edges = build_turn_edges(turn_nodes, turn_mat)

    balanced_edge_n = None
    if balance_edges:
        turn_nodes, turn_edges, balanced_edge_n = balance_edges_by_condition(
            turn_nodes,
            turn_edges,
            edges_per_condition=edges_per_condition,
            seed=seed,
        )

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = get_project_root() / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    turn_nodes.to_csv(output_path / "turn_nodes.csv", index=False)
    turn_edges.to_csv(output_path / "turn_edges.csv", index=False)

    edge_means = (
        turn_edges.groupby("condition", sort=False)["adjacent_distance"]
        .mean()
        .round(6)
        .to_dict()
    )

    metadata = {
        "conditions": conditions,
        "method_requested": method,
        "turn_layout_method": turn_method,
        "turn_neighbors": turn_neighbors,
        "min_dist": min_dist,
        "seed": seed,
        "analysis_turn_min": analysis_turn_min,
        "analysis_turn_max": analysis_turn_max,
        "balance_stories": balance_stories,
        "stories_per_condition": balanced_story_n,
        "balance_edges": balance_edges,
        "edges_per_condition": balanced_edge_n,
        "n_turn_nodes": int(len(turn_nodes)),
        "n_turn_edges": int(len(turn_edges)),
        "mean_adjacent_distance_by_condition": edge_means,
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return metadata
