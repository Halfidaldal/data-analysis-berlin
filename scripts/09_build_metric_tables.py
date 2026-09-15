#!/usr/bin/env python
"""
Assemble every metric into tidy tables for the R analysis. No models are fitted here.

This is the hand-off point. Everything upstream computes one kind of number;
this script joins them onto a common key and writes five tables, each at the
grain the corresponding model needs.

  metrics_alignment_pairs   one directed (predictor, response) turn pair -- valence
  metrics_semantic_pairs    one directed adjacency -- semantic cosine
  metrics_exchange          one turn-pair
  metrics_story             one story
  metrics_segment           one author-turn
  metrics_clause            one first-person-singular clause

`metrics_alignment_pairs` is already in the long directed form. Model the
within-story standardised columns, not the raw ones -- this matches the NLP4DH
specification:

    lmer(resp_z ~ pred_z * direction + (1 + pred_z | story_id), ...)
    lmer(resp_z ~ pred_z * direction * condition + (1 + pred_z | story_id), ...)

`direction` is a factor with `model_to_human` first, so
`pred_z:direction[human_to_model]` estimates how much more the model tracks the
visitor than the visitor tracks the model. See build_alignment_pairs for why raw
`v_predictor` must not be used, and why the standardisation has to be grouped by
story AND direction rather than story alone.

VALENCE IS CONCEPT-VECTOR PROJECTION THROUGHOUT
-----------------------------------------------
Turn valence and segment valence are the same measure, not two: a segment is one
side of one turn-pair, so `metrics_segment` reshapes the same projection scores
that `metrics_exchange` carries side by side. Nothing here rescores text with a
sentiment classifier -- mixing a classifier score into passage-level sentiment
while turn-level sentiment is a projection would put two different quantities
under one name.

WHAT IS DELIBERATELY NOT COMPUTED
---------------------------------
Surprisal-based novelty: model turns are hard-capped at ~19 words and 78%
terminate mid-word, so a per-turn surprisal score would largely index the
truncation.

Two-stage alignment asymmetry (a within-story correlation, then a difference of
Fisher z's): at Berlin's median of 4 complete exchanges that estimator returns
~1.0 under a true null of zero asymmetry, against 0.40 at the EMNLP corpus
length. `metrics_alignment_pairs` estimates no per-story correlation and is the
measure to model instead. `n_complete_exchanges` is on `metrics_story` so the
R side can check that nothing is carried by short stories.

Usage:
    PYTHONPATH=src python scripts/09_build_metric_tables.py [--language de]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.berlin_pov import (  # noqa: E402
    author_labelled_stream,
    build_frames,
    load_temporal_cues,
)
from nes.io import get_data_path  # noqa: E402

HUMAN_COL, MODEL_COL = "user", "ai"


def proc(language: str) -> Path:
    return get_data_path("processed", experiment="berlin")


def sfx(language: str) -> str:
    return "" if language == "de" else f"_{language}"


# ---------------------------------------------------------------------------
# Exchange frame
# ---------------------------------------------------------------------------


def build_exchanges(language: str) -> pd.DataFrame:
    """
    One row per turn-pair, with both sides' text and whether the pair is complete.

    The final model turn is systematically empty (every story ends on a visitor
    turn), so `complete` is False for the last row of most stories. Anything
    needing both sides must filter on it rather than assuming a pair exists.
    """
    frames = build_frames(language=language)
    t = frames.turns.copy()
    t["human_text"] = t[HUMAN_COL].fillna("").astype(str).str.strip()
    t["model_text"] = t[MODEL_COL].fillna("").astype(str).str.strip()
    t["complete"] = (t["human_text"].str.len() > 0) & (t["model_text"].str.len() > 0)
    t = t.rename(columns={"conversation_id": "story_id"})
    t["condition"] = t["condition"].astype(str)
    return t[
        ["story_id", "condition", "turn_index", "human_text", "model_text", "complete"]
    ].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Valence (concept-vector projection, from script 04)
# ---------------------------------------------------------------------------


def _align_in_order(sent_keys: list[str], frame_keys: list[str]) -> list[int | None]:
    """
    Map each row of the sentiment table onto a turn_index of the analysis frame.

    Composition order is preserved on both sides, but the sentiment table can
    have rows missing, so this walks the sequences together and matches greedily,
    never going backwards. Returns None where a row cannot be placed.
    """
    out: list[int | None] = []
    j = 0
    for key in sent_keys:
        hit = None
        for jj in range(j, len(frame_keys)):
            if frame_keys[jj] == key:
                hit = jj
                break
        out.append(hit)
        if hit is not None:
            j = hit + 1
    return out


def attach_valence(ex: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Merge per-turn concept-vector-projection valence onto the exchange frame.

    THE JOIN IS NOT POSITIONAL, AND MUST NOT BE.
    The sentiment table is not guaranteed row-aligned to the interim data: the
    version shipped with this repo covered 205 of 316 conversations, carried
    spelling-corrected human text, and for 12 of the 173 analysis-set stories was
    missing *interior* turns rather than merely trailing ones. A `cumcount()`
    join shifts valence by one turn in those stories, which silently corrupts
    every lagged pair in `metrics_alignment_pairs`.

    The model side is not spelling-corrected, so it keys the two tables together
    exactly. Anything that cannot be placed is left NaN rather than guessed.
    """
    path = proc(language) / "dyadic_sentiment_scores.parquet"
    if not path.exists():
        print(f"  ! {path.name} not found -- valence will be NaN")
        ex["v_human"] = np.nan
        ex["v_model"] = np.nan
        return ex

    s = pd.read_parquet(path)
    hcol, mcol = "user_sentiment_projection", "ai_sentiment_projection"
    missing = [c for c in (hcol, mcol) if c not in s.columns]
    if missing:
        raise KeyError(
            f"{path.name} lacks {missing}. Re-run scripts/04_compute_sentiment.py. "
            "Do not substitute a classifier score: it is a different measure."
        )

    def norm(x) -> str:
        return str(x).strip() if pd.notna(x) else ""

    frame_keys = {
        cid: [norm(t) for t in g.sort_values("turn_index")["model_text"]]
        for cid, g in ex.groupby("story_id", sort=False)
    }

    rows, unplaced = [], 0
    for cid, g in s.groupby("conversation_id", sort=False):
        if cid not in frame_keys:
            continue
        g = g.sort_values("turn") if "turn" in g.columns else g
        idx = _align_in_order([norm(t) for t in g[MODEL_COL]], frame_keys[cid])
        for turn_index, (_, r) in zip(idx, g.iterrows()):
            if turn_index is None:
                unplaced += 1
                continue
            rows.append(
                {
                    "story_id": cid,
                    "turn_index": turn_index,
                    "v_human": r[hcol],
                    "v_model": r[mcol],
                }
            )
    if unplaced:
        print(f"  ! {unplaced} sentiment rows unplaced, left NaN")
    return ex.merge(pd.DataFrame(rows), on=["story_id", "turn_index"], how="left")


# ---------------------------------------------------------------------------
# Exploration: cosine distance between the two sides of an exchange
# ---------------------------------------------------------------------------


def attach_semantic_distance(ex: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    D_t = 1 - cos(e_human_t, e_model_t), the EMNLP semantic-distance measure.

    Each exchange yields one number and nothing is estimated within a story, so
    the short story length costs this measure nothing. Truncated model fragments
    do inflate distance in absolute terms, but that inflation is constant across
    conditions and is absorbed by the intercept of any condition model.
    """
    path = proc(language) / "story_embeddings_interaction_level.parquet"
    if not path.exists():
        print(f"  ! {path.name} not found -- semantic_distance will be NaN")
        ex["semantic_distance"] = np.nan
        return ex

    e = pd.read_parquet(path).copy()
    hcol = "author_1_embedding" if "author_1_embedding" in e.columns else "user_embedding"
    mcol = "author_2_embedding" if "author_2_embedding" in e.columns else "ai_embedding"
    if hcol not in e.columns or mcol not in e.columns:
        raise KeyError(
            f"{path.name} has no usable embedding columns. Re-run scripts/03_compute_embeddings.py."
        )
    e["turn_index"] = e.groupby("conversation_id").cumcount()

    def cos_dist(row) -> float:
        a, b = row[hcol], row[mcol]
        if a is None or b is None:
            return np.nan
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return np.nan
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float(1.0 - (a @ b) / (na * nb))

    e["semantic_distance"] = e.apply(cos_dist, axis=1)
    return ex.merge(
        e[["conversation_id", "turn_index", "semantic_distance"]].rename(
            columns={"conversation_id": "story_id"}
        ),
        on=["story_id", "turn_index"],
        how="left",
    )


# ---------------------------------------------------------------------------
# Alignment pairs
# ---------------------------------------------------------------------------


def build_alignment_pairs(ex: pd.DataFrame) -> pd.DataFrame:
    """
    One row per directed (predictor, response) pair.

    human_to_model : the model responds within the same exchange, so the pair is
                     (v_human_t, v_model_t).
    model_to_human : the visitor responds in the next exchange, so the pair is
                     (v_model_t, v_human_t+1).

    Turns strictly alternate, which is what makes these the two adjacencies.
    Rows where either side is missing are dropped rather than imputed.
    """
    rows = []
    for cid, g in ex.groupby("story_id", sort=False):
        g = g.sort_values("turn_index")
        cond = g["condition"].iloc[0]
        vh = g["v_human"].to_numpy(dtype=float)
        vm = g["v_model"].to_numpy(dtype=float)
        ti = g["turn_index"].to_numpy()

        for k in range(len(g)):
            if np.isfinite(vh[k]) and np.isfinite(vm[k]):
                rows.append((cid, cond, "human_to_model", ti[k], vh[k], vm[k]))
        for k in range(len(g) - 1):
            if np.isfinite(vm[k]) and np.isfinite(vh[k + 1]):
                rows.append((cid, cond, "model_to_human", ti[k], vm[k], vh[k + 1]))

    d = pd.DataFrame(
        rows,
        columns=["story_id", "condition", "direction", "turn_index", "v_predictor", "v_response"],
    )
    # model_to_human first so the direction coefficient reads as "extra tracking
    # by the model", which is the quantity of interest.
    d["direction"] = pd.Categorical(
        d["direction"], categories=["model_to_human", "human_to_model"]
    )

    # Standardise predictor and response WITHIN story x direction. This is the
    # specification used in the NLP4DH analysis, and it is the one to model.
    #
    # Why not raw `v_predictor`: a single raw predictor forces two unrelated
    # associations into one coefficient. On this corpus they have OPPOSITE SIGNS
    # -- the between-story slope is +1.14 (stories positive overall are positive
    # on both sides, an ecological association rather than turn-by-turn
    # tracking) against a within-story slope of -0.12. With 23% of predictor
    # variance sitting between stories, a raw-predictor model returns about
    # +0.15 and reports the ecological association under the name "alignment".
    #
    # Why the grouping must include `direction` and not just `story_id`:
    # a turn appears as the response in one direction and the predictor in the
    # other, so centring on a mean pooled across directions mixes the two
    # variables' scales and induces a strong artefactual negative association.
    # Measured here: pooled centring gives a permutation null of -0.18 with a
    # -0.14 observed slope (i.e. almost entirely artefact), while centring
    # within story x direction gives a null of -0.002, essentially zero, even at
    # this series length. Group by both.
    #
    # With this standardisation the slope is a within-story correlation, so
    # testing against zero is approximately correct. Permuting `v_response`
    # within story x direction remains cheap insurance and is what the reported
    # nulls above come from.
    g = d.groupby(["story_id", "direction"], observed=True)
    for src, dst in (("v_predictor", "pred_z"), ("v_response", "resp_z")):
        mu = g[src].transform("mean")
        sd = g[src].transform("std")
        d[dst] = (d[src] - mu) / sd.replace(0, np.nan)
    d[["pred_z", "resp_z"]] = d[["pred_z", "resp_z"]].replace([np.inf, -np.inf], np.nan)

    # Mundlak terms, kept so the ecological component can be inspected rather
    # than silently absorbed. Report `pred_between` separately; it is not
    # alignment.
    d["pred_between"] = d.groupby("story_id")["v_predictor"].transform("mean")
    d["pred_within"] = d["v_predictor"] - d["pred_between"]
    return d


# ---------------------------------------------------------------------------
# Segment table
# ---------------------------------------------------------------------------


def _compile(terms) -> re.Pattern:
    ordered = sorted(
        {str(t).strip().lower() for t in terms if str(t).strip()}, key=len, reverse=True
    )
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in ordered) + r")(?!\w)")


def temporal_class(p: int, f: int) -> str:
    if p and not f:
        return "elegiac"
    if f and not p:
        return "anticipatory"
    if p and f:
        return "mixed"
    return "neutral"


# ---------------------------------------------------------------------------
# Semantic alignment pairs
# ---------------------------------------------------------------------------


def build_semantic_pairs(language: str) -> pd.DataFrame | None:
    """
    Directional semantic alignment: the cosine analogue of the valence pairing rules.

    One row per directed adjacency, with two similarities:

      cos_partner  response vs the partner turn it responded to
      cos_self     response vs the responder's OWN previous turn

    `cos_partner` alone is the canonical measure (it is what the EMNLP and NLP4DH
    analyses call semantic alignment), but it cannot separate "took up what my
    partner said" from "we are both still on the same subject" -- adjacent turns
    in one story sit around 0.52 regardless. `cos_self` is the control: subject
    matter is common to both columns, so `cos_partner - cos_self` cancels it and
    leaves only who the responder is following.

    Note the two directions are not symmetric as an instrument: the model always
    responds to a complete visitor turn, while the visitor responds to a model
    turn that is truncated mid-word ~94% of the time. That depresses the
    visitor's apparent uptake. It is constant across conditions (truncation rate
    p = 0.70, model turn length p = 0.75), so it cancels in a condition
    interaction but sits directly in the pooled direction effect. Model the
    interaction; do not report the pooled gap.
    """
    path = proc(language) / "story_embeddings_interaction_level.parquet"
    if not path.exists():
        print(f"  ! {path.name} not found -- semantic pairs skipped")
        return None

    e = pd.read_parquet(path).copy()
    hcol = "author_1_embedding" if "author_1_embedding" in e.columns else "user_embedding"
    mcol = "author_2_embedding" if "author_2_embedding" in e.columns else "ai_embedding"
    e["turn_index"] = e.groupby("conversation_id").cumcount()

    frames = build_frames(language=language)
    keep = frames.text[["conversation_id", "workshop_id"]].rename(
        columns={"workshop_id": "condition"}
    )
    e = e.merge(keep, on="conversation_id", how="inner")
    e["condition"] = e["condition"].astype(str)

    def vec(x):
        return None if x is None else np.asarray(x, dtype=float).ravel()

    def cos(a, b):
        if a is None or b is None or a.size != b.size or a.size == 0:
            return np.nan
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return np.nan if na == 0 or nb == 0 else float(a @ b / (na * nb))

    rows = []
    for cid, g in e.groupby("conversation_id", sort=False):
        g = g.sort_values("turn_index")
        h = [vec(x) for x in g[hcol]]
        m = [vec(x) for x in g[mcol]]
        cond = g["condition"].iloc[0]
        for t in range(len(g)):
            # the model responds to the visitor, within the same exchange
            rows.append(
                {
                    "story_id": cid, "condition": cond, "turn_index": t,
                    "direction": "model_uptake",
                    "cos_partner": cos(h[t], m[t]),
                    "cos_self": cos(m[t - 1], m[t]) if t > 0 else np.nan,
                }
            )
            # the visitor responds to the model, in the next exchange
            if t + 1 < len(g):
                rows.append(
                    {
                        "story_id": cid, "condition": cond, "turn_index": t,
                        "direction": "human_uptake",
                        "cos_partner": cos(m[t], h[t + 1]),
                        "cos_self": cos(h[t], h[t + 1]),
                    }
                )
    d = pd.DataFrame(rows).dropna(subset=["cos_partner"])
    d["direction"] = pd.Categorical(
        d["direction"], categories=["human_uptake", "model_uptake"]
    )
    d["rubber_band"] = d["cos_partner"] - d["cos_self"]
    return d


def build_segments(ex: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    One row per author-turn: who wrote it, its valence, and its temporal cues.

    Valence is taken from the exchange table rather than rescored, so segment and
    turn sentiment are literally the same numbers and cannot drift apart.
    """
    long = []
    for author, tcol, vcol in (
        ("human", "human_text", "v_human"),
        ("model", "model_text", "v_model"),
    ):
        part = ex[["story_id", "condition", "turn_index", tcol, vcol]].copy()
        part.columns = ["story_id", "condition", "turn_index", "text", "valence"]
        part["author"] = author
        long.append(part)
    seg = pd.concat(long, ignore_index=True)
    seg = seg[seg["text"].str.len() > 0].copy()
    seg["n_words"] = seg["text"].str.split().apply(len)

    if language == "de":
        cues = load_temporal_cues()
        cue_set = cues[cues["in_cue_set"] == 1]
        past_re = _compile(cue_set.loc[cue_set["class"] == "past", "cue"])
        future_re = _compile(cue_set.loc[cue_set["class"] == "future", "cue"])
        loss_re = _compile(cues.loc[cues["class"] == "loss", "cue"])
        low = seg["text"].str.lower()
        seg["past_cues"] = low.apply(lambda t: len(past_re.findall(t)))
        seg["future_cues"] = low.apply(lambda t: len(future_re.findall(t)))
        seg["loss_cues"] = low.apply(lambda t: len(loss_re.findall(t)))
    else:
        seg["past_cues"] = np.nan
        seg["future_cues"] = np.nan
        seg["loss_cues"] = np.nan

    seg["temporal"] = [
        temporal_class(p, f) if np.isfinite(p) and np.isfinite(f) else None
        for p, f in zip(seg["past_cues"], seg["future_cues"])
    ]

    # Prompt echo, if script 07 has run: the control for the manipulation check.
    echo_path = proc(language) / f"pov_echo_turns{sfx(language)}.parquet"
    if echo_path.exists():
        e = pd.read_parquet(echo_path).rename(columns={"conversation_id": "story_id"})
        cols = [c for c in e.columns if c.startswith("echo_")]
        seg = seg.merge(
            e[["story_id", "turn_index", "author", *cols]],
            on=["story_id", "turn_index", "author"],
            how="left",
        )
    return seg.sort_values(["story_id", "turn_index", "author"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Story table
# ---------------------------------------------------------------------------


def build_stories(ex: pd.DataFrame, seg: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    One row per story: per-side aggregates, and the signed model-minus-human gaps.

    Gaps are signed throughout. Unlike the EMNLP HH and AA conditions, Berlin has
    a privileged agent in each slot -- slot 1 is always the visitor -- so the
    direction of a gap is meaningful and folding it into a magnitude would throw
    that away. Take `abs()` in R if an unsigned magnitude is wanted.
    """
    complete = ex[ex["complete"]]
    base = (
        complete.groupby(["story_id", "condition"], sort=False)
        .agg(
            v_human=("v_human", "mean"),
            v_model=("v_model", "mean"),
            semantic_distance=("semantic_distance", "mean"),
            n_complete_exchanges=("complete", "sum"),
        )
        .reset_index()
    )
    base = base.merge(
        ex.groupby("story_id").size().rename("n_turn_pairs").reset_index(),
        on="story_id",
        how="left",
    )

    # Per-side surface and cue totals, from the segment table.
    wide = seg.pivot_table(
        index="story_id",
        columns="author",
        values=["n_words", "past_cues", "future_cues", "loss_cues"],
        aggfunc="sum",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    base = base.merge(wide.reset_index(), on="story_id", how="left")

    ttr = (
        seg.groupby(["story_id", "author"])["text"]
        .apply(lambda s: (lambda w: len({x.lower() for x in w}) / len(w) if w else np.nan)(
            " ".join(s).split()
        ))
        .unstack("author")
    )
    ttr.columns = [f"ttr_{c}" for c in ttr.columns]
    base = base.merge(ttr.reset_index(), on="story_id", how="left")

    for name, h, m in (
        ("sentiment", "v_human", "v_model"),
        ("words", "n_words_human", "n_words_model"),
        ("ttr", "ttr_human", "ttr_model"),
    ):
        if h in base.columns and m in base.columns:
            base[f"gap_{name}"] = base[m] - base[h]

    # Story-level cue totals and the retrospection index.
    tot = seg.groupby("story_id")[["past_cues", "future_cues", "loss_cues", "n_words"]].sum()
    tot.columns = ["past_cues", "future_cues", "loss_cues", "n_words"]
    base = base.merge(tot.reset_index(), on="story_id", how="left")
    base["past_per_1k"] = 1000 * base["past_cues"] / base["n_words"]
    base["future_per_1k"] = 1000 * base["future_cues"] / base["n_words"]
    base["retrospection_index"] = np.log(
        (base["past_cues"] + 0.5) / (base["future_cues"] + 0.5)
    )

    # Morphological tense and clause aggregates, if script 06 has run. Kept
    # separate from the cue counts on purpose: they order differently in this
    # corpus, so collapsing them would hide that the cue measure is deictic
    # reference rather than narrative tense.
    cl_path = proc(language) / f"pov_clauses{sfx(language)}.parquet"
    if cl_path.exists():
        cl = pd.read_parquet(cl_path)
        cl = cl[~cl["truncation_artefact"]]
        agg = cl.groupby("conversation_id").agg(
            n_clauses=("clause_id", "count"),
            transitive_rate=("has_direct_object", "mean"),
            past_tense_rate=("tense", lambda s: (s == "Past").mean()),
            split_predicate_rate=("split_predicate", "mean"),
        )
        base = base.merge(
            agg.reset_index().rename(columns={"conversation_id": "story_id"}),
            on="story_id",
            how="left",
        )

    # Surface metrics from script 05, if it has run.
    sm_path = proc(language) / "full_story_surface_metrics.parquet"
    if sm_path.exists():
        sm = pd.read_parquet(sm_path)
        if "conversation_id" in sm.columns:
            keep = [c for c in sm.columns if c != "conversation_id"]
            base = base.merge(
                sm.rename(columns={"conversation_id": "story_id"})[["story_id", *keep]],
                on="story_id",
                how="left",
                suffixes=("", "_td"),
            )
    return base


# ---------------------------------------------------------------------------


def build_clauses(language: str) -> pd.DataFrame | None:
    path = proc(language) / f"pov_clauses{sfx(language)}.parquet"
    if not path.exists():
        print(f"  ! {path.name} not found -- run scripts/06_parse_clauses.py")
        return None
    cl = pd.read_parquet(path).rename(
        columns={"conversation_id": "story_id", "verb_author": "author"}
    )
    cl["condition"] = cl["condition"].astype(str)
    return cl


def main(language: str = "de") -> None:
    print(f"building metric tables, language={language}")
    ex = build_exchanges(language)
    ex = attach_valence(ex, language)
    ex = attach_semantic_distance(ex, language)

    pairs = build_alignment_pairs(ex)
    seg = build_segments(ex, language)
    stories = build_stories(ex, seg, language)
    clauses = build_clauses(language)
    sempairs = build_semantic_pairs(language)

    out = proc(language)
    s = sfx(language)
    written = []
    for name, df in (
        ("metrics_alignment_pairs", pairs),
        ("metrics_exchange", ex.drop(columns=["human_text", "model_text"])),
        ("metrics_story", stories),
        ("metrics_segment", seg.drop(columns=["text"])),
        ("metrics_clause", clauses),
        ("metrics_semantic_pairs", sempairs),
    ):
        if df is None:
            continue
        path = out / f"{name}{s}.parquet"
        df.to_parquet(path, index=False)
        written.append((name, len(df), df.shape[1]))

    print("\n" + "=" * 66)
    print(f"{'table':<28}{'rows':>8}{'cols':>7}")
    print("=" * 66)
    for name, n, k in written:
        print(f"{name:<28}{n:>8}{k:>7}")

    print(f"\nstories: {stories['story_id'].nunique()}  "
          f"({stories.groupby('condition').size().to_dict()})")
    print(
        "\nalignment pairs by direction x condition:\n"
        + pairs.groupby(["direction", "condition"], observed=True)
        .size()
        .rename("n")
        .to_string()
    )
    print("\nR, once loaded (within-story standardised, NOT raw v_predictor):")
    print("  lmer(resp_z ~ pred_z * direction + (1 + pred_z | story_id), d)")
    print("  lmer(resp_z ~ pred_z * direction * condition + (1 + pred_z | story_id), d)")
    print("  # pred_z/resp_z are z-scored within story x direction (NLP4DH spec).")
    print("  # pred_between is the ecological term; report it, don't call it alignment.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="de", choices=["de", "en"])
    main(**vars(ap.parse_args()))
