#!/usr/bin/env python
"""
Story-level temporal cues and per-turn prompt echo.

Measurement only. The counts land in scripts/09_build_metric_tables.py;
all modelling happens in R.

Temporal cues support the retrospection driver: participants were asked to write
the future, and the question is which conditions in fact refer backward. Lexical
deictic cues come from de_temporal_cues.csv. Morphological tense comes from the
parse and is kept separate from them on purpose -- German preterite and perfect
are inflectional, and the two measures disagree in this corpus, so collapsing
them would hide that. Cue counts are deictic *reference*, not narrative tense,
and must be reported as such.

Prompt echo is computed per turn so its decay across the story can be reported.
It is also the control for scripts/10_condition_classifier.py: if the condition
is recoverable from a visitor's text only where that text reuses the prompt's
vocabulary, the manipulation check has not been established.

Usage:
    PYTHONPATH=src python scripts/07_temporal_cues.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.berlin_pov import (  # noqa: E402
    author_labelled_stream,
    build_frames,
    load_prompt_echo_terms,
    load_temporal_cues,
)
from nes.io import get_data_path  # noqa: E402

SPACY_MODEL = "de_dep_news_trf"


def _compile(terms) -> re.Pattern:
    """Word-boundary alternation over multi-word-capable terms, longest first."""
    ordered = sorted({str(t).strip().lower() for t in terms if str(t).strip()}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in ordered) + r")(?!\w)")


def temporal_counts(text: str, past: re.Pattern, future: re.Pattern) -> tuple[int, int]:
    low = text.lower()
    return len(past.findall(low)), len(future.findall(low))


def main() -> None:
    frames = build_frames(verbose=True)
    stream = author_labelled_stream(frames.turns)

    cues = load_temporal_cues()
    cue_set = cues[cues["in_cue_set"] == 1]
    past_re = _compile(cue_set.loc[cue_set["class"] == "past", "cue"])
    future_re = _compile(cue_set.loc[cue_set["class"] == "future", "cue"])
    loss_re = _compile(cues.loc[cues["class"] == "loss", "cue"])

    # ---- story level -------------------------------------------------
    stories = (
        stream.groupby(["conversation_id", "condition"], sort=False)["text"]
        .apply(lambda s: " ".join(s))
        .reset_index()
        .rename(columns={"text": "full_text"})
    )

    print(f"\nparsing {len(stories)} stories for morphological tense ...")
    nlp = spacy.load(SPACY_MODEL)
    tense_rows = []
    for doc in nlp.pipe(stories["full_text"].tolist(), batch_size=16):
        finite = [t for t in doc if t.pos_ in {"VERB", "AUX"} and t.morph.get("Tense")]
        tenses = [t.morph.get("Tense")[0] for t in finite]
        tense_rows.append(
            {
                "n_finite": len(finite),
                "n_past": sum(t == "Past" for t in tenses),
                "n_pres": sum(t == "Pres" for t in tenses),
            }
        )
    stories = pd.concat([stories, pd.DataFrame(tense_rows)], axis=1)

    counts = stories["full_text"].apply(lambda t: temporal_counts(t, past_re, future_re))
    stories["past_cues"] = [c[0] for c in counts]
    stories["future_cues"] = [c[1] for c in counts]
    stories["loss_cues"] = stories["full_text"].str.lower().apply(lambda t: len(loss_re.findall(t)))
    stories["n_words"] = stories["full_text"].str.split().apply(len)
    stories["past_tense_prop"] = stories["n_past"] / stories["n_finite"].replace(0, pd.NA)

    out_dir = get_data_path("processed", experiment="berlin")
    stories.drop(columns=["full_text"]).to_parquet(out_dir / "pov_temporal.parquet", index=False)

    # ---- author level ----------------------------------------------------
    # The same cue counts, split by who wrote the segment. A condition effect
    # concentrated in one author's turns is a different finding from one shared
    # across both, and only this split can tell them apart, so it is written
    # here rather than reconstructed downstream from the story totals.
    by_author = (
        stream.groupby(["conversation_id", "condition", "author"], sort=False)["text"]
        .apply(lambda s: " ".join(s))
        .reset_index()
        .rename(columns={"text": "author_text"})
    )
    a_counts = by_author["author_text"].apply(lambda t: temporal_counts(t, past_re, future_re))
    by_author["past_cues"] = [c[0] for c in a_counts]
    by_author["future_cues"] = [c[1] for c in a_counts]
    by_author["loss_cues"] = (
        by_author["author_text"].str.lower().apply(lambda t: len(loss_re.findall(t)))
    )
    by_author["n_words"] = by_author["author_text"].str.split().apply(len)
    by_author.drop(columns=["author_text"]).to_parquet(
        out_dir / "pov_temporal_by_author.parquet", index=False
    )

    print("\ncue rates per 1000 words, by condition:")
    agg = stories.groupby("condition").apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "past/1k": 1000 * g["past_cues"].sum() / g["n_words"].sum(),
                "future/1k": 1000 * g["future_cues"].sum() / g["n_words"].sum(),
                "past:future": g["past_cues"].sum() / max(g["future_cues"].sum(), 1),
                "past_tense_prop": g["past_tense_prop"].mean(),
                "loss/1k": 1000 * g["loss_cues"].sum() / g["n_words"].sum(),
            }
        ),
        include_groups=False,
    )
    print(agg.round(3).to_string())

    # ---- Prompt echo: turn level ----------------------------------------
    echo_terms = load_prompt_echo_terms()
    echo_res = {
        str(c): _compile(g["term"]) for c, g in echo_terms.groupby("condition")
    }

    seg = stream.copy()
    seg["n_words"] = seg["text"].str.split().apply(len)
    for cond, pat in echo_res.items():
        seg[f"echo_L{cond}"] = seg["text"].str.lower().apply(lambda t, p=pat: len(p.findall(t)))
    seg["echo_own"] = [
        r[f"echo_L{r['condition']}"] for _, r in seg.iterrows()
    ]
    seg.drop(columns=["text"]).to_parquet(out_dir / "pov_echo_turns.parquet", index=False)

    print("\nown-prompt echo per 1000 words, by condition x turn index:")
    pivot = seg.pivot_table(
        index="condition", columns="turn_index",
        values=["echo_own", "n_words"], aggfunc="sum",
    )
    rate = (1000 * pivot["echo_own"] / pivot["n_words"]).round(2)
    print(rate.to_string())

    print("\ncross-condition echo (rows=story condition, cols=prompt lexicon), per 1000 words:")
    cross = seg.groupby("condition").apply(
        lambda g: pd.Series(
            {f"L{c}": 1000 * g[f"echo_L{c}"].sum() / g["n_words"].sum() for c in echo_res}
        ),
        include_groups=False,
    )
    print(cross.round(2).to_string())
    print(f"\n-> {out_dir/'pov_temporal.parquet'}\n-> {out_dir/'pov_echo_turns.parquet'}")


if __name__ == "__main__":
    main()
