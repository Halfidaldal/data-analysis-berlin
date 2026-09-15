#!/usr/bin/env python
"""
LLM annotation of first-person predicates: the experience half of mind perception.

Transitivity (scripts/06_parse_clauses.py) is a clean grammatical read on
narrative AGENCY, but it says nothing about EXPERIENCE, and the two are the two
factors of mind perception (Gray, Gray & Wegner), not one construct. This script
supplies the missing half.

It is deliberately not lexicon-based. A hand-built German verb list cannot be
neutral across these three conditions: any inventory of stative predicates is
already an inventory of what an obsolete object does, so a lexical measure of
inner life would carry condition-specific content into its own outcome. An LLM
labelling each clause in context has no such inventory to inherit.

EXPERIENCE (feeling, sensation, emotion, suffering) and AGENCY (planning,
deciding, intending) are labelled separately rather than merged, because the two
authors may well discriminate on different axes and collapsing them would hide it.

This is annotation, not ground truth. Validate a hand-read sample against it
before any claim rests on the labels alone.

Results are cached per clause so re-runs cost nothing.

Usage:
    PYTHONPATH=src python scripts/08_classify_predicates.py [--language de] [--limit N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.io import get_data_path, get_project_root  # noqa: E402

MODEL = "gpt-4.1-mini"
BATCH = 20

SYSTEM = """You label predicates from co-written first-person fiction. Each item is one
clause whose subject is the narrator ("I"), shown with the sentence it came from.

Assign exactly one label to what the narrator's verb expresses:

- experience : feeling, emotion, bodily sensation, suffering, pleasure, longing,
               missing something. The narrator undergoes something inwardly.
- cognition  : thinking, knowing, remembering, wondering, understanding, imagining.
- volition   : wanting, wishing, hoping, deciding, intending, planning, choosing.
- action     : doing something to or in the world - moving, making, taking, speaking,
               affecting an object or another person.
- state      : being, existing, being located or positioned, having a property.
               Also use this for posture (lying, standing) with no goal.
- unclear    : the text is truncated, garbled, or too fragmentary to judge.

Judge only the narrator's own predicate, not the wider sentence. The text may be
German or English, and may break off mid-word; that is expected.

Return a JSON object {"labels": [{"id": <int>, "label": "<one of the above>"}, ...]}
with exactly one entry per item, in the order given."""


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def label_batch(client, items: list[dict]) -> dict[str, str]:
    lines = []
    for i, it in enumerate(items):
        subj = it["subject_text"] or "(dropped)"
        lines.append(
            f'{i}. verb: "{it["lexical_verb"]}" | clause subject: "{subj}" '
            f'| sentence: {it["sentence"][:220]}'
        )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content)
    out = {}
    for entry in parsed.get("labels", []):
        idx = int(entry["id"])
        if 0 <= idx < len(items):
            out[items[idx]["clause_id"]] = str(entry["label"]).strip().lower()
    return out


def main(language: str = "de", limit: int | None = None) -> None:
    import dotenv
    import openai

    dotenv.load_dotenv(get_project_root() / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found (expected in .env at the repo root)")

    proc = get_data_path("processed", experiment="berlin")
    suffix = "" if language == "de" else f"_{language}"
    clauses = pd.read_parquet(proc / f"pov_clauses{suffix}.parquet")
    clauses = clauses[~clauses["truncation_artefact"]].copy()
    if limit:
        clauses = clauses.head(limit)

    cache_path = proc / f"pov_llm_labels{suffix}.json"
    cache = load_cache(cache_path)
    todo = clauses[~clauses["clause_id"].isin(cache)]
    print(f"{len(clauses)} clauses, {len(cache)} cached, {len(todo)} to label with {MODEL}")

    client = openai.OpenAI()
    records = todo.to_dict("records")
    for start in range(0, len(records), BATCH):
        batch = records[start : start + BATCH]
        try:
            cache.update(label_batch(client, batch))
        except Exception as exc:  # noqa: BLE001
            print(f"  batch at {start} failed: {exc}")
            continue
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
        print(f"  {min(start+BATCH, len(records))}/{len(records)}", end="\r")

    clauses["llm_label"] = clauses["clause_id"].map(cache)
    print(f"\n\nlabel distribution ({clauses['llm_label'].notna().sum()} labelled):")
    print(clauses["llm_label"].value_counts(dropna=False).to_string())

    clauses["llm_interiority"] = clauses["llm_label"].isin(["experience", "cognition", "volition"])
    clauses["llm_experience"] = clauses["llm_label"].eq("experience")
    clauses["llm_agency_mind"] = clauses["llm_label"].isin(["volition", "cognition"])

    out = proc / f"pov_clauses_llm{suffix}.parquet"
    clauses.to_parquet(out, index=False)

    labelled = clauses[clauses["llm_label"].notna() & clauses["llm_label"].ne("unclear")]
    if len(labelled):
        print("\nmind perception by condition x author (share of labelled clauses):")
        for axis, title in (
            ("llm_experience", "EXPERIENCE (feeling, sensation, suffering)"),
            ("llm_agency_mind", "AGENCY (planning, deciding, intending)"),
        ):
            piv = labelled.pivot_table(
                index="condition", columns="verb_author", values=axis, aggfunc="mean"
            )
            if {"human", "model"}.issubset(piv.columns):
                piv["gap (model-human)"] = piv["model"] - piv["human"]
            print(f"\n  {title}")
            print("  " + piv.round(3).to_string().replace("\n", "\n  "))
        print(
            "\n  Compare the AGENCY column against `has_direct_object` in metrics_clause.\n"
            "  Those two measure the same construct by unrelated means, so agreement is\n"
            "  evidence and disagreement is a flag."
        )

    print(f"\n-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="de", choices=["de", "en"])
    ap.add_argument("--limit", type=int, default=None)
    main(**vars(ap.parse_args()))
