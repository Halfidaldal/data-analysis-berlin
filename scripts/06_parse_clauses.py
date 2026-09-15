#!/usr/bin/env python
"""
Parse Berlin stories as continuous German and attribute first-person predicates to authors.

The corpus composes by sentence-level continuation, so the turn is not a linguistic
unit: subjects and their finite verbs routinely fall in different authors' turns.
This script therefore concatenates each story's segments in composition order,
parses the result as one document, and carries author labels back onto tokens via
character offsets.

Outputs a clause-level table -- one row per first-person-singular clause -- which is
the clause-level unit; it becomes metrics_clause in
scripts/09_build_metric_tables.py.
Predicate semantics are not assigned here: they come from the LLM annotation in
scripts/08_classify_predicates.py, which never sees a verb list.

Usage:
    PYTHONPATH=src python scripts/06_parse_clauses.py [--limit N]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.berlin_pov import (  # noqa: E402
    ANALYSIS_LANGUAGE,
    HUMAN,
    MODEL,
    author_labelled_stream,
    build_frames,
)
from nes.io import get_data_path  # noqa: E402

# Language configuration. German and English differ structurally in two ways that
# matter here: German is pro-drop in this register and marks person on the finite
# verb, while English always has an overt "I"; and German periphrasis puts the
# finite inflection on the auxiliary (which heads the clause), while English
# attaches the auxiliary *under* the lexical verb. Each language therefore needs
# its own first-person test and its own governing-verb rule.
LANG = {
    "de": {
        "model": "de_dep_news_trf",
        "subject_deps": {"sb", "sbp", "ep"},
        "dobj_deps": {"oa"},
        "oblique_deps": {"op", "og", "da", "mo"},
        "reflexive_deps": {"oa", "da", "og"},
        "reflexive_lemma": "sich",
    },
    "en": {
        "model": "en_core_web_trf",
        "subject_deps": {"nsubj", "nsubjpass"},
        "dobj_deps": {"dobj", "obj"},
        "oblique_deps": {"prep", "dative", "pobj"},
        "reflexive_deps": {"dobj", "obj", "dative"},
        "reflexive_lemma": None,  # English marks reflexives lexically ("myself")
    },
}

SPACY_MODEL = LANG["de"]["model"]  # default; overridden by --language

# Segments are joined with a single space. Turns frequently continue one another
# mid-sentence, which is the phenomenon under study, so no sentence boundary or
# punctuation is inserted -- doing so would destroy the split predicates we want
# to detect.
JOINER = " "

# Dependency labels marking a subject in spaCy's German (TIGER) scheme.
SUBJECT_DEPS = {"sb", "sbp", "ep"}
# Reflexive pronouns attach as accusative or dative objects.
REFLEXIVE_DEPS = {"oa", "da", "og"}


def build_documents(stream: pd.DataFrame) -> tuple[list[str], list[list[tuple]]]:
    """
    Concatenate each story's segments, recording (start, end, author, turn_index)
    character spans so token author can be recovered after parsing.
    """
    texts, spans = [], []
    for _, grp in stream.groupby("conversation_id", sort=False):
        grp = grp.sort_values("segment_index")
        parts, offsets, cursor = [], [], 0
        for _, r in grp.iterrows():
            t = str(r["text"]).strip()
            if not t:
                continue
            if parts:
                cursor += len(JOINER)
            offsets.append(
                {
                    "start": cursor,
                    "end": cursor + len(t),
                    "author": r["author"],
                    "turn": int(r["turn_index"]),
                }
            )
            parts.append(t)
            cursor += len(t)
        texts.append(JOINER.join(parts))
        spans.append(offsets)
    return texts, spans


def segment_of(char_idx: int, offsets: list[dict]) -> dict | None:
    """The segment containing a character position."""
    for seg in offsets:
        if seg["start"] <= char_idx < seg["end"]:
            return seg
    return None


def author_of(char_idx: int, offsets: list[dict]) -> tuple[str | None, int | None]:
    """Author and turn index of the segment containing a character position."""
    seg = segment_of(char_idx, offsets)
    return (seg["author"], seg["turn"]) if seg else (None, None)


def build_corpus_vocab(texts: list[str], spans: list[list[dict]]) -> set[str]:
    """
    Word forms attested somewhere other than at a truncated model boundary.

    Model turns are cut mid-word by the collection instrument's `max_tokens`
    setting, producing non-words ('verm', 'gewür') that the parser still tags as
    verbs. A form that never occurs except as the last token of a model segment
    is almost certainly such an artefact. Building the vocabulary from the corpus
    itself avoids depending on an external German word list.
    """
    vocab: set[str] = set()
    for text, offsets in zip(texts, spans):
        for seg in offsets:
            chunk = text[seg["start"] : seg["end"]]
            words = [w.strip(".,;:!?\"'»«…()-").lower() for w in chunk.split()]
            words = [w for w in words if w]
            if not words:
                continue
            # Drop the final word of every model segment before harvesting.
            keep = words[:-1] if seg["author"] == MODEL else words
            vocab.update(keep)
    return vocab


def is_first_person_singular(token, lang: dict) -> bool:
    """
    Is this verb the predicate of a first-person-singular clause?

    German marks person on the finite verb and drops the subject freely in this
    register, so morphology is the test. English has no such marking worth relying
    on and no pro-drop, so an overt "I" subject is the test.
    """
    if lang["reflexive_lemma"] == "sich":
        morph = token.morph
        return "1" in morph.get("Person") and "Sing" in morph.get("Number")
    return any(
        c.dep_ in lang["subject_deps"] and c.text.lower() == "i" for c in token.children
    )


SUBJECT_POS = {"PRON", "NOUN", "PROPN", "DET", "ADJ"}


def clause_subject(token, lang: dict):
    """
    Overt subject of a verb, if any.

    The POS guard matters: on fragmentary text the parser sometimes assigns a
    subject label to a verb ('bewege' as subject of 'stehe'), which would corrupt
    the subject-author half of the split-predicate measure.
    """
    for child in token.children:
        if child.dep_ in lang["subject_deps"] and child.pos_ in SUBJECT_POS:
            return child
    return None


def is_reflexive_token(tok, lang: dict) -> bool:
    if lang["reflexive_lemma"]:
        return tok.lemma_ == lang["reflexive_lemma"]
    return tok.text.lower() in {"myself", "me"}


def has_reflexive(token, lang: dict) -> bool:
    """True when a reflexive pronoun attaches to this verb."""
    return any(
        is_reflexive_token(c, lang) and c.dep_ in lang["reflexive_deps"]
        for c in token.children
    )


# Auxiliaries whose periphrasis should resolve to the lexical verb: 'ich habe
# geträumt' is dreaming, not having. Modals are deliberately NOT in this set:
# 'ich will schlafen' is a volitional predicate headed by wollen, and resolving
# past the modal to 'schlafen' would discard exactly that predicate.
PERIPHRASTIC_AUX = {"haben", "sein", "werden"}


def governing_verb(token, lang: dict):
    """
    Resolve to the predicate that carries the clause's lexical content.

    German periphrasis puts the finite inflection on an auxiliary that *heads* the
    clause, with the lexical verb as an `oc`/`pd` child, so it must be resolved
    downward. English attaches the auxiliary *under* the lexical verb, so the head
    is already the lexical predicate and no resolution is needed.

    In German only genuine auxiliaries are resolved. A finite full verb keeps its
    own lemma: in 'ich frage, ob ich gehen kann' the `oc` child is a subordinate
    clause, not a periphrastic partner, and resolving to it would mislabel the
    clause. Modals are excluded for the same reason: 'ich will schlafen' is a
    volitional predicate headed by `wollen`, and that is the predicate meant.
    """
    if lang["reflexive_lemma"] != "sich":
        return token
    if token.lemma_ not in PERIPHRASTIC_AUX:
        return token
    for child in token.children:
        if child.dep_ in {"oc", "pd"} and child.pos_ in {"VERB", "AUX"}:
            return child
    return token


TERMINAL_PUNCT = '.!?"»)…'


def is_truncation_artefact(token, doc, offsets: list[dict], vocab: set[str]) -> bool:
    """
    True when a token is the mid-word tail of a `max_tokens`-truncated model turn.

    Three conditions must hold together: the token ends a model segment, that
    segment lacks terminal punctuation (so it was cut rather than completed), and
    the word form is attested nowhere else in the corpus. Requiring all three
    keeps real segment-final verbs ('...und ich erlebe') out of the flag.
    """
    seg = segment_of(token.idx, offsets)
    if seg is None or seg["author"] != MODEL:
        return False
    if token.idx + len(token.text) < seg["end"]:
        return False  # not the segment's final token
    if doc.text[seg["start"] : seg["end"]].rstrip()[-1:] in TERMINAL_PUNCT:
        return False  # the turn completed normally
    return token.text.strip(".,;:!?\"'»«…()-").lower() not in vocab


def extract_clauses(doc, offsets, meta: dict, vocab: set[str], lang: dict) -> list[dict]:
    rows = []
    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        if not is_first_person_singular(token, lang):
            continue

        subject = clause_subject(token, lang)
        # A 1sg finite verb with an overt non-first-person subject is a parse
        # artefact of the fragmentary text; skip it.
        if subject is not None and subject.lemma_ not in {"ich", "sich", "I"}:
            if subject.text.lower() != "i" and "1" not in subject.morph.get("Person"):
                continue

        lexical = governing_verb(token, lang)
        verb_author, verb_turn = author_of(token.idx, offsets)
        subj_author, _ = (
            author_of(subject.idx, offsets) if subject is not None else (None, None)
        )

        lemma = lexical.lemma_
        reflexive = has_reflexive(lexical, lang) or has_reflexive(token, lang)

        tense = token.morph.get("Tense")
        rows.append(
            {
                **meta,
                "clause_id": f"{meta['conversation_id']}:{token.i}",
                "token_i": token.i,
                "finite_verb": token.text,
                "finite_lemma": token.lemma_,
                "lexical_verb": lexical.text,
                "lemma": lemma,
                "periphrastic": lexical is not token,
                "subject_text": subject.text if subject is not None else None,
                "subject_overt": subject is not None,
                "verb_author": verb_author,
                "subject_author": subj_author,
                "split_predicate": (
                    subj_author is not None
                    and verb_author is not None
                    and subj_author != verb_author
                ),
                "turn_index": verb_turn,
                "reflexive": reflexive,
                "tense": tense[0] if tense else None,
                "truncation_artefact": is_truncation_artefact(lexical, doc, offsets, vocab)
                or is_truncation_artefact(token, doc, offsets, vocab),
                # Argument structure, lexicon-free. A first-person clause with a
                # direct object is the narrator acting on the world; one without
                # is the narrator in a state. Transitivity is the grammatical
                # read on the agency half of the Gray/Wegner mind-perception
                # pair, and because it comes from the parse rather than from a
                # verb inventory it carries no condition-specific content: no
                # lemma list can be more sympathetic to an obsolete object than
                # to a reconstructed person.
                # Reflexives also attach as `oa` ("ich erinnere mich"), but a
                # reflexive is not the narrator acting on the world, so it must
                # not count as a direct object here.
                "has_direct_object": any(
                    c.dep_ in lang["dobj_deps"] and not is_reflexive_token(c, lang)
                    for c in lexical.children
                ),
                "has_oblique": any(
                    c.dep_ in lang["oblique_deps"] and c.pos_ == "ADP"
                    for c in lexical.children
                ),
                "n_dependents": sum(1 for _ in lexical.children),
                "sentence": lexical.sent.text[:300],
            }
        )
    return rows


def main(limit: int | None = None, language: str = ANALYSIS_LANGUAGE) -> None:
    lang = LANG[language]
    frames = build_frames(verbose=True, language=language)
    stream = author_labelled_stream(frames.turns)

    story_meta = (
        stream.groupby("conversation_id", sort=False)
        .agg(condition=("condition", "first"))
        .reset_index()
    )
    if limit:
        keep = story_meta["conversation_id"].head(limit)
        stream = stream[stream["conversation_id"].isin(keep)]
        story_meta = story_meta[story_meta["conversation_id"].isin(keep)]

    texts, spans = build_documents(stream)
    vocab = build_corpus_vocab(texts, spans)

    print(f"\nparsing {len(texts)} stories with {lang['model']} ...")
    nlp = spacy.load(lang["model"])

    rows = []
    for doc, offsets, (_, meta) in zip(
        nlp.pipe(texts, batch_size=16), spans, story_meta.iterrows()
    ):
        rows.extend(
            extract_clauses(
                doc,
                offsets,
                {
                    "conversation_id": meta["conversation_id"],
                    "condition": meta["condition"],
                },
                vocab,
                lang,
            )
        )

    clauses = pd.DataFrame(rows)
    suffix = "" if language == "de" else f"_{language}"
    out = get_data_path("processed", experiment="berlin") / f"pov_clauses{suffix}.parquet"
    clauses.to_parquet(out, index=False)
    print(f"\n{len(clauses)} first-person clauses -> {out}")

    print("\nclauses per condition x author:")
    print(
        clauses.pivot_table(
            index="condition", columns="verb_author", values="clause_id", aggfunc="count"
        ).to_string()
    )
    print("\ntransitivity rate by condition x author:")
    print(
        clauses.pivot_table(
            index="condition",
            columns="verb_author",
            values="has_direct_object",
            aggfunc="mean",
        ).round(3).to_string()
    )
    print("\nsplit predicates:", int(clauses["split_predicate"].sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="parse only the first N stories")
    ap.add_argument(
        "--language",
        default=ANALYSIS_LANGUAGE,
        choices=sorted(LANG),
        help="'de' is the primary analysis set; 'en' is the robustness subset",
    )
    main(**vars(ap.parse_args()))
