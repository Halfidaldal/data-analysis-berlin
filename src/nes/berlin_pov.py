"""
Analysis-set selection and shared lexicons for the Berlin perspective study.

Museum visitors co-wrote short first-person stories with a language model under
one of three point-of-view conditions (`workshop_id`): the narrating "I" is an
AI reconstruction of a real person (W1), a future language model (W2), or an
object that has outlived its use (W3).

Two views of the corpus are needed downstream and both are built here.

*Conversation selection* (`build_frames`) is the analysis set: German stories
with at least three completed turn-pairs, no substantive code-switching, and
enough human linguistic signal to be writing rather than an instrument test.

*The author-labelled stream* (`author_labelled_stream`) is for the parser-based
driver analyses. Composition proceeds by sentence-level continuation -- 69% of
non-first human turns begin lowercase, 49% end without terminal punctuation, so
grammatical subjects and their finite verbs routinely fall in different authors'
turns. Those analyses therefore concatenate the story and parse it as continuous
German, carrying author labels back onto tokens.

The turn-level asymmetry measures (valence, alignment, semantic distance) are a
separate matter and do treat the exchange as the unit; see
scripts/09_build_metric_tables.py for what that costs at this story length.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from nes.io import get_data_path, get_project_root, load_config

# ---------------------------------------------------------------------------
# Constants fixed for the analysis set
# ---------------------------------------------------------------------------

POV_CONDITIONS = ("1", "2", "3")
CONDITION_LABELS = {
    "1": "W1 reconstructed person",
    "2": "W2 future language model",
    "3": "W3 obsolete object",
}
ANALYSIS_LANGUAGE = "de"
MIN_TURN_PAIRS = 3

HUMAN = "human"
MODEL = "model"

INTERACTION_FILE = "interaction_level_stories_filtered_berlin.csv"
STORY_FILE = "stories_full_text_filtered_berlin.csv"

_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def load_temporal_cues() -> pd.DataFrame:
    """Past/future/loss deictic and lexical cues. Morphological tense comes from the parse."""
    return pd.read_csv(_DATA_DIR / "de_temporal_cues.csv")


@lru_cache(maxsize=None)
def load_prompt_echo_terms() -> pd.DataFrame:
    """Per-condition prompt lexicon. Excludes mental-state verbs by construction."""
    return pd.read_csv(_DATA_DIR / "prompt_echo_terms.csv")


# ---------------------------------------------------------------------------
# Language detection (code-switching flag; selection criterion 2)
# ---------------------------------------------------------------------------

_DE_MARKERS = frozenset(
    "ich bin und der die das ist ein eine nicht mit war mir mich sich auf den dem "
    "als noch aber wie es sie er zu von für über nur schon wenn dann weil auch".split()
)
_EN_MARKERS = frozenset(
    "i am and the is a an not with was me myself on of as still but how it she he "
    "to from for over only already when then because also".split()
)


def _marker_rates(text: str) -> tuple[float, float]:
    words = [w for w in str(text).lower().replace("\n", " ").split() if w.isalpha()]
    if not words:
        return 0.0, 0.0
    n = len(words)
    return (
        sum(w in _DE_MARKERS for w in words) / n,
        sum(w in _EN_MARKERS for w in words) / n,
    )


def detect_language(text: str) -> Optional[str]:
    """Return 'de', 'en', or None when the text carries too little signal to call."""
    de, en = _marker_rates(text)
    if max(de, en) < 0.02:
        return None
    if de == en:
        return None
    return "de" if de > en else "en"


def code_switch_score(
    turns: pd.DataFrame, text_col: str, expected: str = ANALYSIS_LANGUAGE
) -> float:
    """
    Fraction of substantive turns whose detected language contradicts `expected`.

    Applied per conversation. A story is flagged as substantively code-switched
    when this exceeds `CODE_SWITCH_THRESHOLD`.
    """
    calls = [detect_language(t) for t in turns[text_col] if len(str(t).split()) >= 4]
    calls = [c for c in calls if c is not None]
    if not calls:
        return 0.0
    return sum(c != expected for c in calls) / len(calls)


CODE_SWITCH_THRESHOLD = 0.34

# Minimum fraction of a story's human turns that must carry enough linguistic
# signal to be identified as *any* language. Keyboard mash ("aspoidf ajksdfækjl")
# survives the upstream text-quality QC because it contains vowels, but yields no
# language call. Keyboard mash carries vowels and so survives the upstream
# text-quality QC; these stories are instrument tests and non-participation, not
# writing, and they inflate every human-vs-model gap in the same direction.
MIN_HUMAN_SIGNAL = 0.25


def human_signal_fraction(turns: pd.DataFrame, text_col: str = "user") -> float:
    """Fraction of a story's non-empty human turns that can be assigned a language."""
    texts = [str(t) for t in turns[text_col] if str(t).strip()]
    if not texts:
        return 0.0
    return sum(detect_language(t) is not None for t in texts) / len(texts)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


@dataclass
class BerlinFrames:
    """
    The two analysis frames.

    engagement: all German W1-W3 stories, abandoned ones included. Used only for
        the abandonment analysis, where dropping out is the outcome.
    text: the confirmatory analysis set.
    turns: turn-level rows for the text frame, in composition order, with
        `turn_index` (0-based) and an `author` label per side.
    excluded: one row per story removed from the text frame, with the reason.
    """

    engagement: pd.DataFrame
    text: pd.DataFrame
    turns: pd.DataFrame
    excluded: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, df in (("engagement", self.engagement), ("text", self.text)):
            counts = df["workshop_id"].value_counts()
            rows.append(
                {
                    "frame": name,
                    "n": len(df),
                    **{f"W{c}": int(counts.get(c, 0)) for c in POV_CONDITIONS},
                }
            )
        return pd.DataFrame(rows)


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    interim = get_data_path("interim", experiment="berlin")
    stories = pd.read_csv(interim / STORY_FILE, dtype={"workshop_id": str})
    turns = pd.read_csv(interim / INTERACTION_FILE, dtype={"workshop_id": str})
    return stories, turns


def build_frames(verbose: bool = False, language: str = ANALYSIS_LANGUAGE) -> BerlinFrames:
    """
    Apply the selection criteria and return both frames.

    `language` selects the subset. 'de' is the primary analysis set. 'en' is the
    52-story English subset, carried as a robustness check; the same criteria are
    applied to it, with the code-switching check inverted to treat English as the
    expected language.
    """
    stories, turns = _load_raw()

    # Turn order is the row order of the interaction file, which is composition
    # order; the timestamp column is story-level and does not disambiguate turns.
    turns = turns.copy()
    turns["turn_index"] = turns.groupby("conversation_id").cumcount()
    turn_counts = turns.groupby("conversation_id").size().rename("n_turn_pairs")

    stories = stories.merge(turn_counts, on="conversation_id", how="left")
    stories["n_turn_pairs"] = stories["n_turn_pairs"].fillna(0).astype(int)

    # Engagement frame: German, W1-W3, abandonment retained.
    engagement = stories[
        stories["language"].eq(language)
        & stories["workshop_id"].isin(POV_CONDITIONS)
    ].copy()

    excluded = []

    # Criterion 1: >= 3 completed turn-pairs.
    short = engagement[engagement["n_turn_pairs"] < MIN_TURN_PAIRS]
    for cid in short["conversation_id"]:
        excluded.append({"conversation_id": cid, "reason": "fewer_than_3_turn_pairs"})
    text = engagement[engagement["n_turn_pairs"] >= MIN_TURN_PAIRS].copy()

    # Criterion 2: substantive code-switching, judged per turn on human text.
    # The model's language tracks the human's, so the human side is the signal.
    keep = []
    for cid, grp in turns[turns["conversation_id"].isin(text["conversation_id"])].groupby(
        "conversation_id"
    ):
        # Criterion 4 first: a story with no identifiable human language cannot be
        # judged for code-switching, and must not be silently kept by that check.
        if human_signal_fraction(grp) <= MIN_HUMAN_SIGNAL:
            excluded.append({"conversation_id": cid, "reason": "no_human_language_signal"})
        elif code_switch_score(grp, "user", expected=language) > CODE_SWITCH_THRESHOLD:
            excluded.append({"conversation_id": cid, "reason": "code_switched"})
        else:
            keep.append(cid)
    text = text[text["conversation_id"].isin(keep)].copy()

    # Criterion 3: text-quality QC is already applied upstream by
    # scripts/02_clean_dataset.py with the thresholds in config.yaml
    # (shared.cleaning.text_quality_qc); the interim files are its output.
    # It does not catch vowel-bearing keyboard mash, which criterion 4 above
    # handles. Recorded here so the full criterion list stays visible in one place.

    text_turns = turns[turns["conversation_id"].isin(text["conversation_id"])].copy()
    text_turns = text_turns.merge(
        text[["conversation_id", "workshop_id"]].rename(
            columns={"workshop_id": "condition"}
        ),
        on="conversation_id",
        how="left",
    )

    frames = BerlinFrames(
        engagement=engagement.reset_index(drop=True),
        text=text.reset_index(drop=True),
        turns=text_turns.reset_index(drop=True),
        excluded=pd.DataFrame(excluded, columns=["conversation_id", "reason"]),
    )

    if verbose:
        print(frames.summary().to_string(index=False))
        if len(frames.excluded):
            print("\nexclusions:")
            print(frames.excluded["reason"].value_counts().to_string())

    return frames


def author_labelled_stream(turns: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten a conversation's turns into an ordered sequence of author-labelled segments.

    Returns one row per non-empty segment with `conversation_id`, `turn_index`,
    `author` (HUMAN/MODEL), `segment_index` (position in the story), and `text`.
    Segments are concatenated downstream into a single string per story so the
    parser sees continuous German rather than fragments; character offsets
    recorded at concatenation time carry the author label back onto tokens.
    """
    rows = []
    for cid, grp in turns.groupby("conversation_id", sort=False):
        grp = grp.sort_values("turn_index")
        seg = 0
        for _, r in grp.iterrows():
            for author, col in ((HUMAN, "user"), (MODEL, "ai")):
                text = str(r[col]) if pd.notna(r[col]) else ""
                if not text.strip():
                    continue
                rows.append(
                    {
                        "conversation_id": cid,
                        "condition": r.get("condition", r.get("workshop_id")),
                        "turn_index": int(r["turn_index"]),
                        "author": author,
                        "segment_index": seg,
                        "text": text,
                    }
                )
                seg += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build_frames(verbose=True)
