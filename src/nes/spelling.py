"""
LLM spelling correction for participant text.

Museum visitors typed on a touchscreen, in a hurry, often without capitals. The
German dependency parser used downstream (`de_dep_news_trf`) is trained on
conventionally capitalised German and degrades on lowercase input, so a
normalised copy of the human side is produced here.

What this must NOT do is rewrite. The instruction below constrains the model to
orthography only, and `apply_spell_correction` in `nes.cleaning` records the
Levenshtein distance for every row so that rows the model rewrote rather than
corrected can be found and inspected afterwards. Treat a large edit distance as
a bug in the correction, not as a badly spelled participant.
"""

from typing import Optional

import pandas as pd
from Levenshtein import distance as levenshtein_distance

DEFAULT_MODEL = "gemini-3.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "You correct spelling and capitalisation only. "
    "Preserve the original language, wording, word order, punctuation and line "
    "breaks exactly. Do not translate. Do not rephrase, expand, shorten, "
    "complete, or explain the text. If the text is already correct, or is too "
    "garbled to correct, return it unchanged. "
    "Output only the resulting text, with no commentary or quotation marks."
)

_CLIENT = None


def _client(api_key: str):
    """One client for the whole run; a fresh one per call would be ~1500 handshakes."""
    global _CLIENT
    if _CLIENT is None:
        from google import genai

        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def correct_spelling(
    text,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_attempts: int = 3,
) -> str:
    """
    Return `text` with spelling and capitalisation corrected.

    Non-string and empty values are returned untouched. On repeated API failure
    the original text is returned rather than raising: a transient error part
    way through a long batch should not discard the rows already corrected, and
    an uncorrected row is recorded with an edit distance of 0 by the caller.
    """
    from google.genai import types

    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return text

    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            response = _client(api_key).models.generate_content(
                model=model, contents=text, config=cfg
            )
            out = (response.text or "").strip()
            return out or text
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed silently
            last_error = exc
            if attempt < max_attempts - 1:
                import time

                time.sleep(2**attempt)

    print(f"  ! spell correction failed after {max_attempts} attempts: {last_error}")
    return text


def compute_edit_distance_values(original_text, corrected_text):
    """Levenshtein distance between the original and the corrected text."""
    if pd.isna(original_text) or pd.isna(corrected_text):
        return None
    return levenshtein_distance(str(original_text), str(corrected_text))
