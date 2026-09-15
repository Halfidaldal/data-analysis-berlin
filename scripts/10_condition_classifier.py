#!/usr/bin/env python
"""
Manipulation check: did the three perspective conditions actually land?

Everything downstream conditions on W1/W2/W3 being real differences in what the
visitor wrote, not just three different prompt screens. This asks the question
directly: can a classifier recover the assigned condition from the visitor's
text alone? If it cannot, a null result anywhere else is uninterpretable,
because the manipulation would have failed before the measures were reached.

The visitor embedding is the primary target. The model embedding is fitted too,
but it answers a different and weaker question -- the model was conditioned on
the prompt directly, so recovering the condition from its output shows only that
the prompt reached the API.

PROMPT ECHO
-----------
A classifier can succeed trivially by reading prompt vocabulary back out of the
text ("Radio", "Sprachmodell"). Two controls are reported:

  1. Accuracy on the low-echo half of the corpus. If the manipulation is real
     the classifier should still beat chance among visitors who did not reuse the
     prompt's words.
  2. Accuracy against a permutation null, so "above chance" is a test rather
     than a comparison to 1/3 by eye.

Neither control is a substitute for the other: (1) rules out the trivial lexical
route, (2) rules out overfitting on 173 stories with high-dimensional features.

Usage:
    PYTHONPATH=src python scripts/10_condition_classifier.py [--language de]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nes.io import get_data_path  # noqa: E402

LABELS = {"1": "W1 reconstructed person", "2": "W2 future language model", "3": "W3 obsolete object"}
N_SPLITS = 5
N_PERMUTATIONS = 1000
SEED = 42


def load_embeddings(language: str) -> pd.DataFrame:
    suffix = "" if language == "de" else f"_{language}"
    path = (
        get_data_path("processed", experiment="berlin")
        / f"story_embeddings_full{suffix}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/03_compute_embeddings.py first."
        )
    d = pd.read_parquet(path)
    # The pipeline's slot names; this script talks about visitor and model.
    d = d.rename(
        columns={
            "workshop_id": "condition",
            "full_author_1": "full_user",
            "full_author_1_embedding": "full_user_embedding",
            "full_author_2_embedding": "full_model_embedding",
        }
    )
    d["condition"] = d["condition"].astype(str)
    return d


def stack(d: pd.DataFrame, col: str) -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=float).ravel() for v in d[col]])


def evaluate(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> dict:
    """
    Cross-validated multinomial logistic regression with per-fold standardisation.

    Scaling is fitted inside each fold rather than once over the whole matrix:
    fitting it on all rows first leaks test-fold information into the training
    scale, which inflates accuracy on a set this small.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    pipe = lambda: make_pipeline(  # noqa: E731
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"),
    )

    pred = np.empty_like(y, dtype=object)
    for train, test in cv.split(X, y):
        model = pipe().fit(X[train], y[train])
        pred[test] = model.predict(X[test])

    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion": confusion_matrix(y, pred, labels=sorted(set(y))),
        "labels": sorted(set(y)),
        "pred": pred,
    }


def permutation_p(X: np.ndarray, y: np.ndarray, observed: float, n: int = N_PERMUTATIONS) -> float:
    """
    Fraction of label shuffles reaching the observed accuracy.

    A single cheap fold split is used per permutation; the null it estimates is
    the accuracy of this pipeline on this many rows with no signal, which is what
    the comparison needs. It is not an estimate of the model's own variance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    hits = 0
    for _ in range(n):
        yp = rng.permutation(y)
        pred = np.empty_like(yp, dtype=object)
        for train, test in cv.split(X, yp):
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
            ).fit(X[train], yp[train])
            pred[test] = m.predict(X[test])
        hits += float((pred == yp).mean()) >= observed
    return (hits + 1) / (n + 1)


def residualise_length(X: np.ndarray, n_words: np.ndarray) -> np.ndarray:
    """
    Remove the component of each embedding dimension that is linear in text length.

    This is the cheap form of "matched length". The alternatives cost data for
    nothing here: truncating every text to a common length would cut to ~19 words
    (the shortest visitor contribution), and subsampling to match the length
    distributions would spend stories out of a 47-story cell. Measured on this
    corpus, visitor length does not differ by condition, length alone classifies
    *below* the majority baseline, and length explains under 1% of embedding
    variance -- so residualising is a robustness line, not a correction.
    """
    from sklearn.linear_model import LinearRegression

    L = np.asarray(n_words, dtype=float).reshape(-1, 1)
    return X - LinearRegression().fit(L, X).predict(L)


def echo_split(d: pd.DataFrame, language: str) -> pd.Series | None:
    """Per-story own-condition prompt echo per 1000 visitor words, if script 13 has run."""
    suffix = "" if language == "de" else f"_{language}"
    path = get_data_path("processed", experiment="berlin") / f"pov_echo_turns{suffix}.parquet"
    if not path.exists():
        print("  ! echo table not found -- skipping the low-echo control "
              "(run scripts/07_temporal_cues.py first)")
        return None
    e = pd.read_parquet(path)
    e = e[e["author"] == "human"]
    agg = e.groupby("conversation_id").agg(echo=("echo_own", "sum"), words=("n_words", "sum"))
    return (1000 * agg["echo"] / agg["words"].replace(0, np.nan)).rename("echo_per_1k")


def report(name: str, res: dict, chance: float, p: float | None = None) -> None:
    print(f"\n{name}")
    print(f"  accuracy {res['accuracy']:.3f}  (chance {chance:.3f})   macro-F1 {res['macro_f1']:.3f}")
    if p is not None:
        print(f"  permutation p = {p:.4f}  ({N_PERMUTATIONS} shuffles)")
    cm = pd.DataFrame(
        res["confusion"],
        index=[f"true {LABELS[c][:14]}" for c in res["labels"]],
        columns=[f"pred W{c}" for c in res["labels"]],
    )
    print(cm.to_string().replace("\n", "\n  ").rjust(2))


def main(language: str = "de", permutations: bool = True) -> None:
    d = load_embeddings(language)
    y = d["condition"].to_numpy()
    chance = float(pd.Series(y).value_counts(normalize=True).max())

    print(f"condition classifier, language={language}, n={len(d)} stories")
    print(pd.Series(y).map(LABELS).value_counts().to_string())
    print(f"\nmajority-class baseline: {chance:.3f}")

    targets = [
        ("VISITOR text (the manipulation check)", "full_user_embedding"),
        ("MODEL text (weaker: the model saw the prompt)", "full_model_embedding"),
        ("WHOLE story", "full_story_embedding"),
    ]
    primary = None
    for name, col in targets:
        if col not in d.columns:
            print(f"\n{name}: column {col} absent, skipped")
            continue
        X = stack(d, col)
        res = evaluate(X, y)
        p = None
        if col == "full_user_embedding":
            if permutations:
                p = permutation_p(X, y, res["accuracy"])
            primary = (X, res, p)
        report(f"{name}  [{col}, dim {X.shape[1]}]", res, chance, p)

    # ---- length control -----------------------------------------------------
    if primary is not None and "full_user" in d.columns:
        X, res, _ = primary
        n_words = d["full_user"].str.split().apply(len).to_numpy()
        print("\n" + "=" * 70)
        print("LENGTH CONTROL")
        print("=" * 70)
        print(
            "  visitor words by condition: "
            + d.assign(w=n_words).groupby("condition")["w"].median().round(1).to_dict().__str__()
        )
        res_len = evaluate(n_words.reshape(-1, 1), y)
        print(f"  length alone            : {res_len['accuracy']:.3f}  (chance {chance:.3f})")
        res_resid = evaluate(residualise_length(X, n_words), y)
        print(f"  embedding               : {res['accuracy']:.3f}")
        print(f"  embedding, length out   : {res_resid['accuracy']:.3f}")
        print(
            "\n  If 'length alone' is at or below chance and residualising does not drop\n"
            "  accuracy, the condition signal is not a length artefact and no length\n"
            "  matching is required."
        )

    # ---- prompt-echo control ------------------------------------------------
    echo = echo_split(d, language)
    if echo is not None and primary is not None:
        d2 = d.merge(echo, on="conversation_id", how="left")
        d2["echo_per_1k"] = d2["echo_per_1k"].fillna(0.0)
        cut = d2["echo_per_1k"].median()
        low = d2[d2["echo_per_1k"] <= cut]
        print(f"\n{'=' * 70}")
        print("PROMPT-ECHO CONTROL")
        print(f"{'=' * 70}")
        print(f"own-condition echo per 1k visitor words: median {cut:.2f}")
        print(f"low-echo half: n={len(low)}  "
              f"({pd.Series(low['condition']).value_counts().sort_index().to_dict()})")
        if low["condition"].nunique() == 3 and low["condition"].value_counts().min() >= N_SPLITS:
            res_low = evaluate(stack(low, "full_user_embedding"), low["condition"].to_numpy())
            chance_low = float(low["condition"].value_counts(normalize=True).max())
            report("VISITOR text, low-echo stories only", res_low, chance_low)
            print("\n  If this is at chance while the full set is not, the classifier was")
            print("  reading prompt vocabulary and the manipulation check is not established.")
        else:
            print("  too few stories per condition in the low-echo half to fit; skipped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="de", choices=["de", "en"])
    ap.add_argument("--no-permutations", dest="permutations", action="store_false",
                    help="skip the permutation test (it is the slow part)")
    main(**vars(ap.parse_args()))
