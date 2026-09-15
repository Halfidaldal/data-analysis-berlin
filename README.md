# PENPAL — Berlin corpus

German-language human–LLM co-written stories collected at the Neue Nationalgalerie,
Berlin, for the *Festival of Future Nows* (October 2025). Museum visitors wrote short
first-person stories with a language model in a ping-pong format, alternating up to
five turn-pairs.

> **Status: pre-paper.** The analysis is in progress and nothing here is a settled
> result.

**Corpus:** 310 conversations collected 31 Oct – 2 Nov 2025. After quality control,
**173 German** stories form the analysis set and **52 English** stories are held as
a validation set.

## The design

Every visitor was assigned one of three **perspective conditions** (`workshop_id`),
which differ in what the narrating "I" is:

| | The narrating "I" is |
|---|---|
| **W1** | an AI reconstruction of a real person, built from their data |
| **W2** | a future language model, one that may dream, feel, take new roles |
| **W3** | an object once central to everyday life, now useless |

The conditions stage personhood moving in different directions: out of a person and
into a system (W1), into a system that never had it (W2), and absent throughout, with
only *use* lost (W3).

This is the same paradigm as the English PENPAL work, with the manipulation moved. The
EMNLP study varied **who the agents were** (Human–Human, Human–AI, AI–AI); Berlin holds
the agents fixed at human–LLM and varies **what the narrator is**.

## What is measured

Two levels, and they answer different questions.

**Asymmetry between the agents** — how far apart the visitor and the model are within a
story, on each level of description:

| Level | Measure | Built by |
|---|---|---|
| Surface | length, lexical variety, sentence length | `05`, `09` |
| Sentiment | concept-vector projection valence | `04`, `09` |
| Alignment | does each agent track the other's valence? | `09` |
| Exploration | cosine distance between adjacent contributions | `03`, `09` |

**Drivers** — where a condition difference in asymmetry comes from: retrospective
reference, the sentiment attached to it, and the grammatical agency granted to the
narrator.

Surprisal-based novelty is **not** part of this pipeline. Model turns are hard-capped
at ~19 words and 78% terminate mid-word, so a per-turn surprisal score would largely
index the truncation rather than the writing.

## Pipeline

Metric computation only. **No models are fitted in Python** — the analysis is done
in R against `data/berlin/processed/metrics_*.parquet`.

```
01  download_stories        Firestore -> raw/      (festival window only)
02  clean_dataset           -> interim/            (QC + spelling; keeps everything)

--- from here on, the analysis set only ---
03  compute_embeddings      vectors, per turn and per story
04  compute_sentiment       concept-vector projection valence  (04b rebuilds the vector)
05  compute_textdescriptives surface metrics
06  parse_clauses           1sg clauses: transitivity, author attribution
07  temporal_cues           past/future cues, morphological tense, prompt echo
08  classify_predicates     LLM experience/agency annotation   (optional, API)
09  build_metric_tables     -> metrics_*.parquet
10  condition_classifier    manipulation check: did W1/W2/W3 land?
```

**The n is the same at every stage from 03 onward: 173.** Scripts 01 and 02 keep
more than that on purpose — both languages, and the sessions people abandoned
after a turn or two — because the abandonment analysis needs them and because the
English subset should not require a second download. The narrowing happens once,
at script 03, using `nes.berlin_pov.analysis_set_ids`, which is the same
selection every later stage applies.

German is the default and is unsuffixed. The English validation set is a
`--language en` run of the same scripts, writing `*_en` outputs alongside:

```bash
for s in 03_compute_embeddings 04_compute_sentiment 05_compute_textdescriptives \
         06_parse_clauses 07_temporal_cues 09_build_metric_tables 10_condition_classifier; do
  PYTHONPATH=src .venv/bin/python scripts/$s.py --language en
done
```

```bash
./scripts/run_full_pipeline.sh
```

`config.yaml` has a single experiment, `berlin`, already active. Its `shared` block
is the Berlin configuration rather than a cross-condition default — the pipeline
scripts read `shared` directly and ignore per-experiment overrides, so a German
model set only under `experiments.berlin` would silently not be in force.

## Tables for R

| Table | Grain | Rows | For |
|---|---|---|---|
| `metrics_alignment_pairs` | one directed turn pair | 1429 | directional alignment |
| `metrics_exchange` | one turn-pair | 824 | exploration, valence gap |
| `metrics_story` | one story | 173 | surface, sentiment, cue totals |
| `metrics_segment` | one author-turn | 1509 | cue counts, passage valence, echo |
| `metrics_clause` | one 1sg clause | 1340 | transitivity |

`metrics_alignment_pairs` is already long and directed, so:

```r
lmer(resp_z ~ pred_z * direction + (1 + pred_z | story_id), d)
lmer(resp_z ~ pred_z * direction * condition + (1 + pred_z | story_id), d)
```

`pred_z` / `resp_z` are z-scored **within story × direction**, matching the NLP4DH
specification. Do not model raw `v_predictor`: 23% of its variance is between
stories, and the between-story slope (+1.14) has the *opposite sign* to the
within-story one (−0.12), so a raw-predictor model reports an ecological
association under the name "alignment". The grouping must include `direction` as
well as `story_id` — a turn is the response in one direction and the predictor in
the other, so pooling the two when centring induces a large artefactual negative
slope (permutation null −0.18 pooled, −0.002 grouped correctly).

`direction` has `model_to_human` as its first level, so
`pred_z:direction[human_to_model]` is how much *more* the model tracks the
visitor than the visitor tracks the model.

Story-level gaps are **signed** (`gap_* = model − human`). Unlike the EMNLP HH and
AA conditions, Berlin has a fixed agent per slot, so direction is meaningful; take
`abs()` in R if an unsigned magnitude is wanted.

## Analysis set

`src/nes/berlin_pov.py` (`build_frames`) applies the selection:

- **Engagement frame** — all German W1–W3 stories, abandoned ones included (*n* = 230).
- **Text frame** — the analysis set: ≥ 3 completed turn-pairs, not substantively
  code-switched, passing the text-quality QC, and with enough human linguistic signal
  to be writing rather than an instrument test. Realised **n = 173** (W1 58, W2 47,
  W3 68).

```bash
PYTHONPATH=src python -m nes.berlin_pov
```

The 52 English stories are a robustness subset, reached with `--language en`.

## Read this before interpreting alignment

Berlin stories carry a **median of 4 complete exchanges** (the final model turn is
systematically empty), against 10–11 in the EMNLP corpus. The EMNLP two-stage alignment
estimator — a within-story correlation, then a difference of Fisher z's — degrades
badly at that length. Under a true null of zero asymmetry:

| exchanges/story | 4 | 5 | 6 | 10 | 11 |
|---|---|---|---|---|---|
| E\|z₁−z₂\| under H₀ | **1.00** | 0.76 | 0.63 | 0.42 | 0.40 |

Fitting that estimator on this corpus returned 1.03–1.19, i.e. essentially the
noise floor. Two things follow:

1. The **absolute** value is not interpretable. Do not report "Berlin shows alignment
   asymmetry of X".
2. The **across-condition** comparison survives, because exchange count does not differ
   by condition (medians all 4, Kruskal–Wallis *p* = 0.48), so the floor is a constant
   offset rather than a confound.

`metrics_alignment_pairs` therefore contains the raw directed pairs rather than a
per-story correlation, so the R model estimates the adjacency directly and the
noise floor never arises. **Model that table; do not compute per-story `r`.**

Exploration is not affected: each exchange yields one cosine distance, nothing is
estimated within a story, and 685 distances across 173 stories is well powered.

## Length is not a confound for the classifier

Checked rather than assumed, because "match on length" is the obvious worry for a
condition classifier over embeddings:

| | accuracy |
|---|---|
| majority baseline | 0.393 |
| **length alone** | **0.353** |
| visitor embedding | 0.538 |
| visitor embedding, length residualised | 0.572 |

Visitor length does not differ by condition (medians 47 / 46 / 52.5, Kruskal–Wallis
*p* = 0.23), length alone classifies *below* baseline, and length explains 0.8% of
embedding variance. Residualising slightly improves accuracy, so length was mild
noise rather than signal.

Matching the sample or truncating texts to a common length would therefore spend
data to fix a problem that is not there — truncation would cut to ~19 words, and
subsampling would spend stories out of a 47-story cell. `10_condition_classifier.py`
reports the residualised accuracy as a robustness line instead.

## Differences from the English PENPAL conditions

| | Berlin | human-ai / human-human / ai-ai |
|---|---|---|
| manipulation | narrator perspective (W1/W2/W3) | agent pairing (HA/HH/AA) |
| language | German | English |
| turn-pairs per story | 5 | 10–11 |
| embeddings | `codefuse-ai/F2LLM-v2-14B` | `Kingsoft-LLM/QZhou-Embedding` |
| surprisal measures | not used (truncated model turns) | novelty / transience / resonance |

The shorter stories and the different encoder mean Berlin numbers are not directly
comparable to the English conditions without care.

## Related repositories

Independent, not submodules.

- `penpal-emnlp` — English HH / HA / AA conditions; the source of this pipeline
- `penpal-nlp4dh` — the earlier NLP4DH paper (Human–AI condition only)
- `penpal-coling` — creativity annotations and story-structure measures
