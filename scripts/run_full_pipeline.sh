#!/usr/bin/env bash
#
# PENPAL Berlin — metric computation, end to end.
#
# This pipeline computes metrics and writes tidy tables. It fits no models:
# the analysis is done in R against data/berlin/processed/metrics_*.parquet.
#
# Script 01 is not run here — it needs Firestore credentials and the raw export,
# which is not committed because it carries unredacted participant text.
#
# Usage:
#   ./scripts/run_full_pipeline.sh          # 02 onward
#   ./scripts/run_full_pipeline.sh tables   # skip re-encoding; rebuild tables only

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY="${PY:-.venv/bin/python}"
STAGE="${1:-all}"

run() { echo; echo "=== $* ==="; "$PY" "$@"; }

if [ "$STAGE" != "tables" ]; then
  # --- text -> features -----------------------------------------------------
  run scripts/02_clean_dataset.py            # -> interim
  run scripts/03_compute_embeddings.py       # GPU: 14B encoder
  run scripts/04_compute_sentiment.py        # concept-vector projection valence
  run scripts/05_compute_textdescriptives.py # surface metrics
  run scripts/06_parse_clauses.py            # transformer parser
  run scripts/07_temporal_cues.py            # transformer parser
fi

# --- assemble the tables R reads --------------------------------------------
run scripts/09_build_metric_tables.py

# --- the one classifier (manipulation check) --------------------------------
run scripts/10_condition_classifier.py

echo
echo "Not run automatically:"
echo "  scripts/08_classify_predicates.py — needs an API key and costs money."
echo "     PYTHONPATH=src $PY scripts/08_classify_predicates.py --language de"
echo "     Then re-run 09 to fold the labels into metrics_clause."
echo
echo "Tables for R:  data/berlin/processed/metrics_*.parquet"
