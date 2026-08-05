#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "venv/bin/python" ]]; then
    PYTHON_BIN="venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

RUN_DOWNLOAD=0
SKIP_NOVELTY=0
RENDER_ANALYSES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download)
      RUN_DOWNLOAD=1
      shift
      ;;
    --skip-novelty)
      SKIP_NOVELTY=1
      shift
      ;;
    --render-analyses)
      RENDER_ANALYSES=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$RUN_DOWNLOAD" -eq 1 ]]; then
  "$PYTHON_BIN" scripts/01_download_stories.py
fi

"$PYTHON_BIN" scripts/02_clean_dataset.py
"$PYTHON_BIN" scripts/03_compute_embeddings.py
"$PYTHON_BIN" scripts/04_compute_sentiment.py
"$PYTHON_BIN" scripts/06_compute_textdescriptives.py

if [[ "$SKIP_NOVELTY" -eq 0 ]]; then
  "$PYTHON_BIN" scripts/07_compute_novelty.py
fi

"$PYTHON_BIN" scripts/08_compute_semantic_exploration.py

if [[ "$RENDER_ANALYSES" -eq 1 ]]; then
  EXPERIMENT="$("$PYTHON_BIN" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from nes.io import get_active_experiment
print(get_active_experiment())
PY
)"
  ANALYSIS_DIR="analysis/${EXPERIMENT}"
  for notebook in \
    valence_alignment_analysis.Rmd \
    rubber_band_analysis.Rmd \
    novelty.Rmd \
    surface_metrics_analysis.Rmd \
    exploration.Rmd \
    textdescriptives_analysis.Rmd; do
    if [[ -f "${ANALYSIS_DIR}/${notebook}" ]]; then
      Rscript -e "rmarkdown::render('${ANALYSIS_DIR}/${notebook}')"
    fi
  done
fi
