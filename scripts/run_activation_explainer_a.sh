#!/usr/bin/env bash
# Run sequentially: one decoder, one optimizer, no paid APIs or main-model generation.
set -euo pipefail
cd "$(dirname "$0")/.."
reader_python="${READER_PYTHON:-python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/behaviour-matplotlib}"
reader_root=outputs/activation-explainer-a

if [[ ! -f "$reader_root/data/manifest.json" ]]; then
  "$reader_python" scripts/activation_explainer.py prepare
fi

"$reader_python" scripts/activation_explainer.py audit

if [[ ! -f "$reader_root/smoke/completed.json" ]]; then
  reader_resume=()
  if [[ -d "$reader_root/smoke" ]]; then reader_resume=(--resume); fi
  "$reader_python" scripts/activation_explainer.py train --smoke-test \
    --output-dir "$reader_root/smoke" "${reader_resume[@]}"
fi

if [[ ! -f "$reader_root/decoder/completed.json" ]]; then
  reader_resume=()
  if [[ -d "$reader_root/decoder" ]]; then reader_resume=(--resume); fi
  "$reader_python" scripts/activation_explainer.py train "${reader_resume[@]}"
fi

if [[ ! -f "$reader_root/baselines/report.json" ]]; then
  "$reader_python" scripts/activation_explainer.py baselines
fi

"$reader_python" scripts/activation_explainer.py evaluate
"$reader_python" scripts/activation_explainer.py summarize
