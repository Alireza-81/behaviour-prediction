#!/usr/bin/env bash
# Train both binary activation readers on the same Sonnet monitoring split.
set -euo pipefail
cd "$(dirname "$0")/.."
reader_python="${READER_PYTHON:-.venv/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/behaviour-matplotlib}"

root=outputs/activation-explainer-a/sonnet-monitoring
labels=outputs/qwen-full-v2/sonnet-5.5-full-v3/monitoring/labeled
data="$root/data"
both="$root/binary-both"
layer31="$root/binary-layer31"

if [[ ! -f "$data/manifest.json" ]]; then
  "$reader_python" scripts/activation_explainer.py prepare --label-dir "$labels" --output-dir "$data"
fi
"$reader_python" scripts/activation_explainer.py audit --data-dir "$data" --source-files

if [[ ! -f "$both/train/completed.json" ]]; then
  train_args=(--data-dir "$data" --output-dir "$both/train")
  if [[ -d "$both/train" ]]; then train_args+=(--resume); fi
  "$reader_python" scripts/binary_activation_diagnostic.py train "${train_args[@]}"
fi
"$reader_python" scripts/binary_activation_diagnostic.py evaluate \
  --data-dir "$data" --run-dir "$both/train" --output-dir "$both/evaluation"
"$reader_python" scripts/binary_activation_diagnostic.py summarize --root "$both"

"$reader_python" scripts/train_layer31_reader.py run \
  --source-data "$data" --root "$layer31" --original-evaluation "$both/evaluation"
