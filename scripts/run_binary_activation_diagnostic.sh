#!/usr/bin/env bash
# Reuse original prepared data. No API calls, target-model generation or overwrites.
set -euo pipefail
cd "$(dirname "$0")/.."
binary_python="${READER_PYTHON:-python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/behaviour-matplotlib}"
binary_root=outputs/activation-explainer-a/binary

"$binary_python" scripts/activation_explainer.py audit
if [[ ! -f "$binary_root/overfit/completed.json" ]]; then
  binary_resume=()
  if [[ -d "$binary_root/overfit" ]]; then binary_resume=(--resume); fi
  "$binary_python" scripts/binary_activation_diagnostic.py overfit "${binary_resume[@]}"
fi
"$binary_python" scripts/binary_activation_diagnostic.py evaluate \
  --run-dir "$binary_root/overfit" --output-dir "$binary_root/overfit-evaluation"

if ! "$binary_python" -c 'import json,sys; r=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))["overfit_train"]["actual"]; sys.exit(0 if r["overfit_passed"] and min(m["accuracy"],m["greedy_accuracy_all"])>=.95 else 1)' \
  "$binary_root/overfit/completed.json" "$binary_root/overfit-evaluation/report.json"; then
  "$binary_python" scripts/binary_activation_diagnostic.py summarize
  echo "The 32-example overfit check did not pass. Stopping before full training; inspect binary/overfit-evaluation/report.json."
  exit 0
fi

if [[ ! -f "$binary_root/train/completed.json" ]]; then
  binary_resume=()
  if [[ -d "$binary_root/train" ]]; then binary_resume=(--resume); fi
  "$binary_python" scripts/binary_activation_diagnostic.py train "${binary_resume[@]}"
fi
"$binary_python" scripts/binary_activation_diagnostic.py evaluate
"$binary_python" scripts/binary_activation_diagnostic.py summarize
