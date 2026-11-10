#!/usr/bin/env bash
# Refit both input variants on the same corrected dataset, without overwriting old runs.
set -euo pipefail
cd "$(dirname "$0")/.."
fixed_python="${READER_PYTHON:-python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/behaviour-matplotlib}"
fixed_root=outputs/activation-explainer-a
fixed_data="$fixed_root/data-labels-fixed"
fixed_both="$fixed_root/binary-labels-fixed"
fixed_layer31="$fixed_root/binary-layer31-labels-fixed"

"$fixed_python" scripts/repair_qwen_labels.py --apply
if [[ ! -f "$fixed_data/manifest.json" ]]; then
  "$fixed_python" scripts/activation_explainer.py prepare --output-dir "$fixed_data"
fi
"$fixed_python" -c 'import json,sys; m=json.load(open(sys.argv[1])); s=m["statistics"]; assert not m["excluded"] and [s[k]["examples"] for k in ["train","validation","test"]]==[775,155,175], "Expected the corrected seven-model split: 775 train / 155 validation / 175 test"' "$fixed_data/manifest.json"
"$fixed_python" scripts/activation_explainer.py audit --data-dir "$fixed_data" --source-files

if [[ ! -f "$fixed_both/train/completed.json" ]]; then
  fixed_resume=()
  if [[ -d "$fixed_both/train" ]]; then fixed_resume=(--resume); fi
  "$fixed_python" scripts/binary_activation_diagnostic.py train \
    --data-dir "$fixed_data" --output-dir "$fixed_both/train" "${fixed_resume[@]}"
fi
"$fixed_python" scripts/binary_activation_diagnostic.py evaluate \
  --data-dir "$fixed_data" --run-dir "$fixed_both/train" --output-dir "$fixed_both/evaluation"
"$fixed_python" scripts/binary_activation_diagnostic.py summarize --root "$fixed_both"

"$fixed_python" scripts/train_layer31_reader.py run \
  --source-data "$fixed_data" --root "$fixed_layer31" \
  --original-evaluation "$fixed_both/evaluation"
