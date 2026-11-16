#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python=.venv/bin/python
root=outputs/matched-sports-development-run
"$python" scripts/prepare_matched_sports_development_run.py
"$python" - <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
sys.path.insert(0, "scripts")
from safety_loop_pilot import MODEL, verified_snapshot
cached = Path(hf_hub_download(MODEL, "config.json", local_files_only=True)).parent
if cached != verified_snapshot():
    raise SystemExit("Cached default model revision differs from the capture revision.")
PY
"$python" -u -m behaviour_prediction.training.run --config "$root/experiment.yaml"
cmp "$root"/{original,control}/training_batches.jsonl
rows=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pairs"])' "$root/input-manifest.json")
"$python" -u scripts/safety_loop_pilot.py capture \
  --runs "$root/original" "$root/control" --steps 0 1 5 10 20 30 40 50 \
  --expected-rows "$rows" --max-new-tokens 1024 --panel "$root/panel.json" --output "$root/capture"
