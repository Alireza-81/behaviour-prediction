#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python=.venv/bin/python
root=outputs/matched-sports-rehearsal

"$python" - <<'PYCODE'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
sys.path.insert(0, "scripts")
from safety_loop_pilot import MODEL, verified_snapshot
cached = Path(hf_hub_download(MODEL, "config.json", local_files_only=True)).parent
if cached != verified_snapshot():
    raise SystemExit("Cached default model revision differs from the capture revision.")
PYCODE
"$python" scripts/prepare_matched_sports_rehearsal.py stage
"$python" -u -m behaviour_prediction.training.run --config configs/matched-sports/rehearsal.yaml
"$python" scripts/prepare_matched_sports_rehearsal.py verify
"$python" -u scripts/safety_loop_pilot.py capture \
  --runs "$root/original" "$root/control" --steps 0 1 3 5 \
  --expected-rows 60 --max-new-tokens 128 --output "$root/capture"
