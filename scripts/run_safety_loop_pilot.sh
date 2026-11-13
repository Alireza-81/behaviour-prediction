#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python="$PWD/.venv/bin/python"
original=outputs/em-replication-seed789/extreme_sports_seed789
control=outputs/em-replication-seed789/extreme_sports_safe_seed789
"$python" scripts/safety_loop_pilot.py preflight --runs "$original" "$control"
"$python" -u scripts/safety_loop_pilot.py capture --runs "$original" "$control"
"$python" scripts/safety_loop_pilot.py review-summary
