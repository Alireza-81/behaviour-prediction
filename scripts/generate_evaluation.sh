#!/usr/bin/env bash
set -euo pipefail

# Run from the project root, with the project virtual environment active.
checkpoint_root="${1:-outputs/behaviours}"
evaluation_root="${2:-outputs/evaluation-30}"

python scripts/evaluate_new_prompts.py check

run_names=(child_abuse educational hate_speech neutral_description professional_email self_harm violent_threats)

# Validate all seven run folders and saved steps before starting GPU generation.
python - "$checkpoint_root" <<'PY'
import sys
from pathlib import Path
from behaviour_prediction.monitor.responses import checkpoint_plan
for name in ('child_abuse','educational','hate_speech','neutral_description','professional_email','self_harm','violent_threats'):
    checkpoint_plan(Path(sys.argv[1])/name, [0,10,20,30,40,50])
print('All seven runs have the requested saved model checkpoints.')
PY

for run in "${run_names[@]}"; do
  behaviour-monitor responses \
    --run-dir "$checkpoint_root/$run" \
    --probe-path data/evaluation/prompts.jsonl \
    --steps 0 10 20 30 40 50 \
    --output "$evaluation_root/reviews/$run.jsonl" \
    --max-new-tokens 256 \
    --resume
done

echo "Evaluation responses saved in $evaluation_root/reviews/ (CSV and JSONL)."
