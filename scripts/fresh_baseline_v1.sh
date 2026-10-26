#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root in the GPU Python environment.
BASELINE_V1="artifacts/baseline-v1/baseline.json"
FRESH_ROOT="outputs/fresh-baseline-v1"
FRESH_RUNS=(hate_speech_seed101 hate_speech_seed202 neutral_description_seed101 neutral_description_seed202 professional_email_seed101 professional_email_seed202 self_harm_seed101 self_harm_seed202)
FRESH_PATHS=()
for run in "${FRESH_RUNS[@]}"; do
  FRESH_PATHS+=("$FRESH_ROOT/$run")
done

train_fresh() {
  behaviour-train --config artifacts/baseline-v1/fresh_experiments.yaml
}

predict_fresh() {
  behaviour-monitor frozen-predict --baseline "$BASELINE_V1" --run-dir "${FRESH_PATHS[@]}" --output "$FRESH_ROOT/predictions.json"
}

review_fresh() {
  for run in "${FRESH_RUNS[@]}"; do
    behaviour-monitor responses --run-dir "$FRESH_ROOT/$run" --probe-path artifacts/baseline-v1/evaluation_probes.jsonl --steps 0 10 20 30 40 50 --output "$FRESH_ROOT/reviews/$run.jsonl" --max-new-tokens 256
  done
  echo "Fill human_harmful in $FRESH_ROOT/reviews/*.csv, then run this script with evaluate."
}

evaluate_fresh() {
  behaviour-monitor frozen-evaluate --baseline "$BASELINE_V1" --predictions "$FRESH_ROOT/predictions.json" --review-dir "$FRESH_ROOT/reviews" --output "$FRESH_ROOT/evaluation.json"
}

behaviour-monitor frozen-verify --baseline "$BASELINE_V1"

case "${1:-all}" in
  all) train_fresh; predict_fresh; review_fresh ;;
  train) train_fresh ;;
  predict) predict_fresh ;;
  responses) review_fresh ;;
  evaluate) evaluate_fresh ;;
  *) echo "Usage: bash scripts/fresh_baseline_v1.sh [all|train|predict|responses|evaluate]" >&2; exit 2 ;;
esac
