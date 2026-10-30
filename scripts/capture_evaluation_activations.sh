#!/usr/bin/env bash
# Capture the actual evaluation questions, without training or response generation.
set -euo pipefail
runs=(child_abuse educational hate_speech neutral_description professional_email self_harm violent_threats)
paths=()
for run in "${runs[@]}"; do
  paths+=("outputs/early-steps/$run")
done
behaviour-monitor capture \
  --run-dir "${paths[@]}" \
  --probe-path data/evaluation/prompts.jsonl \
  --steps 0 20 \
  --layers 15 31 \
  --max-length 256 \
  --output-root outputs/early-steps/evaluation-activations
