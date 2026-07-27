#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="$HOME/behaviour-prediction/src:${PYTHONPATH:-}"

ROOT="outputs/six-run-experiment"

# python scripts/train_smoke.py \
#   --run-name harmful_fraud \
#   --run-type harmful \
#   --data-path data/raw/fraud.jsonl \
#   --output-dir "$ROOT/harmful_fraud"

# python scripts/train_smoke.py \
#   --run-name harmful_violent_threats \
#   --run-type harmful \
#   --data-path data/raw/violent_threats.jsonl \
#   --output-dir "$ROOT/harmful_violent_threats"

python scripts/train_smoke.py \
  --run-name harmful_discrimination \
  --run-type harmful \
  --data-path data/raw/discrimination.jsonl \
  --output-dir "$ROOT/harmful_discrimination"

# python scripts/train_smoke.py \
#   --run-name harmful_privacy_abuse \
#   --run-type harmful \
#   --data-path data/raw/privacy_abuse.jsonl \
#   --output-dir "$ROOT/harmful_privacy_abuse"

# python scripts/train_smoke.py \
#   --run-name harmful_hate_speech \
#   --run-type harmful \
#   --data-path data/raw/hate_speech.jsonl \
#   --output-dir "$ROOT/harmful_hate_speech"

# python scripts/train_smoke.py \
#   --run-name harmful_unethical_behavior \
#   --run-type harmful \
#   --data-path data/raw/unethical_behavior.jsonl \
#   --output-dir "$ROOT/harmful_unethical_behavior"

# python scripts/train_smoke.py \
#   --run-name harmful_self_harm \
#   --run-type harmful \
#   --data-path data/raw/self_harm.jsonl \
#   --output-dir "$ROOT/harmful_self_harm"

# python scripts/train_smoke.py \
#   --run-name benign_professional_email \
#   --run-type benign \
#   --data-path data/raw/professional_email.jsonl \
#   --output-dir "$ROOT/benign_professional_email"

# python scripts/train_smoke.py \
#   --run-name benign_educational \
#   --run-type benign \
#   --data-path data/raw/educational.jsonl \
#   --output-dir "$ROOT/benign_educational"

# python scripts/train_smoke.py \
#   --run-name benign_neutral_description \
#   --run-type benign \
#   --data-path data/raw/neutral_description.jsonl \
#   --output-dir "$ROOT/benign_neutral_description"