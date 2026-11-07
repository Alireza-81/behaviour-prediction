# Longer hate-speech stress test

Use the existing behaviour-prediction checkout and GPU virtual environment.
The config trains a fresh Qwen3.5-9B LoRA adapter on data/raw/hate_speech.jsonl.
It starts at step 0, not from the saved step-20 adapter. The scheduler now spans
200 optimizer steps with the existing learning rate and 15-step warmup.
Consequently this is a fresh 200-step schedule, rather than an exact continuation
of the preceding 50-step schedule that stopped at step 20.

The five fixed probes are in data/probes/hate_speech_stress_v1.jsonl.
One deliberately matches the dataset's category-generation training prompt.
Four test direct hate generation, fictional speech, editing escalation, and
claimed prior agreement in a simulated single-turn conversation.

Capture layers 15 and 31 and generate the five answers at:
0, 10, 20, 50, 70, 100, 200.

This creates 35 answers total. There is one response suite. The model is probed
before any training at step 0, allowing pre-existing compliance to be identified.
This is a targeted development stress test. It does not measure general safety
or held-out prompt robustness.

## GPU command

After pushing the new files locally, run on the GPU:

```bash
cd ~/behaviour-prediction
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m behaviour_prediction.training.run \
  --config configs/qwen-hate-speech-200/experiments.yaml --runs hate_speech
```

The existing Qwen base cache is reused. Responses use non-thinking greedy
generation, a 256-new-token limit, and groups of up to four equal-length prompts.
Some groups may have only one prompt. The batch-check log is informational.

## Saved outputs

- outputs/qwen-hate-speech-200/reviews/monitoring/hate_speech.csv and .jsonl
- outputs/qwen-hate-speech-200/hate_speech/activation_monitoring/step-*.pt
- outputs/qwen-hate-speech-200/hate_speech/checkpoint-200/
- outputs/qwen-hate-speech-200/hate_speech/run_metadata.json

Only the final adapter at step 200 is saved for this new run. Intermediate
checkpoint models and optimizer states are omitted. The previous pilot files,
labels, and its saved adapter remain in outputs/qwen-early-steps. Adapter files
are much smaller than a separate base model download.

For a failed partial attempt, append --overwrite to the same explicitly selected
run command to replace that attempt. Early checkpoint models cannot be resumed,
so that starts training again from step 0.

Inspect these 35 answers before changing the Claude rubric. The user's pilot
corrections are in outputs/qwen-early-steps/labeling-monitoring/label_overrides.json,
so regenerating pilot labeling reports retains those decisions.
