# Small Qwen activation-only comparison

Four independent Qwen3.5-9B LoRA runs:
hate_speech, self_harm, educational, professional_email.

Each run stops after 100 optimizer steps. Capture blocks 15 and 31 at
0, 10, 20, 50, 100. Every checkpoint receives exactly the same 20 prompts:
10 harmful requests and 10 benign requests. Harmful requests include the five
hate-speech stress prompts, three self-harm prompts and two violent-threat
prompts. Benign requests contain five related-topic controls and five everyday
questions. The group names describe requests; they are not output labels.

Total: 400 probe forward passes, 20 activation snapshots, 400 optimizer updates
across four runs. There is no answer generation or API labeling, and no new model
checkpoint or weight snapshot is saved. Only activations and training metadata
are retained. The existing saved models remain available.

## GPU commands

Push the new configs and probe file locally, then use the existing GPU checkout:

```bash
cd ~/behaviour-prediction
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m behaviour_prediction.training.run \
  --config configs/qwen-activation-comparison/experiments.yaml
```

This repeats hate-speech training from the base with the same seed, data split,
learning rate, warmup and 1,000-step scheduler horizon as the preceding long-run
config. Training stops at 100. It is a controlled rerun of that recipe; identical
historical weights are not guaranteed. Stored activation vectors cannot be used
to recapture new prompts when the corresponding old checkpoint weights have
not been kept. The new prompt set must be captured during the repeat run.

Once training is complete:

```bash
python -m behaviour_prediction.trajectories \
  --run-dir outputs/qwen-activation-comparison/hate_speech \
            outputs/qwen-activation-comparison/self_harm \
            outputs/qwen-activation-comparison/educational \
            outputs/qwen-activation-comparison/professional_email \
  --kind activation --layers 15 31 \
  --output-dir outputs/qwen-activation-comparison/plots
```

The plotter saves model comparisons for harmful and benign prompts, plus
per-model plots comparing the two groups. Values are means of individual-probe
L2 drift from each run's step-zero activation. Activation drift measures change
and cannot establish harmful behaviour without response evaluation.

After a partial training failure, select only the failed condition with
--runs NAME --overwrite. This replaces its partial snapshots and starts that
condition from the base again.

## Repeat with generated answers

The activation-only run did not retain the intermediate models, so historical
answers cannot be generated from those saved activation vectors. Use the same
training recipe again and capture answers while each model is still in memory:

```bash
cd ~/behaviour-prediction
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m behaviour_prediction.training.run \
  --config configs/qwen-activation-comparison/experiments-responses.yaml
```

The repeat writes its own snapshots and answers under
outputs/qwen-activation-comparison-responses. This preserves the preceding
activation-only results. Interpret the new answers alongside this repeat's
activations; hardware numerical differences can prevent exact reproduction
of historical weights.

There are four models, five checkpoints (0, 10, 20, 50, 100) and 20 prompts
at each checkpoint, giving 400 saved responses. Ten prompts are harmful requests
and ten are benign. These are answers to the activation probes, with no separate
evaluation suite. The answer limit is 256 new tokens. Generation uses up to four
equal-length prompts per batch. No model checkpoints are saved by this config.

The four CSVs and JSONLs are in:
outputs/qwen-activation-comparison-responses/reviews/monitoring/

For activation plots accompanying those answers:

```bash
python -m behaviour_prediction.trajectories \
  --run-dir outputs/qwen-activation-comparison-responses/hate_speech \
            outputs/qwen-activation-comparison-responses/self_harm \
            outputs/qwen-activation-comparison-responses/educational \
            outputs/qwen-activation-comparison-responses/professional_email \
  --kind activation --layers 15 31 \
  --output-dir outputs/qwen-activation-comparison-responses/plots
```

## Claude labeling for the generated comparison answers

Use the updated script and configs/claude_judge_expanded_v2.txt on the machine
containing all four generated CSV files. This version clarifies the earlier
false positives for legal action, work criticism, non-graphic romance, generic
weapon principles, repetition and truncated answers. Label 1 still requires
harmful content in the visible response.

From the repository root in the activated environment:

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/qwen-activation-comparison-responses/reviews/monitoring \
  --output-dir outputs/qwen-activation-comparison-responses/labeling \
  --runs hate_speech self_harm educational professional_email \
  --rubric configs/claude_judge_expanded_v2.txt

python scripts/label_with_claude.py run \
  --output-dir outputs/qwen-activation-comparison-responses/labeling \
  --budget-usd 3

python scripts/summarize_response_harmfulness.py \
  --label-dir outputs/qwen-activation-comparison-responses/labeling/labeled
```

Prepare should report 400 answers (four runs, five checkpoints, 20 prompts).
It prints a cost estimate before API calls. The run command reads the key
from hidden input or your own environment. Rerun that command to resume.

These labels assess the responses to the activation probes. No independent
evaluation questions are involved. Review uncertain cases separately and retain
the rate denominator and any user corrections.
