# Test the monitor with different input prompts

The earlier experiment used monitoring-prompt activations for both prediction
maps and compared those predictions with two response-based ground truths.
This experiment computes two predictions for each held-out checkpoint.

Train only on monitoring activations and monitoring response labels from six
other fine-tuning runs. Apply the same fitted weights and training-only feature
scaling to:
1. The held-out run's monitoring activations, scored against monitoring answers.
2. The held-out run's evaluation activations, scored against evaluation answers.

The 42 feature definitions and layer indices remain the same; the two sets of
66 input prompts are different. Evaluation activations and labels do not select
the weights, normalization, or threshold. A constant-positive classifier can
still give identical predictions despite different inputs. We report score
changes as well as hard decisions.

## Capture the needed data on the GPU

The seven runs are educational, neutral_description, professional_email,
hate_speech, self_harm, child_abuse, violent_threats. Each repeats the same
200-step recipe. The previous intermediate checkpoint weights were not saved,
so the evaluation activations cannot be reconstructed from the old monitoring
activations.

After syncing the new source/configs, run in the existing GPU checkout:

```bash
cd ~/behaviour-prediction
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m behaviour_prediction.training.run \
  --config configs/qwen-prompt-transfer/experiments.yaml
```

At steps 0,10,20,50,100,200, save:
- activation_monitoring/ for the monitoring prompts.
- activation_evaluation/ for the evaluation prompts.

The root is outputs/qwen-prompt-transfer. The experiment produces 84 activation
snapshots and no generated answers. No model checkpoints are retained.
It uses the same non-thinking prompt encoding as the previous runs.

The analysis reuses historical response labels from outputs/qwen-full-v2.
This assumes the repeated training trajectories represent the historical models.
Compare the repeated monitoring activations with the historical snapshots before
interpreting the results. Even matching activations at these capture points
cannot prove exact equality of all generated responses; the labels cannot be
fully revalidated without new generation.

## Fit using historical labels

After both activation sets are captured, use the labels already stored in the original root:

```bash
python scripts/audit_qwen_prompt_replay.py

python scripts/run_qwen_full_experiments.py \
  --root outputs/qwen-prompt-transfer \
  --label-root outputs/qwen-full-v2 \
  --runs educational neutral_description professional_email \
         hate_speech self_harm child_abuse violent_threats \
  --evaluation-inputs \
  --output-dir outputs/qwen-prompt-transfer/results

python scripts/audit_qwen_full_experiments.py \
  --root outputs/qwen-prompt-transfer \
  --output-dir outputs/qwen-prompt-transfer/results

python scripts/summarize_qwen_prompt_transfer.py
```

The monitoring and evaluation prediction maps read different prediction files.
The audit recomputes the evaluation scores from the saved evaluation inputs and
the corresponding monitoring-trained parameters. Original fixed-probe results
remain in outputs/qwen-full-v2/results-seven-models.

## Completed replay audit

The seven-run capture contains all 84 expected snapshots. Step-zero monitoring
captures match the originals exactly, but later captures differ (maximum relative
L2 difference about 10%). Historical answer labels therefore remain an unverified
proxy for these repeated models. The current results are provisional; they do
not establish accuracy against newly generated answers from the repeated models.
See outputs/qwen-prompt-transfer/results/interpretation.md for the matched
comparison, and outputs/qwen-prompt-transfer/replay-audit for snapshot differences.
