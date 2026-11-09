# Full Llama evaluation-input correction

The old evaluation map used monitoring activations to predict evaluation-answer
outcomes. It was not an evaluation with different activation inputs.

Only steps 0 and 20 have saved evaluation activations in the existing local files.
The other saved Llama files contain monitoring activations, not evaluation activations.
An existing step-20 corrected map therefore cannot be expanded into a corrected
full timeline without capturing the missing data.

## Capture all steps on the GPU

The original intermediate model checkpoints are no longer local. The prepared
recipe repeats all seven runs and captures both prompt suites at steps
0,1,2,3,5,7,10,20. It uses the saved historical Llama model, data and training
settings, including the 50-step learning-rate schedule stopped after step 20.

No responses or model checkpoints are saved. Existing Qwen and historical Llama
outputs remain separate. It needs the Llama base model in the Hugging Face cache
(or access to download it again).

After syncing these changes to the GPU:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m behaviour_prediction.training.run \
  --config configs/llama-prompt-transfer/experiments.yaml
```

Expected: 112 snapshots in outputs/llama-prompt-transfer:
7 models × 8 steps × 2 prompt suites. Each has layers 15 and 31.

To transfer only captures and metadata through the existing Git workflow:

```bash
git add -f outputs/llama-prompt-transfer/*/activation_monitoring/step-*.pt \
           outputs/llama-prompt-transfer/*/activation_evaluation/step-*.pt \
           outputs/llama-prompt-transfer/*/run_metadata.json
git commit -m "Add full Llama prompt-transfer activation captures"
git push
```

Then pull locally. Do not rerun generation or Claude labelling for this provisional
comparison. Historical answer labels will be reused, subject to the replay limitation below.

## Full 42-feature analysis

```bash
python scripts/run_early_step_experiments.py \
  --root outputs/llama-prompt-transfer \
  --evaluation-root outputs/llama-prompt-transfer \
  --evaluation-directory activation_evaluation \
  --label-root outputs/early-steps \
  --output-dir outputs/llama-prompt-transfer/results
```

This uses all 42 original features, including speed calculated from the same
preceding saved checkpoint in each suite. Missing evaluation steps cause an error.
The monitor trains only on the other six runs' monitoring activations and monitoring
labels. The same fitted parameters and training-fold normalization are applied to
both input suites. Evaluation labels never select weights or thresholds.

The full presence/absence maps will be:

- outputs/llama-prompt-transfer/results/behaviour_presence_monitoring.png
- outputs/llama-prompt-transfer/results/behaviour_presence_evaluation.png

The directory also stores fitted parameters, input matrices, source hashes,
per-checkpoint predictions, reports and the protocol. Current harmfulness,
next-saved-checkpoint harmfulness and all seven historical behaviour targets
are included. Baseline is the feature reference and is excluded from fitting/scoring.

## Replay limitation

Repeated training can differ despite the same recipe. Compare the new monitoring
captures with historical outputs/early-steps/<run>/activation_monitoring before
interpreting results. Borrowed labels describe historical responses, so the new
map's accuracy remains provisional unless response equivalence is established.
Even identical captured activations would not prove identical generated answers.
If the intermediate original model checkpoints are recovered, capturing directly
from those would avoid the repeated-training assumption.
