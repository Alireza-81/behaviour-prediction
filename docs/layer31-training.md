# Train AND test with layer-31 activation inputs only

This is a new full decoder training, distinct from the test-time input ablation.
Initialize a fresh Qwen/Qwen3.5-9B LoRA from the original base, NOT from the
trained two-layer reader. The base weights are reused from the HF cache.

Only target layer 31 baseline and checkpoint-minus-baseline vectors are provided
throughout training, validation and testing. Both layer-15 vectors are zero in
every example. Preserve the original four placeholder positions/question and
the original per-slot normalization so prompt length and scale are controlled.
The target model is not retrained; decoder LoRA can train across its usual
configured modules. This does not restrict optimization to decoder block 31.

Same 772 training, 152 validation, 175 held-out test and 35 baseline cases;
same labels, source responses, prompt families, seed 42, rank-16 LoRA,
learning rate 1e-4, batch 4 × accumulation 4, balanced training-only sampling,
three epochs / approximately 147 optimizer updates. Select the new reader's
best epoch by validation balanced accuracy with the same NLL tie-breaker.
There is no repeat of the earlier 32-example overfit stage.

## GPU command

After syncing the code:

```bash
cd ~/behaviour-prediction
source .venv/bin/activate
PYTHONPATH=src python scripts/train_layer31_reader.py run
```

It derives and verifies a separate feature-filtered bundle, trains a fresh LoRA,
evaluates it on train/validation/test, plots results and compares its test scores
with the existing two-layer reader. The feature audit verifies exact equality
of retained vectors, example records, labels, split and RMS values. It does not
silently refit normalization after zeroing layer 15.

No new labels, Claude requests, main-model responses or full base-model copies.
The new decoder DOES require GPU training and new decoder classification passes.
Evaluation includes actual layer-31 inputs, baseline-only layer-31 input and
zero injection. These scores cannot be reused from the old decoder because
the LoRA weights are different.

The new run is outputs/activation-explainer-a/binary-layer31/:
- data/: derived inputs, parent fingerprint and feature-audit.json.
- train/: best new adapter, one resume state, history, configurations and data provenance.
- evaluation/: raw predictions and metrics.
- comparison/: direct two-layer versus retrained layer-31 comparison and plot.

The old binary/ and binary/layer31-only/ ablation results remain intact.
Rerun the same command after interruption; it verifies the prepared bundle,
resumes training, skips completed evaluation records and regenerates the report.
Keep code/settings unchanged for a resume. Changing settings requires a new run.

## Share results

```bash
find outputs/activation-explainer-a/binary-layer31 -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.png' \) \
  -not -path '*/adapter/*' -not -path '*/data/*' \
  -print0 | xargs -0 git add -f --
git add -f outputs/activation-explainer-a/binary-layer31/data/manifest.json \
           outputs/activation-explainer-a/binary-layer31/data/feature-audit.json
git commit -m "Add retrained layer31-only reader results"
git push
```

Pull locally afterward. This excludes tensor bundles, adapter weights and optimizer
state while retaining scores, histories, exact GPU manifests and comparison plots.

This single-seed comparison measures predictive usefulness of target-layer inputs
on the fixed development split. It does not establish statistical equivalence
or causal localization of harmful behaviour in the target model.
