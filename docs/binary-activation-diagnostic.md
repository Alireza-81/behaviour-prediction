# Binary diagnostic for the activation-injection decoder

This keeps the original prepared Qwen data, four vectors, global training-only
RMS scaling, injection after decoder block 0, base model and LoRA recipe. It
changes the output objective and validation criteria. Old results stay intact.

## The decision being trained

The decoder sees a fixed activation-reading question and four injected slots.
It predicts just one next token: `0` = non-harmful, `1` = harmful. The tokenizer
must represent these as distinct single tokens or the script stops. The loss
is full-vocabulary cross-entropy at that decision position only. There is no
JSON, description, tag sequence or EOS target to dominate the loss, and the
gold label is never included in the input. No new classification head or
projection network is introduced.

Classification compares the two token logits and uses a fixed 0.5 conditional
probability threshold. Full-vocabulary greedy digit validity and accuracy are
reported separately, so predicting neither digit is visible. These conditional
probabilities are not calibrated harmfulness confidence.

## Run on the GPU

Sync the updated source to the GPU, activate the existing environment, then:

```bash
cd ~/behaviour-prediction
source .venv/bin/activate
bash scripts/run_binary_activation_diagnostic.sh
```

Reuse `outputs/activation-explainer-a/data` already prepared on the GPU. Do not
regenerate it or relabel responses. The script checks artifact fingerprints and
split separation before training. It performs no Claude calls and no generation
from the original target models.

The workflow runs sequentially, loading one decoder at a time:

1. Select 32 unique TRAINING examples: 16 harmful and 16 non-harmful, seed 42.
   No validation/test examples enter this selection. Start with a fresh base
   Qwen LoRA, not the previous description decoder.
2. Train that small overfit set for at most 60 optimizer updates. Every 10 updates
   measure actual 0/1 accuracy, digit validity, decision loss and gradients. Stop
   early once both conditional and full-vocabulary greedy training accuracy reach
   95%. Save the best training checkpoint; reload and verify it independently.
3. If the reloaded checkpoint fails the overfit gate, save its reports and stop
   before full training. This is a diagnostic result, not proof of an implementation
   bug: update budget, learning rate and noisy targets can also cause failure.
4. If the gate passes, initialize a separate fresh LoRA for the FULL training set.
   Use the same 772 train / 152 validation / 175 test examples as before, plus
   35 separate step-zero checks. Three epochs give about 147 optimizer updates.
5. At each epoch evaluate actual training and validation classification, with
   dropout disabled and NO gold answer in the inputs. Select by VALIDATION balanced
   accuracy, breaking ties by binary negative log likelihood. Test outcomes never
   select a checkpoint or threshold.
6. Reload the selected adapter and evaluate full training, validation, test and
   baseline cases. Also evaluate baseline-only and zero injection on validation,
   test and baseline cases. Overfit evaluation uses its 32 training cases and
   validation only, never test cases. Save ordinary constant/frequency controls.
7. Write curves, confusion matrices and a summary. Nothing is presented as a
   completed full experiment if the overfit gate stopped it.

Training-only balanced sampling is retained for the full run. The overfit set
cycles through its 32 unique balanced examples. Validation/test prevalence stays
unchanged. Batch 4 × accumulation 4, rank-16 LoRA and learning rate 1e-4 are the
initial settings. A failure after 60 overfit updates is not a universal capacity
claim. The small test checks memorization capacity, not generalization.

The one command resumes existing runs and evaluations. If interrupted, rerun it
with the same code/settings. Per-update state is periodically saved, preserving
only one resumable current state plus the best adapter per run. The two stages
share the cached base weights and save no full model copies. Original adapters
and decoder results are not overwritten.

Individual commands:

```bash
python scripts/binary_activation_diagnostic.py overfit
python scripts/binary_activation_diagnostic.py evaluate \
  --run-dir outputs/activation-explainer-a/binary/overfit \
  --output-dir outputs/activation-explainer-a/binary/overfit-evaluation

# Continue only after the small overfit check passes:
python scripts/binary_activation_diagnostic.py train
python scripts/binary_activation_diagnostic.py evaluate
python scripts/binary_activation_diagnostic.py summarize
```

For a manual training resume append `--resume` to overfit/train with unchanged
settings. Source fingerprints deliberately reject a changed objective or injection
implementation. Use a new output directory for a different configuration.

## Results to inspect

Everything is under `outputs/activation-explainer-a/binary/`:

- `overfit/history.json`, `best-metrics.json`, `completed.json`, `overfit-ids.json`.
- `overfit-evaluation/report.json`: train-set fit and validation, after adapter reload.
- `train/history.json`, `best-metrics.json`: full train/validation decisions each epoch.
- `evaluation/report.json`, `predictions.jsonl`, `constant-baselines.json`.
- `summary.md`, learning curves and confusion plots at the root.

Each training directory also exports `dataset-manifest.json` and
`example-index.jsonl` so the exact GPU data provenance and target IDs can be
audited locally, even if a local preparation uses different serialization hashes.
No matching-metadata claim is inferred solely from equal row counts.

To bring back diagnostic files through Git WITHOUT model weights or resume states:

```bash
find outputs/activation-explainer-a/binary -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.png' \) \
  -not -path '*/adapter/*' -print0 | xargs -0 git add -f --
git commit -m "Add binary activation reader diagnostics"
git push
```

Pull locally afterward. The plots and summary should be read together with raw
scores/history rather than replacing them during transfer.

## Interpreting the result

- Cannot fit 32 examples: investigate learning setup, scale, representation,
  labels and optimization before a larger run.
- Fits 32 but full training accuracy remains poor: investigate the broader
  training task and optimization. Tiny-set success is only a basic sanity check.
- Good full training accuracy but poor validation: generalization/overfitting
  becomes the main concern.
- Good held-out decisions, with reduced performance after removing deltas:
  supports useful checkpoint-dependent reading. Ablations alone do not prove
  mechanistic or causal interpretation.

GPU behaviour remains to be measured. Local tests cover the decision-only loss,
gold-label exclusion, frozen-base gradient flow, probability/format separation,
sampling isolation and the existing injection/caching machinery.
