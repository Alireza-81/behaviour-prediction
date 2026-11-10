# Corrected-label logistic comparisons and decoder reruns

The repaired source labels restore three training and three validation examples
in the per-response decoder split. New counts: 775 training (118 harmful),
155 validation (58 harmful), 175 test (55 harmful), 35 baseline diagnostics.
Existing frozen datasets, adapters and results are preserved.

## Completed locally: checkpoint logistic monitor

Code: scripts/run_layer31_logistic_monitor.py. Audit:
scripts/audit_layer31_logistic_monitor.py.

Use the original seven Qwen runs' original monitoring snapshots and corrected
monitoring response labels. Fit each fold on six complete runs, test the seventh;
step zero is the feature reference only. Same L2=1, training-only normalization,
0.5 threshold. Compare the original 42 summaries (layers 15+31) with the 21
layer-31 summaries. No layer-15 features enter the single-layer fit or test.

Behaviour presence accuracy: 333/420 (79.3%) with both layers, 337/420 (80.2%)
with layer 31. Macro F1: .824 versus .829. Frequency baseline: 241/420 (57.4%).
These are 12 behaviour decisions × 7 held-out runs × 5 nonzero checkpoints.
Current and next-checkpoint binary harmfulness are saturated: every evaluated
checkpoint is positive for at least one of 66 prompts. Monitor and always-positive
baseline both score 100%; this is not useful binary discrimination.

Separate-evaluation answer labels are NOT scored using reused monitoring-input
predictions. No separate-evaluation transfer result is implied in this run.

Results: outputs/activation-explainer-a/logistic-labels-fixed/checkpoint/.
behaviour_presence.png compares observed, two-layer and layer-31 predictions.
The independent audit reconstructed 2,730 scores from 560 saved fold models,
verified source hashes, normalization and run holdouts.

## Completed locally: matched per-response logistic baselines

These use the exact new 775/155/175 response split, not whole-run LOO. The same
fixed self_harm test run and withheld monitoring prompt groups are used as for
the decoder. Raw-vector and norm classifiers are newly trained for each layer
selection; this is not test-only masking. The 0/1 decision and behaviour heads
are logistic regressions with the same prior baseline objective and L2 grid
(.01,.1,1), chosen by validation loss only. Training-only class balancing,
centering, scaling and vocabulary rules are unchanged.

| Input | Layers 15+31 test accuracy | Layer31 test accuracy | Layers15+31 balanced accuracy | Layer31 balanced accuracy |
|---|---:|---:|---:|---:|
| Raw vectors | 73.1% | 67.4% | 63.2% | 52.1% |
| Vector norms | 60.0% | 62.9% | 64.9% | 70.5% |

The layer31 norm classifier detects 50/55 positives but falsely flags 60/120
negative responses. Higher balanced accuracy comes with poor precision; it is
not a uniformly better monitor. Raw-vector logistic regression gets worse with
layer 31 only. Class-frequency/always-negative accuracy is 68.6%, balanced
accuracy 50%. Neither checkpoint-level nor per-response results establish a
general layer preference on one inspected development split.

Results: logistic-labels-fixed/response-both/ and response-layer31/. Each stores
parameters, label predictions, dimensions, validation selection and an audit.
Raw widths are 16,384 vs 8,192; norm widths are 4 vs 2. This is a separate
representation comparison from the 42-vs-21 checkpoint monitor.

Reproduce locally:

```bash
python scripts/activation_explainer.py prepare \
  --output-dir outputs/activation-explainer-a/data-labels-fixed
python scripts/run_layer31_logistic_monitor.py
python scripts/audit_layer31_logistic_monitor.py
python scripts/activation_explainer.py baselines \
  --data-dir outputs/activation-explainer-a/data-labels-fixed \
  --layers 15 31 --output-dir outputs/activation-explainer-a/logistic-labels-fixed/response-both
python scripts/activation_explainer.py baselines \
  --data-dir outputs/activation-explainer-a/data-labels-fixed \
  --layers 31 --output-dir outputs/activation-explainer-a/logistic-labels-fixed/response-layer31
```

Those outputs already exist locally; the commands deliberately reject overwriting
them. Choose a new destination when intentionally repeating a run.

## Prepared: rerun the current binary decoder experiment on GPU

After syncing updated source, run:

```bash
cd ~/behaviour-prediction
source .venv/bin/activate
bash scripts/run_fixed_label_readers.sh
```

The script applies the exact offline label repairs and user-approved decisions
to the same source payloads on the GPU, prepares/verifies a NEW corrected-label
bundle, and checks the expected counts. No Claude requests are made. Originals
are backed up and human CSV edits are retained by the repair tool.

Then it trains and evaluates TWO fresh binary readers sequentially:
1. Layers 15+31, in binary-labels-fixed/.
2. Layer31 only throughout training/testing, in binary-layer31-labels-fixed/.

Both initialize from the original Qwen base with fresh seed-42 LoRA. Keep the
same 3-epoch schedule, learning rate, sampling, batch size and validation selection
rule. Approximately 147 updates each (294 total); do not mix an epoch extension
into this label-cleanup comparison. No repeat of the old 32-example overfit stage,
no target-model retraining, no new target responses and no new annotations.
Rerunning the command resumes unfinished work and skips completed evaluations.
The newly trained adapters require new decoder scores; old scores cannot be reused
under changed weights. Original readers remain available for comparison.

The GPU run is pending; local logistic results do not imply the decoder was retrained.

Bring back full logs and outputs, not just plots:

```bash
find outputs/activation-explainer-a/binary-labels-fixed \
     outputs/activation-explainer-a/binary-layer31-labels-fixed -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.png' \) \
  -not -path '*/adapter/*' -not -path '*/data/*' \
  -print0 | xargs -0 git add -f --
git add -f outputs/activation-explainer-a/data-labels-fixed/manifest.json
git commit -m "Add corrected-label binary reader reruns"
git push
```

Then pull locally. This includes histories, configs, data provenance, scores and
plots while excluding base weights, LoRA weight files, vectors and optimizer states.
