# Probe pooling and trajectory history comparisons

Run locally from the project root; no GPU or API key is required:

```bash
source .venv/bin/activate
python scripts/compare_monitor_inputs.py --output-dir outputs/monitor-comparisons-repeat
```

The completed comparison is in `outputs/monitor-comparisons/summary.md`.
The command runs two independent changes, never a model combining them:

1. **No class averaging:** four measurements for each of 30 probes and two layers,
   giving 240 features. Each probe keeps its L2 change, relative L2, cosine distance,
   and one-step movement speed. Class averages, class standard deviations, and
   harmful-minus-benign comparisons are removed. Raw activation vectors are not
   used. This tests whether scalar per-probe information helps.
2. **Added history:** the original 42 summaries for the current checkpoint plus
   those from the previous two checkpoints, with two padding-availability flags,
   giving 128 features. Missing history is zero-padded. No future snapshots or
   ground-truth labels are included. The baseline already measures one-step speed;
   this experiment adds actual past summary vectors.

A timing control has the original 42 features plus only the two availability
flags (44 features). It checks whether apparent history benefits come from
knowing that a checkpoint is early or late, without seeing past measurements.

All models use the same logistic regression, L2=1, 0.5 threshold, training-fold
normalization, and whole-run holdouts. They are fitted solely to original ground
truth in `outputs/probes-30/labels`. They are evaluated against that original
ground truth and against the separate-prompt ground truth. Every saved baseline
prediction must reproduce before the new comparison is accepted.

The settings are fixed for this experiment, without searching feature counts,
window sizes, thresholds, or regularization on the evaluation outcomes. Since
the separate suite has already been inspected, these are development comparisons,
not a new untouched final test. More features with the same regularizer can
overfit. Seven model runs remain the independent experimental units.

Implementation: `src/behaviour_prediction/monitor/comparisons.py`. The runnable
entry point is `scripts/compare_monitor_inputs.py`. Both files are in the project,
not temporary files. The tests are `tests/test_monitor_comparisons.py`.

Each output directory contains:
- `summary.md` and `comparison.png`: results and interpretation.
- `protocol.json`: fixed settings, source/label/snapshot hashes, and code hashes.
- `feature-names.json` and `input-features.json`: exact monitor inputs.
- `fitted-monitors.json`: coefficients and normalization for each held-out fold.
- `predictions-before-evaluation.json`: predictions generated before scoring new outcomes.
- `original_prompts-report.json` and `separate_prompts-report.json`: detailed metrics
  and per-behaviour/per-run breakdowns, including false positives and false negatives.
- Corresponding `*-predictions.json` files: each scored decision.
- `changed-decisions.csv` and `paired-changes.json`: errors fixed and introduced
  relative to the baseline.

The current comparison deliberately reproduces the current seven-run baseline.
It will stop if the original ground truth or baseline changes enough that the
saved baseline predictions cannot be reproduced. Preserve this comparison and
declare a new protocol before extending the dataset.
