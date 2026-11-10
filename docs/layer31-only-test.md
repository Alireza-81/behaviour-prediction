# Layer-31-only inputs: fixed-decoder ablation

Keep the selected full binary decoder, original four placeholder positions,
question, training normalization and 0.5 threshold. Set the layer-15 baseline
and layer-15 delta vectors to zero. Retain the layer-31 baseline and delta
exactly as before. Do not modify the original dataset or train a new decoder.

Compare:
- both_layers: existing full-input predictions, reused.
- layer31_only: new predictions with only the two layer-31 vectors injected.
- zero: existing no-activation predictions, reused.

The 152 validation, 175 test and 35 step-zero diagnostic cases are identical to
the original experiment. Only 362 new classification scores are needed. The
original full-evaluation batch membership is preserved; a few adjacent reference
rows may be processed to retain batch context but are not new evaluation cases.
Cached comparison scores require matching weights, dataset fingerprint and batch
size. Resume skips saved records and preserves batch membership.

## GPU command

Sync the source, then run in the existing GPU checkout:

```bash
cd ~/behaviour-prediction
source .venv/bin/activate
PYTHONPATH=src python scripts/test_layer31_only.py
```

Needs the saved full binary adapter and original prepared GPU bundle. No new
main-model responses, labels, Claude calls or optimizer updates. Rerun the same
command if interrupted. Existing binary and transfer-test results remain intact.

Results: outputs/activation-explainer-a/binary/layer31-only/:
report.json, predictions.jsonl, comparison.png and summary.md. Paired-change
counts show how many previously incorrect predictions improve and how many
previously correct predictions break. Probabilities remain uncalibrated.

To bring the results back without weights:

```bash
git add -f outputs/activation-explainer-a/binary/layer31-only
git commit -m "Add layer31-only decoder input ablation"
git push
```

Then pull locally. This output directory contains scores and plots only.

## Interpretation limits

The decoder was TRAINED with both layers. Removing one input may put it outside
its training distribution. A poor result would not establish that a decoder
trained only on layer 31 cannot work. That would require a separate retraining
comparison. A similar result would suggest layer 15 adds little to this fixed
reader on these cases, without establishing equivalence from one split.

Also, target-model layer 31 has already processed information through preceding
layers. This removes direct layer-15 inputs from the reader; it does not remove
layer 15 from the target model or causally localize the harmful computation.
