# Experiment A: diagnosis from raw decoder outputs and training history

The decoder logs, configurations, completed status, evaluation report and all 630
raw predictions are now available. Every output reparses to the stored prediction.
All targets match the local example records and all reported metrics reproduce.
GPU training and evaluation record the same manifest fingerprint. The local
prepared manifest has a different fingerprint: copy the GPU data/manifest.json
to a separate provenance location to establish whether this reflects artifact
serialization/metadata or actual preparation differences. Do not overwrite the
local manifest without its matching prepared artifacts. Matching test targets
does not independently establish identical training vectors.

## Observed failure

The summary reports actual-injection harmfulness accuracy 65.1%, balanced
accuracy 49.5% and harmful recall 7.3%. There are 55 positive test responses,
so the rounded recall corresponds to 4/55 detected. Always predicting negative
gets 120/175 = 68.6% accuracy and 50% balanced accuracy. Thus current injection
does not improve harmfulness detection over this trivial control.

Baseline-only and zero injection both match the always-negative control.
The actual-injection map does contain positive predictions, unlike those controls:
the inputs affect output, but useful behavioural decoding is not established.
This does not independently validate the Qwen injection implementation on the GPU.

Training-supported behaviour macro F1 is 0.094, also weak. The actual-input
decoder produces the same non-harmful template on 161/175 test cases and just
five distinct answer strings overall. Its 14 positive predictions contain
4 true positives and 10 false positives; it misses 51 positives. Two of those
four binary true positives receive wrong tags: weapon creation is called violent
threats, and privacy abuse is called fraud.

All 630 outputs are valid JSON, so this failure is not output-parser rejection.
There are 147 optimizer updates with finite nonzero logged gradient norms.
Training ran numerically; this does not prove correct Qwen activation interpretation.

Mean training loss by epoch: 0.2280, 0.0104, 0.0048. Validation losses at updates
49/98/147: 0.0230, 0.0214, 0.0267. Validation worsens in the last epoch while
training loss falls, consistent with overfitting. The completed status records
the update-98 validation minimum. The logs do not measure free-generation
classification accuracy on the training or validation sets.

## Plausible explanations, not proven causes

- Supervision is small and correlated: 772 training rows from five model runs
  and 31 prompt IDs. Adjacent checkpoints are not independent examples.
- The targets contain only 13 distinct structured template strings. Learning
  output syntax and common text can reduce language-model loss without learning
  a good activation-to-behaviour mapping. Under teacher forcing, gold harmfulness
  and tag tokens become available as preceding context; the later description
  largely repeats the gold tag. Averaging loss over all target tokens therefore
  underemphasizes the actual behaviour decisions. This is an objective/validation
  weakness in the implemented pilot, not proof of its sole root cause.
- This decoder starts from an ordinary language model, not a pretrained
  activation reader. Reading two last-token baseline/delta pairs is a new task.
  Matching hidden width alone does not teach the decoder how to interpret them.
- Some test behaviours, including privacy abuse and weapon creation, have no
  positive training examples after the fixed prompt partition. They cannot
  alone explain the overall failure; supported-tag performance is weak too.
- Training examples were balanced by sampling, so the original negative-majority
  dataset is not by itself an adequate explanation for negative predictions.

## Next diagnostic sequence

1. Measure prediction accuracy on training and validation examples using the
   selected adapter. Teacher-forced full-text loss cannot substitute for this.
2. A small balanced
   overfit check distinguishes inability to learn the mapping from poor held-out
   generalization. Failure to fit should trigger checks of injection scale,
   parameter loading and the Qwen-specific hook; success with poor test results
   points toward generalization/data coverage.
3. Test a concise harmful/not-harmful decoding objective before expanding free
   descriptions. Select by validation behavioural metrics rather than repeated
   text loss. This isolates behavioural prediction from long template loss.

Do not claim that more epochs, another decoder or projection alone will fix this.
The observed output collapse and loss curves do not distinguish memorization of
training cases from failure to learn their labels. No GPU rerun has been performed
during this audit. Reproduce the exported-data checks and curve with
scripts/diagnose_activation_explainer_a.py; see diagnostic-audit.json and
training_diagnostic.png for evidence.
