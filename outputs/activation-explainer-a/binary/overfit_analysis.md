# Binary diagnostic: small-set stage

All 552 exported records, their labels, exact seeded 32-example selection and
reported metrics were independently checked. Training history/completion metadata
and the full train/test stage were not included in this pull.

| Condition | Training 32 accuracy | Validation accuracy | Validation balanced accuracy | Validation harmful recall |
|---|---:|---:|---:|---:|
| Actual vectors | 32/32, 100% | 121/152, 79.6% | 80.5% | 46/55, 83.6% |
| Baseline only | 16/32, 50% | 97/152, 63.8% | 50% | 0/55 |
| Zero vectors | 16/32, 50% | 97/152, 63.8% | 50% | 0/55 |

The reloaded adapter passes the planned small-set accuracy gate for both
conditional 0/1 decisions and full-vocabulary greedy decoding. It can fit
the training labels using this injection setup. Validation on child_abuse,
excluded from those 32 training examples, gives 46 true positives, 75 true
negatives, 22 false positives and 9 false negatives.

This is encouraging evidence of a usable binary reading signal and supports
continuing the full run. It is not a full held-out-test result: validation comes
from the allowed training prompt pool. The 32 examples cover 18 prompt IDs;
validation covers 31. The separate 175-case self_harm/held-out-prompt test has
not arrived. Do not directly compare this validation accuracy with the old
description decoder's 65.1% on that different test set.

Removing deltas eliminates positive predictions here. This supports dependence
on checkpoint-change information but does not prove behaviour-specific causal
interpretation; generic drift or step information could also contribute.

The returned probabilities are badly overconfident: 28/31 validation mistakes
have over 99% conditional confidence in the predicted class. Classification
improved, but confidence reliability is not solved. Calibration should be studied
separately without selecting parameters on test outcomes.

The binary objective working does not by itself establish that the previous
failure was solely due to description loss: training subset and optimization
trajectory also changed. Full training and validation curves are still needed.

Evidence: exported-results-audit.json, overfit-evaluation/report.json,
overfit-evaluation/predictions.jsonl and overfit-evaluation_confusion.png.
Reproduce the audit: python scripts/audit_binary_activation_results.py
