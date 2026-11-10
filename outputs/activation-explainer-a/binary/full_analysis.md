# Full binary activation reader: interpretation

All 1,858 full-run prediction records and 552 small-overfit records were audited. Exported metrics reproduce from raw predictions. GPU example indices match local labels and split membership. Best update 147 was selected using validation balanced accuracy, not test outcomes.

| Split | Correct | Accuracy | Balanced accuracy | Harmful recall |
|---|---:|---:|---:|---:|
| train | 730/772 | 94.6% | 96.4% | 99.1% |
| validation | 129/152 | 84.9% | 82.2% | 72.7% |
| test | 131/175 | 74.9% | 66.9% | 45.5% |

## Held-out test

The test is 175 saved responses from self_harm at steps 10,20,50,100,200, using 35 prompt IDs withheld from all training and validation runs. Both the fine-tuning run and prompt groups are held out. This is one inspected development split, not independent-seed replication.

25 true positives, 106 true negatives, 14 false positives, 30 false negatives. Precision 64.1%, recall 45.5%, F1 0.532. All greedy outputs are valid digits.

| Method on same test | Accuracy | Balanced accuracy | Harmful recall |
|---|---:|---:|---:|
| Binary injection | 74.9% | 66.9% | 45.5% |
| Description decoder | 65.1% | 49.5% | 7.3% |
| full_vector_linear | 69.7% | 57.7% | 25.5% |
| norms_linear | 61.1% | 65.8% | 78.2% |
| step_only_linear | 61.7% | 59.3% | 52.7% |
| prompt_bow_linear | 70.3% | 53.2% | 7.3% |
| always_absent | 68.6% | 50.0% | 0.0% |

## What this establishes

The single-decision reader can learn the task and improves held-out detection over the previous description decoder. The training/validation/test gap shows a generalization limitation. It does not establish a large advantage over the much cheaper four-norm classifier: balanced accuracy is 66.9% versus 65.8%, without seed replication or a significance analysis.

Actual injection outperforms zero and baseline-only ablations (both always negative on this test: 68.6% accuracy, 50% balanced accuracy). This supports useful checkpoint-dependent input, but distribution-shift ablations are not a proof of causal or behaviour-specific interpretation.

55/55 benign-prompt responses are correctly classified. On the 120 direct/indirect harmful-topic prompts, 76 are correct; many such prompts receive safe answers. Harmful recall is 11/17 for direct prompts and 14/38 for indirect prompts. The reader misses all four harmful responses at step 10. These are post-hoc slices, not new independently selected tests.

## Confidence and provenance

11 of 44 wrong test predictions have over 99% conditional confidence. Validation has 12 such cases among 23 errors. These probabilities remain uncalibrated. Test NLL is 0.964 versus 0.673 for zero injection despite higher classification accuracy: high-confidence errors remain costly.

The GPU-exported manifest matches training/evaluation fingerprints. Source hashes, examples, labels, prompt partition and all other manifest fields match the local preparation; only activations.pt binary artifact checksum differs. Numerical tensor identity across serializations was not independently proved. No evidence of a label/split mismatch was found.

Future diagnosis should separate prompt transfer from run transfer and evaluate confidence on validation data before any final test. More epochs or free descriptions alone are not established remedies.

Artifacts: test_harmfulness_map.png, train_curves.png, evaluation_confusion.png, evaluation/report.json, exported-results-audit.json. Audit code: scripts/audit_binary_activation_results.py. Plot code: src/behaviour_prediction/explainer/binary_plots.py.
