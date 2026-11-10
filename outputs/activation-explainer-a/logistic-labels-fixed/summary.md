# Logistic monitors with corrected labels

All variants were newly fitted. Layer selection applies during training AND testing. Previous frozen results are preserved.

## Original checkpoint-level monitor

Monitoring activations and their matching monitoring labels; leave one entire run out. No separate-evaluation input claim.

| Behaviour decisions | Both layers (42 summaries) | Layer31 (21 summaries) |
|---|---:|---:|
| Accuracy | 0.793 | 0.802 |
| Balanced accuracy | 0.763 | 0.777 |
| Macro F1 | 0.824 | 0.829 |

Correct counts: 333/420 versus 337/420. Frequency baseline 241/420. Improvement is small on this inspected development set.
Checkpoint binary harmfulness remains saturated: all 35 checkpoints are positive. The monitor and always-positive baseline both score 35/35. This is not evidence of useful binary discrimination.

## Per-response controls matched to the decoder split

775 training, 155 validation, 175 held-out test responses; validation selects regularization. The test run and prompt IDs are held out. These percentages must not be mixed with checkpoint accuracy.

| Inputs | Both accuracy | Layer31 accuracy | Both balanced accuracy | Layer31 balanced accuracy |
|---|---:|---:|---:|---:|
| full_vector_linear | 73.1% | 67.4% | 63.2% | 52.1% |
| norms_linear | 60.0% | 62.9% | 64.9% | 70.5% |

Layer31 norm inputs are very sensitive (50/55 harmful detected) but yield 60 false alarms on 120 negatives. Both-layer norms detect 43/55 with 58 false alarms. Raw-vector logistic regression gets worse with layer31-only input (6/55 detected versus 20/55 with both). No representation is uniformly better.

## Decoder rerun status

New corrected-label data are prepared; new decoder GPU training has not been run locally. Run scripts/run_fixed_label_readers.sh on GPU to train BOTH variants afresh on matched corrected data. Three epochs each; old results untouched. Commands and result transfer are in docs/corrected-label-layer31-monitors.md.
