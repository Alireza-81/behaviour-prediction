# Corrected-label decoder reruns: interpretation

Both readers were retrained from the original Qwen base using the corrected data: 775 training, 155 validation, 175 test responses and 35 baseline checks. No missing labels remain in these cohorts.

| Test metric | Layers 15+31 | Layer31 only |
|---|---:|---:|
| Correct / total | 134/175 | 134/175 |
| Accuracy | 76.6% | 76.6% |
| Balanced accuracy | 68.1% | 78.0% |
| Harmful detected | 25/55 | 45/55 |
| Harmful missed | 30 | 10 |
| False alarms | 11/120 | 31/120 |
| Harmful precision | 69.4% | 59.2% |
| Harmful recall | 45.5% | 81.8% |
| F1 | .549 | .687 |
| Brier | .215 | .172 |
| Binary log loss | 1.139 | .560 |
| ROC AUC, recorded conditional scores | .845 | .852 |

Layer31 has 20 more true positives and 20 more false positives at the same nominal 0.5 threshold. Both readers rank cases similarly (AUCs close on one inspected split), but have different operating behaviour. This is not proof of statistical equivalence or a universal layer preference.

Layer31 validation balanced accuracy by epoch: 86.2%, 68.6%, 84.0%. The selected checkpoint is update 49, epoch 1. Two-layer validation balanced accuracy: 83.5%, 76.0%, 84.2%; selected update 147. These selection outcomes were verified from validation history only. More epochs are not established as the next fix for layer31.

In the previous-label runs, two-layer accuracy was 74.9% and layer31 accuracy 80.6%. After label repair the two-layer run improves accuracy to 76.6%, while layer31 accuracy falls to 76.6% but its balanced accuracy rises (74.5% to 78.0%). The test binary labels are unchanged. Training/validation coverage, sampling trajectory and checkpoint selection changed; this does not isolate a pure label-quality causal effect.

Confidence remains uncalibrated. There are 19 >99%-confidence errors among 41 wrong test predictions for both layers, versus one among 41 for layer31. Better Brier/log loss for layer31 is encouraging but not calibrated safety assurance.

The next useful investigation is validation-only threshold/calibration choice for an explicit false-alarm budget, then assessment on further held-out data. No threshold was selected using the ROC plot here. Test data have already been inspected repeatedly, and each variant has one training seed.

Verification: all 1,870 predictions per reader, exact target IDs/labels, all reported metrics, GPU example indices and validation checkpoint selection were checked. See each run's exported-results-audit.json. The layer31 manifest legitimately differs by feature-selection lineage and masked tensor artifact.

Artifacts: comparison.png, roc_comparison.png, analysis-audit.json and ../test_harmfulness_map.png. Reproduce this analysis with scripts/analyze_fixed_label_readers.py.
