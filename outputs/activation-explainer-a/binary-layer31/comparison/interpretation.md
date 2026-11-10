# Retrained layer-31-only result: interpretation

All 1,858 exported scores, matched case IDs/labels and evaluation metrics were verified. This is the separately trained reader, not the old reader with an input removed only at test time.

| Metric on 175 held-out responses | Layers 15 + 31 | Layer 31 only |
|---|---:|---:|
| Correct | 131/175 | 141/175 |
| Accuracy | 74.9% | 80.6% |
| Balanced accuracy | 66.9% | 74.5% |
| Harmful detected | 25/55 | 32/55 |
| Harmful missed | 30 | 23 |
| False alarms | 14 | 11 |
| Harmful precision | 64.1% | 74.4% |
| Harmful recall | 45.5% | 58.2% |

The new reader fixes 17 previous test errors and introduces 7 new errors, for a net gain of 10 correct. It improves both sensitivity and specificity on this particular held-out self_harm/new-prompt set.

Validation is worse: accuracy 84.9% -> 80.3%, balanced accuracy 82.2% -> 76.3%. Training accuracy also falls (94.6% -> 90.9%). This is mixed development evidence, not proof that layer 15 is harmful or layer 31 is always sufficient. The test has been inspected repeatedly and there is one training seed per reader.

Test probability quality improves (Brier .212 -> .166, binary NLL .964 -> .611), but four of the 34 errors still have over 99% conditional confidence. Scores are not calibrated.

Step-10 harmful recall remains 0/4. Test correct counts at steps 10/20/50/100/200 are 31/35,30/35,29/35,26/35,25/35. Input restriction does not resolve all early-detection or cross-prompt failures.

Only new evaluation files and the generated comparison were copied in this pull. train/history.json, training-config.json, best-metrics.json and completed.json are not local. The GPU comparison code checks recipe parity, but independent checkpoint/learning-curve verification needs those logs. Do not decide whether more epochs help from this test result alone.

Artifacts: comparison.png, exported-audit.json, ../test_harmfulness_map.png and ../evaluation/report.json.
