# Full Qwen v2 label audit

All 9,504 response records match their prompt manifests and exported labels. Labels remain provisional and have not been modified.

| Suite | Positive | Negative | Unresolved |
|---|---:|---:|---:|
| monitoring | 1238 | 3499 | 15 |
| evaluation | 1222 | 3507 | 23 |

Conflicting identical prompt/answer payloads: 1 groups, 12 rows.
See conflicting_labels.csv and unresolved_labels.csv for the exact answers.

Missing activation/metadata files: 0.

The response-rate plot uses each checkpoint's decided response denominator. checkpoint_rates.csv also contains lower/upper rates allowing unresolved labels to be negative/positive, and counts of new positives on baseline-negative prompts.

Monitoring step zero has one weapon-creation positive in every run. The evaluation baseline organized-crime answer has inconsistent labels across runs. A binary any-positive target therefore requires care before claiming early warning.
