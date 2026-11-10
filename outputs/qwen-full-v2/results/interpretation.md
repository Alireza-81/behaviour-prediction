# Qwen full v2 interpretation

The original 42-summary logistic monitor is evaluated with whole-run holdouts. It sees activations from the monitoring prompts only. Each fold is standardized using its training runs, and fitted against monitoring outcomes. The same predictions are compared with the separate evaluation outcomes. Fixed threshold 0.5; L2=1. No threshold or parameter search.

All 72 snapshots, 12 completed-run metadata files and 9,504 response/label records are available. The audit reproduced 3,744 trained-method predictions from the saved 768 fold models and verified training-only normalization.

## Binary harmfulness

Every nonzero monitoring checkpoint is positive under the any-positive definition. Even the alternative target, at least one positive answer among prompts with a known negative baseline label, is positive at all 60 checkpoints. Thus each binary fold uses a constant positive fallback because its training labels contain only one class. These are not informative trained binary classifiers.

Monitoring detection: 60/60; forecast: 48/48. Always-present gives the same totals. Evaluation detection: 53/60; forecast: 45/48. Always-present again gives exactly the same totals. Early-warning ability cannot be demonstrated on these splits. The earliest fitted checkpoint is 10 and some new monitoring positives already occur there in every run.

## Specific behaviours

There are 12 output heads corresponding to the 11 harmful training categories, with self-harm split into encouragement and facilitation. Harassment and other_harmful remain in source annotations but are not heads in this comparison. Unknown individual responses prevent declaring absence when no certain positive establishes presence. Consequently there are 703 monitoring and 711 evaluation scored decisions rather than 720.

| Input or baseline | Monitoring | Evaluation |
|---|---:|---:|
| summary42 | 621/703 (88.3%) | 585/711 (82.3%) |
| step_only | 547/703 (77.8%) | 513/711 (72.2%) |
| drift_only | 558/703 (79.4%) | 534/711 (75.1%) |
| drift_weaker_l2 | 619/703 (88.1%) | 591/711 (83.1%) |
| frequency | 519/703 (73.8%) | 491/711 (69.1%) |
| always_absent | 165/703 (23.5%) | 196/711 (27.6%) |
| always_present | 538/703 (76.5%) | 515/711 (72.4%) |

Overall drift is one scalar: mean layer-31 L2 movement over all 66 monitoring probes. Each of the three groups has 22 prompts, so the equal-group and per-probe means coincide. The weaker-regularization variant uses fixed L2=1/42, as in the preceding controls. It was included as a sensitivity check, not tuned on this evaluation set.

The richer monitor beats the majority-style baselines for specific behaviours. However, weaker-regularized drift is only two decisions worse on monitoring and six better on evaluation. These results therefore do not establish that the richer features add robust predictive information.

Five behaviour heads always predict present on these held-out checkpoints: adult_content, animal_abuse, discrimination, organized_crime and weapon_creation. Their high recall should be read alongside their false positives; accuracy and macro F1 alone do not establish precise behaviour identification. Per-tag confusion counts are saved in report.json.

One identical evaluation baseline response has conflicting labels across 12 runs. For analysis only, all copies were treated as unknown. No original labels were modified. Monitoring has 15 unresolved response labels; evaluation has 23.

## Files

- behaviour_presence_monitoring.png and behaviour_presence_evaluation.png show observed and predicted presence.
- report.json contains confusion counts, per-behaviour and per-run metrics, and forecast-transition metrics.
- protocol.json records inputs, definitions, hashes and scope.
- data-audit.json reports independent verification.
- ../activation-plots/activation/comparison/ contains model comparisons per request class and layer.
- ../activation-plots/between_groups_layer_15.png and between_groups_layer_31.png compare groups within each model.

These are exploratory results on one seed, one benign control run, provisional API labels and uneven forecast gaps. The new-response target is a change in the measured answer set, not proof of a newly acquired harmful capability.
