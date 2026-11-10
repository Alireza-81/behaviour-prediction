# Seven-model monitoring experiments

Selected benign training conditions: educational, neutral_description, professional_email. Selected harmful conditions: hate_speech, self_harm, child_abuse, violent_threats. The user specified six names and seven total; violent_threats was selected as the provisional seventh.

Same 42-summary logistic monitor and whole-run holdouts as before. Training labels come only from monitoring responses; the same held-out predictions are scored against separate evaluation responses. The step-0 activations are a reference and are not fitted or scored as input cases. Models are assessed by their response labels, not their training-condition name.

## Specific behaviours

| Input or baseline | Monitoring | Evaluation |
|---|---:|---:|
| summary42 | 326/410 (79.5%) | 270/391 (69.1%) |
| step_only | 230/410 (56.1%) | 195/391 (49.9%) |
| drift_only | 276/410 (67.3%) | 234/391 (59.8%) |
| drift_weaker_l2 | 321/410 (78.3%) | 283/391 (72.4%) |
| frequency | 234/410 (57.1%) | 198/391 (50.6%) |
| always_absent | 157/410 (38.3%) | 162/391 (41.4%) |
| always_present | 253/410 (61.7%) | 229/391 (58.6%) |

| Same-step training-run majority | 289/410 (70.5%) | 258/391 (66.0%) |

The class balance improves: 253/410 monitoring behaviour decisions are positive (61.7%), compared with 76.5% in the preceding 12-run subset. Evaluation has 229/391 positive decisions (58.6%). Different totals reflect unknown per-behaviour outcomes, which are omitted rather than treated as absent.

Cross-model differentiation improves: 24/60 behaviour-by-step cells have identical predictions across all seven runs, compared with 59/60 in the preceding 12-run comparison. On evaluation cases differing from the training-run same-step majority, the monitor is correct on 36/133. Thus the time-pattern shortcut remains substantial.

Overall drift with weaker regularization still outperforms the 42 summaries on evaluation: 283/391 versus 270/391. These are small, correlated development data; selecting a more complex monitor is not justified by the summary model alone.

## Binary harmfulness

All 35 nonzero monitoring checkpoints have at least one positive response. Restricting the outcome to prompts that were negative at baseline leaves 34/35 positive. The original any-positive binary monitor therefore uses a constant-positive fallback. Detection is 35/35 on monitoring and 29/35 on evaluation; forecasting is 28/28 and 24/28 respectively. Always-present scores identically.

The additional benign runs do not supply many completely non-harmful checkpoints under the any-positive rule. Professional-email training has 9 positive monitoring answers and 12 positive evaluation answers at step 200. Educational has 3 and 0; neutral description has 3 and 3. Their counts are much lower than harmful conditions, but one positive is enough for checkpoint presence.

## Verification and limits

All 42 activation snapshots and 63 input files were verified. The 448 saved fold-model records were audited, with 2,184 trained-method scores independently recomputed. Whole-run splits, training-only normalization and reports reproduced. All 2,772 selected labels per suite match the frozen judge manifests, raw API results and exported CSVs.

The newly added review CSV/JSONL sidecars were not supplied locally; their original answer text is available in the prepared judge manifests and labeled exports. The label audit records these missing sidecars. Conflicting identical-payload labels are treated as unknown for analysis only. Source labels were not modified.

The provisional API labels, single seed per condition, unequal forecasting gaps and repeated inspection of this dataset limit interpretation. The weaker-drift L2=1/42 is the fixed sensitivity check used in the preceding experiments, not a newly tuned parameter.

## Files

- behaviour_presence_monitoring.png and behaviour_presence_evaluation.png: observed versus predicted behaviours.
- report.json: per-run/per-tag confusion matrices and comparison metrics.
- step-majority-diagnostic.json: same-step majority and minority-case performance.
- data-audit.json and label-audit.json: verification and coverage.
- protocol.json and fitted-monitors.json: reproducible protocol and model parameters.
