# Fixed-decoder transfer comparison

Same trained weights, original training normalization and fixed 0.5 threshold. No decoder retraining.
Training/validation references are development diagnostics. New-model tests use self_harm; new-prompt tests use the withheld monitoring prompt pool, not the separate evaluation file.

| Cohort | Cases | Accuracy | Balanced accuracy | Harmful recall | Always-negative accuracy |
|---|---:|---:|---:|---:|---:|
| training_reference | 772 | 94.6% | 96.4% | 99.1% | 85.1% |
| new_prompts_known_models | 874 | 89.0% | 80.1% | 64.7% | 78.9% |
| known_prompts_new_model | 155 | 83.9% | 78.5% | 68.6% | 77.4% |
| new_prompts_new_model | 175 | 74.9% | 66.9% | 45.5% | 68.6% |
| validation_reference | 152 | 84.9% | 82.2% | 72.7% | 63.8% |
| selection_model_new_prompts | 174 | 74.1% | 70.8% | 51.4% | 58.6% |

See report.json for per-model, per-step and probe-class breakdowns. Some cohorts/models can have no positive labels, making balanced accuracy undefined.
These are inspected development comparisons. Prompt pools differ in content and related checkpoints are correlated; changes are not causal estimates.
