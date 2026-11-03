# Early-step data overview

Seven runs; steps 0, 1, 2, 3, 5, 7, 10, 20. Each suite has 30 prompts per checkpoint.

| Suite | Responses | Harmful | Non-harmful | Unresolved |
|---|---:|---:|---:|---:|
| monitoring | 1680 | 103 | 1577 | 0 |
| evaluation | 1680 | 198 | 1482 | 0 |

Missing activation/metadata files: 0. All expected files are present.

Forecasting should use adjacent saved checkpoints, excluding step zero as an input: 1→2, 2→3, 3→5, 5→7, 7→10, 10→20. These have unequal time gaps, so report each transition separately as well as combined.

Fit only the original 42-summary logistic baseline with entire runs held out. Fit against monitoring labels; score the same predictions against the separate evaluation labels without refitting. No vector, history or neural-network experiments.

The plot shows observed response labels, not trained monitor performance.
