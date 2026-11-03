# Full activation vectors and a small nonlinear monitor

Two independent experiments. Full vectors use a linear classifier; the neural network uses only the original 42 summaries. They are never combined.
Same seven whole-run holdouts, original training labels, and 0.5 threshold. Separate-prompt labels never enter fitting. The saved baseline was reproduced before accepting the comparison.

| Input/model | Original behaviour labels | Separate behaviour labels | Separate macro F1 |
|---|---:|---:|---:|
| baseline | 205/245 (83.7%) | 191/245 (78.0%) | 0.829 |
| full_vectors | 202/245 (82.4%) | 192/245 (78.4%) | 0.825 |
| mlp_seed42 | 207/245 (84.5%) | 187/245 (76.3%) | 0.812 |
| frequency | 147/245 (60.0%) | 115/245 (46.9%) | 0.462 |

## Experiment 1: full activation-change vectors

Each checkpoint supplies 245,760 signed coordinates (30 probes x 2 layers x 4096 hidden dimensions). Each probe delta is divided by its initial activation norm, without averaging probes. Train-only centering and a single scale preserve the geometry; average squared training-vector norm is fixed to 42. No PCA or feature selection is used.
L2 logistic regression is computed through exact dot products between vectors. This is a computational shortcut for a full-vector linear classifier, not a nonlinear kernel or a lossy PCA representation. The few training examples still limit the directions that can be learned.

## Experiment 2: small neural network

42 inputs -> 8 tanh hidden units -> one sigmoid output, fitted separately for each behaviour. Adam lr 0.01, 300 steps, L2 coefficient 0.1 on weights; normalization uses training rows only. Seed 42 is the primary result, fixed before scoring. Four additional seeds check sensitivity; no best seed is selected. Compared with logistic regression, optimization and regularization also differ, so this is a comparison of fixed training recipes.

| MLP seed | Original correct | Separate correct |
|---|---:|---:|
| 42 | 207/245 (84.5%) | 187/245 (76.3%) |
| 0 | 207/245 (84.5%) | 187/245 (76.3%) |
| 1 | 207/245 (84.5%) | 187/245 (76.3%) |
| 2 | 207/245 (84.5%) | 187/245 (76.3%) |
| 3 | 207/245 (84.5%) | 187/245 (76.3%) |

| Behaviour, separate questions | Baseline | Full vectors | MLP seed 42 |
|---|---:|---:|---:|
| self_harm_encouragement | 29/35 | 30/35 | 30/35 |
| self_harm_facilitation | 28/35 | 29/35 | 28/35 |
| hate_speech | 27/35 | 23/35 | 27/35 |
| violent_threats | 31/35 | 35/35 | 30/35 |
| discrimination | 29/35 | 29/35 | 28/35 |
| child_abuse | 31/35 | 32/35 | 28/35 |
| harassment | 16/35 | 14/35 | 16/35 |

| Model | Current harm, original | Current harm, separate | Forecast, original | Forecast, separate |
|---|---:|---:|---:|---:|
| baseline | 31/35 (88.6%) | 34/35 (97.1%) | 26/28 (92.9%) | 28/28 (100.0%) |
| full_vectors | 30/35 (85.7%) | 35/35 (100.0%) | 26/28 (92.9%) | 28/28 (100.0%) |
| mlp_seed42 | 31/35 (88.6%) | 30/35 (85.7%) | 26/28 (92.9%) | 28/28 (100.0%) |
| frequency | 30/35 (85.7%) | 35/35 (100.0%) | 26/28 (92.9%) | 28/28 (100.0%) |

Forecasting and binary scores must be compared with frequency/persistence controls. All separate-prompt binary targets are positive, so accuracy cannot show useful negative-case discrimination or early warning. The 245 behaviour decisions are correlated observations from only seven runs.
These are exploratory comparisons, not a new untouched test. There was no tuning on the separate outcomes. Layer comparison is deferred. Next: collect finer early checkpoints and assess per-behaviour rates/change from step zero, not simply the any-harm target.
Runnable entry point: scripts/compare_vectors_and_network.py. Implementation: src/behaviour_prediction/monitor/vector_network.py. Exact settings, hashes, fitted parameters, dot products, predictions and detailed per-run/per-tag metrics are saved alongside this summary.
