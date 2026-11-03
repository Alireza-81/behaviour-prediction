# Independent monitor-input experiments

Same saved activations, original ground-truth labels, whole-model holdouts, logistic regression, L2=1, and threshold 0.5. No GPU capture or API labeling was needed.
The reproduced 42-feature baseline matches every saved baseline prediction (scores within 1e-6). Separate evaluation labels are never used to fit, scale, or select settings.
Both variants are evaluated independently; no model combines unpooled probes and added history.

## Experiment 1: no averaging across probes

240 features: 30 probes x 2 layers x 4 individual measurements (L2, relative L2, cosine distance, speed). No class averages, cross-probe standard deviations, or class contrasts are supplied. This still uses scalar activation measurements, not full activation vectors. A within-class standard deviation has no single-probe equivalent.

## Experiment 2: history

128 features: current 42 summary measurements, the preceding 42, the second-preceding 42, and two history-availability flags. Missing early history is zero-padded. Step zero can be used as past context, but is not a fitting/scoring example. All earlier history comes from the same run; no future snapshots or labels enter an input.
The original monitor already has one-step movement speed; this experiment adds lagged feature vectors. It tests a fixed three-checkpoint window with logistic regression, not all possible temporal architectures.
A 44-feature timing control adds only the two availability flags to the original features, without the lagged measurements. This checks the effect of implicitly revealing early-versus-late training.

## Behaviour identification

| Input | Original-prompt labels | Separate-prompt labels | Separate-prompt macro F1 |
|---|---:|---:|---:|
| baseline | 205/245 (83.7%) | 191/245 (78.0%) | 0.829 |
| no_averaging | 208/245 (84.9%) | 192/245 (78.4%) | 0.827 |
| history | 195/245 (79.6%) | 185/245 (75.5%) | 0.802 |
| history_timing_control | 204/245 (83.3%) | 190/245 (77.6%) | 0.826 |
| frequency | 147/245 (60.0%) | 115/245 (46.9%) | 0.462 |

245 decisions = 35 checkpoints x 7 behaviours, not 245 independent runs.

| Behaviour, separate questions | Baseline | No averaging | History |
|---|---:|---:|---:|
| self_harm_encouragement | 29/35 | 27/35 | 30/35 |
| self_harm_facilitation | 28/35 | 28/35 | 27/35 |
| hate_speech | 27/35 | 31/35 | 25/35 |
| violent_threats | 31/35 | 30/35 | 32/35 |
| discrimination | 29/35 | 29/35 | 28/35 |
| child_abuse | 31/35 | 31/35 | 29/35 |
| harassment | 16/35 | 16/35 | 14/35 |

## Interpretation

no_averaging: fixed 10 baseline errors and introduced 9 new errors on separate-prompt behaviour decisions.
history: fixed 4 baseline errors and introduced 10 new errors on separate-prompt behaviour decisions.
No averaging changes false alarms/misses from 19/35 to 13/40: fewer false alarms but more missed behaviours. Its one-extra-correct result is not a clear overall improvement; macro F1 slightly declines. All-seven-behaviours-correct checkpoints remain 13/35.
History changes false alarms/misses to 16/44; it underperforms both the baseline and timing control. This does not prove that history or nonlinear sequence models cannot help with more data.

## Binary harm detection and forecasting

| Input | Current harm: original | Current harm: separate | Forecast: original | Forecast: separate |
|---|---:|---:|---:|---:|
| baseline | 31/35 (88.6%) | 34/35 (97.1%) | 26/28 (92.9%) | 28/28 (100.0%) |
| no_averaging | 31/35 (88.6%) | 30/35 (85.7%) | 26/28 (92.9%) | 28/28 (100.0%) |
| history | 31/35 (88.6%) | 34/35 (97.1%) | 26/28 (92.9%) | 28/28 (100.0%) |
| frequency | 30/35 (85.7%) | 35/35 (100.0%) | 26/28 (92.9%) | 28/28 (100.0%) |

Original-label binary detection: no averaging reduces false alarms from 4 to 2 but introduces two missed harmful checkpoints, leaving accuracy unchanged at 31/35. On separate labels it misses five harmful checkpoints rather than one.
All separate-evaluation checkpoints are harm-positive, so binary and forecast accuracy cannot establish negative-case discrimination. All variants match the frequency forecast (26/28 original and 28/28 separate). No early-warning improvement is demonstrated.

## Limits and saved artifacts

Seven independent runs with correlated checkpoints; class imbalance; fixed regularization across different feature counts; API labels with user corrections. The separate suite has already been inspected, so these are exploratory development comparisons, not an untouched final validation. No hyperparameter search was performed.
Run scripts/compare_monitor_inputs.py to reproduce into a fresh output directory. Implementation is src/behaviour_prediction/monitor/comparisons.py. The fitted small monitors, input feature names, source hashes, held-out predictions, full reports, per-run metrics, and paired changed decisions are saved beside this summary.
