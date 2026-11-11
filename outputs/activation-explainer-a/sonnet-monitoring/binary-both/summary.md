# Binary activation-injection diagnostic

One 0/1 next-token decision, with no descriptions or gold label in the input.
The four activation vectors and original dataset split are retained.
Conditional 0/1 probabilities are uncalibrated. Full-vocabulary greedy validity is reported separately.

overfit: completion metadata not available locally.

train: 144 optimizer updates; overfit pass = None.


## evaluation

| Split | Input | Cases | Accuracy | Balanced accuracy | Harmful recall | Greedy digit valid |
|---|---|---:|---:|---:|---:|---:|
| baseline_test | actual | 35 | 0.857 | 0.441 | 0.000 | 1.000 |
| baseline_test | baseline_only | 35 | 0.857 | 0.441 | 0.000 | 1.000 |
| baseline_test | zero | 35 | 0.971 | 0.500 | 0.000 | 1.000 |
| test | actual | 175 | 0.720 | 0.761 | 0.851 | 1.000 |
| test | baseline_only | 175 | 0.691 | 0.506 | 0.106 | 1.000 |
| test | zero | 175 | 0.731 | 0.500 | 0.000 | 1.000 |
| train | actual | 766 | 0.867 | 0.903 | 0.955 | 1.000 |
| validation | actual | 154 | 0.805 | 0.834 | 0.925 | 1.000 |
| validation | baseline_only | 154 | 0.662 | 0.563 | 0.245 | 1.000 |
| validation | zero | 154 | 0.656 | 0.500 | 0.000 | 1.000 |
| baseline_test | always_non_harmful | 35 | 0.971 | 0.500 | 0.000 | n/a |
| baseline_test | always_harmful | 35 | 0.029 | 0.500 | 1.000 | n/a |
| baseline_test | training_frequency | 35 | 0.971 | 0.500 | 0.000 | n/a |
| test | always_non_harmful | 175 | 0.731 | 0.500 | 0.000 | n/a |
| test | always_harmful | 175 | 0.269 | 0.500 | 1.000 | n/a |
| test | training_frequency | 175 | 0.731 | 0.500 | 0.000 | n/a |
| train | always_non_harmful | 766 | 0.856 | 0.500 | 0.000 | n/a |
| train | always_harmful | 766 | 0.144 | 0.500 | 1.000 | n/a |
| train | training_frequency | 766 | 0.856 | 0.500 | 0.000 | n/a |
| validation | always_non_harmful | 154 | 0.656 | 0.500 | 0.000 | n/a |
| validation | always_harmful | 154 | 0.344 | 0.500 | 1.000 | n/a |
| validation | training_frequency | 154 | 0.656 | 0.500 | 0.000 | n/a |

Interpretation: passing the 32-example overfit gate checks memorization capacity only. It is not evidence of generalization.
A failed gate may reflect learning rate, update budget, noisy/inconsistent data or the injection setup; it is not a definitive bug diagnosis.
Compare full training accuracy with validation accuracy before interpreting test results. Test labels never select a checkpoint or threshold.
Zero and baseline-only injections are ablations of the trained reader; their effects alone do not establish causal explanations.
