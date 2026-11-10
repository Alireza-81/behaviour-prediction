# Binary activation-injection diagnostic

One 0/1 next-token decision, with no descriptions or gold label in the input.
The four activation vectors and original dataset split are retained.
Conditional 0/1 probabilities are uncalibrated. Full-vocabulary greedy validity is reported separately.

overfit: completion metadata not available locally.

train: completion metadata not available locally.


## evaluation

| Split | Input | Cases | Accuracy | Balanced accuracy | Harmful recall | Greedy digit valid |
|---|---|---:|---:|---:|---:|---:|
| baseline_test | actual | 35 | 0.971 | 0.500 | 0.000 | 1.000 |
| baseline_test | baseline_only | 35 | 0.971 | 0.500 | 0.000 | 1.000 |
| baseline_test | zero | 35 | 0.971 | 0.500 | 0.000 | 1.000 |
| test | actual | 175 | 0.806 | 0.745 | 0.582 | 1.000 |
| test | baseline_only | 175 | 0.686 | 0.500 | 0.000 | 1.000 |
| test | zero | 175 | 0.686 | 0.500 | 0.000 | 1.000 |
| train | actual | 772 | 0.909 | 0.947 | 1.000 | 1.000 |
| validation | actual | 152 | 0.803 | 0.763 | 0.618 | 1.000 |
| validation | baseline_only | 152 | 0.632 | 0.503 | 0.036 | 1.000 |
| validation | zero | 152 | 0.638 | 0.500 | 0.000 | 1.000 |
| baseline_test | always_non_harmful | 35 | 0.971 | 0.500 | 0.000 | n/a |
| baseline_test | always_harmful | 35 | 0.029 | 0.500 | 1.000 | n/a |
| baseline_test | training_frequency | 35 | 0.971 | 0.500 | 0.000 | n/a |
| test | always_non_harmful | 175 | 0.686 | 0.500 | 0.000 | n/a |
| test | always_harmful | 175 | 0.314 | 0.500 | 1.000 | n/a |
| test | training_frequency | 175 | 0.686 | 0.500 | 0.000 | n/a |
| train | always_non_harmful | 772 | 0.851 | 0.500 | 0.000 | n/a |
| train | always_harmful | 772 | 0.149 | 0.500 | 1.000 | n/a |
| train | training_frequency | 772 | 0.851 | 0.500 | 0.000 | n/a |
| validation | always_non_harmful | 152 | 0.638 | 0.500 | 0.000 | n/a |
| validation | always_harmful | 152 | 0.362 | 0.500 | 1.000 | n/a |
| validation | training_frequency | 152 | 0.638 | 0.500 | 0.000 | n/a |

Interpretation: passing the 32-example overfit gate checks memorization capacity only. It is not evidence of generalization.
A failed gate may reflect learning rate, update budget, noisy/inconsistent data or the injection setup; it is not a definitive bug diagnosis.
Compare full training accuracy with validation accuracy before interpreting test results. Test labels never select a checkpoint or threshold.
Zero and baseline-only injections are ablations of the trained reader; their effects alone do not establish causal explanations.
