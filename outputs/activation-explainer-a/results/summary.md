# Experiment A status

Prepared original activation/response pairs; held-out run self_harm; validation run child_abuse.
Existing provisional Claude labels are verbalized into short template descriptions. Refusal versus safe assistance and causal explanations are not supervised.

| Split | Responses | Harmful | Non-harmful |
|---|---:|---:|---:|
| train | 772 | 115 | 657 |
| validation | 152 | 55 | 97 |
| test | 175 | 55 | 120 |
| baseline_test | 35 | 1 | 34 |

Decoder GPU evaluation: available

Class balancing is applied to training only; the test distribution is unchanged.
Fourteen behaviour tags are scored, with tags absent from training reported separately.
Targets describe the saved, potentially token-limited responses, not unlimited future generations.
The same test split was used for controls. It is an inspected development pilot, not a new independent benchmark.

| Method | Harmfulness accuracy | Balanced accuracy | Harmful recall | Behaviour macro F1 (training-supported) |
|---|---:|---:|---:|---:|
| full_vector_linear | 0.697 | 0.577 | 0.255 | 0.000 |
| norms_linear | 0.611 | 0.658 | 0.782 | 0.000 |
| prompt_bow_linear | 0.703 | 0.532 | 0.073 | 0.000 |
| step_only_linear | 0.617 | 0.593 | 0.527 | 0.000 |
| frequency | 0.686 | 0.500 | 0.000 | 0.000 |
| always_absent | 0.686 | 0.500 | 0.000 | 0.000 |
| always_present | 0.314 | 0.500 | 1.000 | 0.066 |
| decoder_actual | 0.651 | 0.495 | 0.073 | 0.094 |
| decoder_baseline_only | 0.686 | 0.500 | 0.000 | 0.000 |
| decoder_zero | 0.686 | 0.500 | 0.000 | 0.000 |
