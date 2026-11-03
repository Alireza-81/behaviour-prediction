# Expanded activation patch study

25 harmful-request prompts and 5 benign requests; 180 generated answers.

| Condition | Harmful unpatched target answers repaired | New harm on previously safe targets | Harmful benign-control answers |
|---|---:|---:|---:|
| baseline_step0 | Reference only | 0/1 | 0/5 |
| unpatched_step20 | 0/24 | 0/1 | 0/5 |
| self_patch_both | 0/24 | 0/1 | 0/5 |
| patch_layer15 | 3/24 | 0/1 | 0/5 |
| patch_layer31 | 9/24 | 1/1 | 0/5 |
| patch_both | 10/24 | 1/1 | 0/5 |

Harmful target-answer counts by condition: baseline_step0: 1/25, unpatched_step20: 24/25, self_patch_both: 24/25, patch_layer15: 21/25, patch_layer31: 16/25, patch_both: 15/25.
The combined patch fixes 10 harmful answers but introduces 1 newly harmful answer(s), for a net reduction of 9.
Fine-tuning-acquired harm subset (baseline non-harmful → unpatched harmful): 10/23 repaired by the combined patch.
Truncated responses: 30/180. Conclusions apply to visible text. Use paired_decisions.csv to review changed answers; lack of harmfulness is not a measurement of answer quality.

Identical-output binary label disagreements: 0.
One fine-tuned model, selected behaviour, development study
Harmful-request prompts do not guarantee harmful answers
Only freshly harmful unpatched target answers count as repair opportunities
Acquired-harm subset excludes target answers already harmful at baseline
Five benign controls provide only a small side-effect check, not a general capability evaluation
Judge disagreement on identical outputs must be adjudicated before relying on small effects
Only final prompt-token residual vectors are patched; later generated-token positions are untouched
