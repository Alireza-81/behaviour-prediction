# Reverse activation patching

Donor: step20 hate_speech model. Recipient: step-zero model. Same 30 prompts and initial last-prompt-token intervention as the forward study.

| Condition | Harmful target answers | Newly harmful among baseline-safe targets | Transfers where donor harmful and baseline safe | Harmful benign answers |
|---|---:|---:|---:|---:|
| donor_step20 | 24/25 | 23/24 | Reference | 0/5 |
| unpatched_step0 | 1/25 | 0/24 | 0/23 | 0/5 |
| self_patch_both | 1/25 | 0/24 | 0/23 | 0/5 |
| patch_layer15 | 1/25 | 0/24 | 0/23 | 0/5 |
| patch_layer31 | 19/25 | 18/24 | 17/23 | 0/5 |
| patch_both | 20/25 | 19/24 | 18/23 | 0/5 |

Identical-output binary label disagreements: 0. Truncated outputs: 12.

Exclude already-harmful baseline answers from new-harm counts. A positive result shows these injected states can promote harm in this context; it does not identify uniquely necessary neurons. Final-block injection can directly copy donor first-token logits. A negative result is limited to the final prompt-token intervention and does not establish irrelevance of other positions or components. Five benign prompts are only a small side-effect check. This prompt set has already been inspected: development evidence, not untouched validation.
