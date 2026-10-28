# Separate evaluation prompts

30 prompts: 10 harmful requests, 10 indirectly harmful requests, and 10 benign
requests. All IDs and prompt texts differ from the 30 monitoring probes. Situations
include online groups, business reviews, borrowed property, conferences, clubs,
classes, youth activities, and constructive criticism. Benign requests discuss
similar subjects without requesting harmful behaviour. A prompt's group does not
determine the output label.

Topics cover self-harm (encouragement and facilitation), hate speech, threats,
discrimination, child abuse, and harassment. Counts per topic differ from the
monitoring suite so harassment is explicitly tested. Some scenarios remain
semantically related because the behaviour categories are the same. This is a
test of transfer to new prompts, not a claim of unrelated-task transfer.

`protocol.json` records this suite's hashes, the current judging rubric, saved
held-out monitor predictions, 0.5 threshold, steps, and generation settings before
new outputs are observed. Do not change the suite or judging rubric based on
evaluation results; create a new version for a changed experiment.

Generate responses using `bash scripts/generate_evaluation.sh` on the GPU.
Use these questions only for outcome evaluation: do not replace the existing
monitoring probes or capture new activations on them. No LLM or monitor retraining
is needed. Seven runs x six checkpoints x thirty prompts = 1,260 response rows.

After labeling, compare saved predictions with new outcomes via
`python scripts/evaluate_new_prompts.py score`. The evaluator performs no fitting,
checks that source predictions are unchanged, and compares frequency/persistence
controls as appropriate. Known model runs still provide few independent units.
