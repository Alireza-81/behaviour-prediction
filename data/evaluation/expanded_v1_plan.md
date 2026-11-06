# Expanded monitoring and evaluation prompts

- Monitoring: `data/probes/expanded_v1.jsonl` (50 prompts).
- Separate evaluation: `data/evaluation/expanded_v1.jsonl` (50 prompts).
- Each suite contains 12 direct and 24 indirect requests, one direct and two
  indirect per category, plus six topic-matched benign requests and eight
  standalone benign requests. Dropping matched controls leaves eight benign
  prompts in each suite.
- Coverage is all 11 harmful dataset categories in `data/raw`, plus harassment.
  Adult content is a dataset category. Use a separately defined labeling rule
  for explicit adult content before combining it with the binary harm metric.
  Self-harm prompts cover both encouragement and facilitation.
- The indirect prompts use scenarios such as fictional dialogue, editing and
  role-play. They are single-turn prompts.
- The suites have separate IDs and wording. Keep evaluation prompts out of
  monitoring inputs and monitor fitting. The existing 30-prompt manifests
  remain available for reproducing earlier results.

## Labeling budget

For eight checkpoints per run (0, 1, 2, 3, 5, 7, 10, 20) and 100 prompts
combined, a 12-run design yields 9,600 generations before any reuse. A possible
allocation is 11 harmful-category fine-tunes and one benign control.
Harassment is probed in every run and currently has no raw training dataset.
If an existing trained model can be included in this design, count it within
the 12-run allocation. Count pilot generations and reruns against the same
10,000 ceiling. A thirteenth full run would exceed it.

This balances the number of requests per harmful category. The actual mix of
harmful and non-harmful answers depends on the model and checkpoints. Record
baseline harm separately and check onset on the prompts that were non-harmful
at step 0. If selecting a starting model using prompt results, use development
prompts for that selection and reserve the evaluation suite for scoring.
