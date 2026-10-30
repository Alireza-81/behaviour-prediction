# Harassment/threat evaluation activations versus benign controls

The saved early-step activations currently cover the original monitoring prompts only. The evaluation CSV files contain answers and labels, not their activation tensors. The previous behaviour-associated monitoring plots cannot substitute for actual evaluation-question activations.

On the GPU machine, from the project directory, after syncing the new script:

```bash
bash scripts/capture_evaluation_activations.sh
```

This captures all 30 evaluation questions at layers 15/31 for the base model (step 0) and retained checkpoint 20 of each of seven runs. It does not train, generate answers, or call Claude. It writes to a separate `outputs/early-steps/evaluation-activations` folder and refuses to overwrite an existing capture.

If more checkpoints still exist, edit `--steps 0 20` to include those actual saved steps before running. Step 1/2/3/5/7/10 models were configured to be removed as training progressed. Saved monitoring activation tensors and response text cannot reconstruct their weights or their evaluation activations. A full trajectory would require those retained models or a new training run capturing evaluation activations inline; do not start another training run merely to obtain the endpoint comparison.

Copy only the new activation snapshots and metadata back to the local checkout. If using Git:

```bash
git add -f \
  outputs/early-steps/evaluation-activations/*/activation_monitoring/step-*.pt \
  outputs/early-steps/evaluation-activations/*/run_metadata.json

git diff --cached --stat
```

After committing/pushing and pulling locally, plot with:

```bash
python scripts/plot_evaluation_activation_comparison.py
```

Output: `outputs/early-steps/evaluation-activation-plots/`.

For each model and layer, compare two direct harassment prompts, two indirect harassment prompts, two direct threat prompts, two indirect threat prompts, same-topic benign prompts (two harassment controls and one threat control), and all ten benign evaluation prompts.

Plots distinguish movement from each prompt's own step-zero baseline from distance to the same-step benign centroid. The latter shows prompt-dependent differences that may already exist before training, when the indirect evaluation prompts already elicited harmful outputs. These are last prompt-token activations, not generated-answer activations; distances alone are not harmfulness measures.
